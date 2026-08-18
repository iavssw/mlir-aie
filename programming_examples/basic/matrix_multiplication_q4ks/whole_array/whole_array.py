# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""NPU2 whole-array BF16 x native-llama.cpp-Q4_K matrix multiplication."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D
from aie.iron import (
    CompileTime,
    In,
    Kernel,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    StreamDims,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.iron.kernel import ExternalFunction
from aie.utils import config as aie_config
from aie.utils.benchmark import BenchmarkResult, run_iters
from aie.utils.compile import resolve_target_arch
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args, add_trace_arg
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.trace import TraceConfig
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packing import (  # noqa: E402
    ACTIVATION_INPUTS,
    CACHE_MODES,
    COMPUTE_TYPES,
    N_AIE_ROWS,
    Q4KSConfig,
    Q4_K_GROUP,
    bfp16ebs8_to_float,
    b_partition_taps,
    decode_q4_k,
    make_deterministic_native_q4_k,
    float_to_bfp16ebs8,
    prepare_q4ks_weights,
    quantize_activations_int8,
    reference_matmul,
)

KERNEL_SOURCES = {
    "bf16": str(Path(__file__).resolve().parent / "q4ks_bf16.cc"),
    "bfp16": str(Path(__file__).resolve().parent / "q4ks.cc"),
    "int8": str(Path(__file__).resolve().parent / "q4ks.cc"),
}
AIE_KERNEL_INCLUDE = str(Path(__file__).resolve().parents[4] / "aie_kernels")
VERIFY_SCALAR_PRODUCT_THRESHOLD = 1024 * 1024 * 1024


def _device_for(dev: str, columns: int):
    if dev != "npu2" or columns not in (1, 2, 4, 8):
        raise ValueError("matrix_multiplication_q4ks requires NPU2 and 1/2/4/8 columns")
    device = from_name("npu2", n_cols=None)
    if resolve_target_arch(device) != "aie2p":
        raise ValueError("selected device is not AIE2P")
    return device


def _kernels(config: Q4KSConfig, a_ty, b_ty, c_ty):
    name = (
        f"q4ks_{config.compute_type}_{config.m_c}x{config.k}x{config.n}"
        f"_a{config.m_a}.o"
    )
    flags = [
        f"-DDIM_M_C={config.m_c}",
        f"-DDIM_M_A={config.m_a}",
        f"-DDIM_K={config.k}",
        f"-DDIM_N={config.n}",
        f"-DPACKED_TILE_BYTES={config.tile_bytes}",
        f"-DCOMPUTE_{config.compute_type.upper()}",
        f"-I{AIE_KERNEL_INCLUDE}",
    ]
    if config.compute_type == "bf16":
        flags.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
    matmul = ExternalFunction(
        f"q4ks_matmul_{config.compute_type}",
        object_file_name=name,
        source_file=KERNEL_SOURCES[config.compute_type],
        arg_types=[a_ty, b_ty, c_ty, np.int32],
        include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
        compile_flags=flags,
        use_chess=True,
    )
    zero = Kernel("q4ks_zero_bf16", matmul.object_file_name, [c_ty])
    return matmul, zero


def _build_design(
    dev,
    M: int,
    K: int,
    N: int,
    m_c: int,
    m_a: int,
    k: int,
    n: int,
    n_aie_cols: int,
    compute_type: str,
    cache_mode: str,
    activation_input: str,
    cache_k: int,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    if resolve_target_arch(dev) != "aie2p":
        raise ValueError("matrix_multiplication_q4ks is NPU2-only")
    config = Q4KSConfig(
        M=M,
        K=K,
        N=N,
        m_c=m_c,
        m_a=m_a,
        k=k,
        n=n,
        n_aie_cols=n_aie_cols,
        compute_type=compute_type,
        cache_mode=cache_mode,
        activation_input=activation_input,
        cache_k=cache_k,
    )
    # Activation and joint caching require the converter-worker graph.  Keep
    # those modes visible to the capacity/sweep tools, but never silently
    # compile them as the streaming schedule.
    if cache_mode not in ("stream", "l1-weight", "memtile-weight"):
        raise ValueError(
            f"cache mode {cache_mode!r} is a capacity-model experiment; "
            "use stream, l1-weight, or memtile-weight for device compilation"
        )
    if cache_mode == "memtile-weight" and cache_k != K:
        raise ValueError("memtile-weight requires cache_k == K; use joint-slab for K slabs")
    if activation_input != "bf16":
        raise ValueError("preconverted activation inputs are ceiling-model only")

    n_cores = N_AIE_ROWS * n_aie_cols
    n_tiles_per_core = (M // m_c) * (N // n) // n_cores
    n_row_blocks = M // m_c // N_AIE_ROWS
    n_shim_a = min(N_AIE_ROWS, n_aie_cols)
    a_rows_per_shim = N_AIE_ROWS // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(config.prepared_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    A_l2_ty = np.ndarray[(m_c * k * a_rows_per_shim,), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(m_a, k), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(config.packed_rows, k), np.dtype[np.uint8]]
    B_l2_ty = (
        np.ndarray[(config.n_k_tiles * config.tile_bytes,), np.dtype[np.uint8]]
        if cache_mode == "memtile-weight"
        else B_l1_ty
    )
    C_l2_ty = np.ndarray[(m_c * n * N_AIE_ROWS,), np.dtype[bfloat16]]
    C_l1_ty = np.ndarray[(m_c, n), np.dtype[bfloat16]]

    kernel_a_ty = np.ndarray[(m_a * k,), np.dtype[bfloat16]]
    kernel_b_ty = np.ndarray[(config.tile_bytes,), np.dtype[np.uint8]]
    kernel_c_ty = np.ndarray[(m_c * n,), np.dtype[bfloat16]]
    matmul_kernel, zero_kernel = _kernels(config, kernel_a_ty, kernel_b_ty, kernel_c_ty)

    A_l3l2: list[ObjectFifo] = []
    A_l2l1: list[ObjectFifo] = []
    B_l3l2: list[ObjectFifo] = []
    B_l2l1: list[ObjectFifo] = []
    C_l1l2: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    C_l2l3: list[ObjectFifo] = []

    a_to_stream: StreamDims = [
        (config.a_subtiles, m_a * k),
        (k // 8, 8),
        (m_a, k),
        (8, 1),
    ]
    a_from_stream: StreamDims = [
        (k // 8, 64),
        (m_a // 8, 8 * k),
        (64, 1),
    ]
    for shim in range(n_shim_a):
        parent = ObjectFifo(A_l2_ty, name=f"A_L3L2_{shim}", depth=2)
        A_l3l2.append(parent)
        start = shim * a_rows_per_shim
        stop = start + a_rows_per_shim
        children = parent.cons().split(
            [m_c * k * row for row in range(a_rows_per_shim)],
            depths=[2] * a_rows_per_shim,
            obj_types=[A_l1_ty] * a_rows_per_shim,
            names=[f"A_L2L1_{row}" for row in range(start, stop)],
            dims_to_stream=[a_to_stream] * a_rows_per_shim,
            dims_from_stream=[a_from_stream] * a_rows_per_shim,
        )
        A_l2l1.extend(children)

    c_dims: StreamDims = [
        (m_c // 8, 8 * n),
        (8, 8),
        (n // 8, 64),
        (8, 1),
    ]
    for col in range(n_aie_cols):
        parent_b = ObjectFifo(
            B_l2_ty,
            name=f"B_L3L2_{col}",
            depth=1 if cache_mode == "memtile-weight" else 2,
        )
        B_l3l2.append(parent_b)
        B_l2l1.append(
            parent_b.cons().forward(
                obj_type=B_l1_ty,
                depth=1,
                name=f"B_L2L1_{col}",
                repeat_count=n_row_blocks if cache_mode == "memtile-weight" else None,
            )
        )
        parent_c = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=config.c_fifo_depth,
            dims_to_stream=c_dims,
        )
        C_l2l3.append(parent_c)
        children = parent_c.prod().join(
            [m_c * n * row for row in range(N_AIE_ROWS)],
            depths=[config.c_fifo_depth] * N_AIE_ROWS,
            obj_types=[C_l1_ty] * N_AIE_ROWS,
            names=[f"C_L1L2_{col}_{row}" for row in range(N_AIE_ROWS)],
        )
        for row in range(N_AIE_ROWS):
            C_l1l2[row].append(children[row])

    def core_fn(in_a, in_b, out_c, zero, matmul):
        tile_loop = range_(n_tiles_per_core) if n_tiles_per_core > 1 else range(1)
        for _ in tile_loop:
            elem_c = out_c.acquire(1)
            zero(elem_c)
            k_loop = range_(K // k) if K // k > 1 else range(1)
            for _ in k_loop:
                elem_b = in_b.acquire(1)
                for subtile in range(config.a_subtiles):
                    elem_a = in_a.acquire(1)
                    matmul(elem_a, elem_b, elem_c, subtile)
                    in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        N_AIE_ROWS,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1[row].cons(),
                B_l2l1[col].cons(),
                C_l1l2[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
            trace=1 if trace_config and row * n_aie_cols + col == 1 else 0,
        ),
    )
    flat_workers = [worker for row in workers for worker in row]

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m_c * a_rows_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    if cache_mode == "memtile-weight":
        tiles_per_col = config.n_n_tiles // n_aie_cols
        panel_bytes = config.n_k_tiles * config.tile_bytes
        bytes_per_col = tiles_per_col * panel_bytes
        B_tiles = [
            TensorAccessPattern(
                (config.prepared_bytes,),
                offset=col * bytes_per_col + n_round * panel_bytes,
                sizes=[1, config.n_k_tiles, config.packed_rows, config.k],
                strides=[0, config.tile_bytes, config.k, 1],
            )
            for n_round in range(tiles_per_col)
            for col in range(n_aie_cols)
        ]
    else:
        B_tiles = [
            TensorAccessPattern(
                (config.prepared_bytes,),
                offset=tap.offset,
                sizes=list(tap.sizes),
                strides=list(tap.strides),
            )
            for tap in b_partition_taps(config)
        ]
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (m_c * N_AIE_ROWS, n),
        tile_group_repeats=(2, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        prune_step=False,
    )

    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []
    A_prods = [fifo.prod() for fifo in A_l3l2]
    B_prods = [fifo.prod() for fifo in B_l3l2]
    C_conses = [fifo.cons() for fifo in C_l2l3]

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        if cache_mode == "memtile-weight":
            task_group = TaskGroup()
            tiles_per_col = config.n_n_tiles // n_aie_cols
            for n_round in range(tiles_per_col):
                # Load each full-K compressed panel once.  The MemTile DMA
                # replays its K-tile descriptor for every M row block.
                for col in range(n_aie_cols):
                    b_tap = B_tiles[n_round * n_aie_cols + col]
                    B_hs[col].fill(B, tap=b_tap, group=task_group)
                    B_taps.append(b_tap)

                for row_base in range(0, n_row_blocks, 2):
                    current_rows = min(2, n_row_blocks - row_base)
                    for col in range(n_aie_cols):
                        n_tile = n_round * n_aie_cols + col
                        c_tap = TensorAccessPattern(
                            (M, N),
                            offset=(
                                row_base * N_AIE_ROWS * m_c * N + n_tile * n
                            ),
                            sizes=[current_rows, 1, N_AIE_ROWS * m_c, n],
                            strides=[N_AIE_ROWS * m_c * N, 0, N, 1],
                        )
                        C_hs[col].drain(C, tap=c_tap, wait=True, group=task_group)
                        C_taps.append(c_tap)

                        for tile_row in range(current_rows):
                            if col < n_shim_a:
                                a_tap = TensorAccessPattern(
                                    (M, K),
                                    offset=(
                                        (row_base + tile_row)
                                        * N_AIE_ROWS
                                        * m_c
                                        * K
                                        + col * a_rows_per_shim * m_c * K
                                    ),
                                    sizes=[
                                        1,
                                        config.n_k_tiles,
                                        m_c * a_rows_per_shim,
                                        k,
                                    ],
                                    strides=[0, k, K, 1],
                                )
                                A_hs[col].fill(A, tap=a_tap, group=task_group)
                                A_taps.append(a_tap)
            task_group.finish()
            return

        c_index = 0
        tg = TaskGroup()
        for row_base in range(0, n_row_blocks, 2):
            current_rows = min(2, n_row_blocks - row_base)
            for col in range(n_aie_cols):
                C_hs[col].drain(C, tap=C_tiles[c_index], wait=True, group=tg)
                C_taps.append(C_tiles[c_index])
                c_index += 1
                for tile_row in range(current_rows):
                    offset = ((row_base + tile_row) * n_shim_a + col) % len(A_tiles)
                    if col < n_shim_a:
                        A_hs[col].fill(A, tap=A_tiles[offset], group=tg)
                        A_taps.append(A_tiles[offset])
                    B_hs[col].fill(B, tap=B_tiles[col], group=tg)
                    B_taps.append(B_tiles[col])
            if row_base:
                tg.finish()
                tg = TaskGroup()
        tg.finish()

    runtime = Runtime(sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses])
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        if n_aie_cols == 8:
            raise ValueError("trace requires a spare shim column; use fewer than 8 columns")
        program.enable_trace(
            trace_config.trace_size,
            workers=[flat_workers[1]],
            egress_shim_col=n_aie_cols,
        )
    module = program.resolve_program()
    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array_q4ks(
    A: In,
    prepared_B: In,
    C: Out,
    *,
    M: CompileTime[int] = 1024,
    K: CompileTime[int] = 1024,
    N: CompileTime[int] = 2048,
    m_c: CompileTime[int] = 64,
    m_a: CompileTime[int] = 32,
    k: CompileTime[int] = 128,
    n: CompileTime[int] = 64,
    n_aie_cols: CompileTime[int] = 8,
    compute_type: CompileTime[str] = "bf16",
    cache_mode: CompileTime[str] = "stream",
    activation_input: CompileTime[str] = "bf16",
    cache_k: CompileTime[int] = 1024,
    trace_config: CompileTime[TraceConfig | None] = None,
):
    return _build_design(
        iron.get_current_device(),
        M,
        K,
        N,
        m_c,
        m_a,
        k,
        n,
        n_aie_cols,
        compute_type,
        cache_mode,
        activation_input,
        cache_k,
        trace_config,
    )


whole_array = whole_array_q4ks


def generate_taps(**kwargs):
    columns = kwargs.get("n_aie_cols", 8)
    dev = _device_for("npu2", columns)
    iron.set_current_device(dev)
    defaults = dict(
        M=1024,
        K=1024,
        N=2048,
        m_c=64,
        m_a=32,
        k=128,
        n=64,
        n_aie_cols=columns,
        compute_type="bf16",
        cache_mode="stream",
        activation_input="bf16",
        cache_k=kwargs.get("K", 1024),
        trace_config=None,
    )
    defaults.update(kwargs)
    return _build_design(dev, **defaults, generate_taps=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="NPU2 llama.cpp Q4_K matmul")
    add_compile_args(parser, short_dev=None, dev_choices=("npu2",), default_dev="npu2")
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=2048)
    parser.add_argument("--m-c", type=int, default=64)
    parser.add_argument("--m-a", type=int, default=32)
    parser.add_argument("-k", type=int, default=128)
    parser.add_argument("-n", type=int, default=64)
    parser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--compute-type", choices=COMPUTE_TYPES, default="bf16")
    parser.add_argument("--cache-mode", choices=CACHE_MODES, default="stream")
    parser.add_argument("--activation-input", choices=ACTIVATION_INPUTS, default="bf16")
    parser.add_argument("--cache-k", type=int)
    parser.add_argument("--q4-k-file", type=Path)
    parser.add_argument(
        "--verify-mode",
        choices=["auto", "full", "sampled", "none"],
        default="auto",
    )
    parser.add_argument("--verify-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0x4B535F34)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--min-gflops", type=float, default=0.0)
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--benchmark-csv", type=Path)
    add_trace_arg(parser, with_short=False)
    add_benchmark_args(parser, default_warmup=1, default_iters=1)
    return parser


def _config(opts) -> Q4KSConfig:
    return Q4KSConfig(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m_c=opts.m_c,
        m_a=opts.m_a,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        compute_type=opts.compute_type,
        cache_mode=opts.cache_mode,
        activation_input=opts.activation_input,
        cache_k=opts.cache_k,
    )


def _kwargs(opts) -> dict:
    cfg = _config(opts)
    return dict(
        M=cfg.M,
        K=cfg.K,
        N=cfg.N,
        m_c=cfg.m_c,
        m_a=cfg.m_a,
        k=cfg.k,
        n=cfg.n,
        n_aie_cols=cfg.n_aie_cols,
        compute_type=cfg.compute_type,
        cache_mode=cfg.cache_mode,
        activation_input=cfg.activation_input,
        cache_k=cfg.cache_k,
        trace_config=TraceConfig(trace_size=opts.trace_size) if opts.trace_size else None,
    )


def _validate(opts) -> None:
    try:
        _config(opts)
    except (TypeError, ValueError) as error:
        sys.exit(str(error))
    if opts.warmup < 0 or opts.iters < 1 or opts.benchmark_repeats < 1:
        sys.exit("warmup must be >= 0 and iters/repeats must be >= 1")
    if opts.min_gflops < 0 or opts.verify_samples < 1:
        sys.exit("min-gflops must be >= 0 and verify-samples must be >= 1")


def _sample_reference(A, native, cfg, rows, cols):
    q, scales_native, biases_native = decode_q4_k(native, K=cfg.K, N=cfg.N)
    scales = scales_native.astype(bfloat16).astype(np.float32)
    biases = biases_native.astype(bfloat16).astype(np.float32)
    values = np.empty(len(rows), dtype=np.float32)
    if cfg.compute_type == "int8":
        A8, a_scales, a_sums = quantize_activations_int8(A)
        for i, (row, col) in enumerate(zip(rows, cols)):
            stored = 0.0
            for k0 in range(0, cfg.K, cfg.k):
                partial = 0.0
                for group in range(
                    k0 // Q4_K_GROUP, (k0 + cfg.k) // Q4_K_GROUP
                ):
                    ks = slice(group * Q4_K_GROUP, (group + 1) * Q4_K_GROUP)
                    dot = int(
                        A8[row, ks].astype(np.int32)
                        @ q[ks, col].astype(np.int32)
                    )
                    partial += float(a_scales[row, group]) * (
                        scales[group, col] * dot
                        - biases[group, col] * a_sums[row, group]
                    )
                stored = float(np.asarray(stored + partial, dtype=bfloat16))
            values[i] = stored
        return values
    group_index = np.arange(cfg.K) // Q4_K_GROUP
    for i, (row, col) in enumerate(zip(rows, cols)):
        weights = (
            q[:, col].astype(np.float32) * scales[group_index, col]
            - biases[group_index, col]
        ).astype(bfloat16)
        activation = A[row].astype(np.float32)
        if cfg.compute_type == "bfp16":
            activation = bfp16ebs8_to_float(
                float_to_bfp16ebs8(activation)
            )
            rounded_weights = np.empty(cfg.K, dtype=np.float32)
            for k0 in range(0, cfg.K, 8):
                rounded_weights[k0 : k0 + 8] = bfp16ebs8_to_float(
                    float_to_bfp16ebs8(weights[k0 : k0 + 8].astype(np.float32))
                )
            weights = rounded_weights
        stored = 0.0
        for k0 in range(0, cfg.K, cfg.k):
            ks = slice(k0, k0 + cfg.k)
            stored = float(
                np.asarray(
                    stored
                    + activation[ks] @ weights[ks].astype(np.float32),
                    dtype=bfloat16,
                )
            )
        values[i] = stored
    return values


def _verify(opts, cfg, A, native, actual):
    mode = opts.verify_mode
    if mode == "none":
        return {"max_abs": None, "max_rel": None, "nrmse": None}
    if mode == "auto":
        mode = "sampled" if cfg.M * cfg.K * cfg.N > VERIFY_SCALAR_PRODUCT_THRESHOLD else "full"
    if mode == "full":
        expected = reference_matmul(A, native, cfg)
        observed = actual.astype(np.float32)
        reference = expected.astype(np.float32)
    else:
        rng = np.random.default_rng(opts.seed ^ 0x5134)
        rows = rng.integers(0, cfg.M, opts.verify_samples)
        cols = rng.integers(0, cfg.N, opts.verify_samples)
        reference = _sample_reference(A, native, cfg, rows, cols)
        observed = actual[rows, cols].astype(np.float32)
    difference = np.abs(observed - reference)
    denominator = np.maximum(np.abs(reference), 1.0e-12)
    metrics = {
        "max_abs": float(difference.max(initial=0)),
        "max_rel": float((difference / denominator).max(initial=0)),
        "nrmse": float(
            np.sqrt(np.mean(difference * difference))
            / max(np.sqrt(np.mean(reference * reference)), 1.0e-12)
        ),
    }
    np.testing.assert_allclose(observed, reference, rtol=0.05, atol=0.5)
    print(f"Verification passed ({mode}); errors: {metrics}")
    return metrics


def _write_results(opts, result):
    if opts.benchmark_json:
        opts.benchmark_json.parent.mkdir(parents=True, exist_ok=True)
        opts.benchmark_json.write_text(json.dumps(result, indent=2) + "\n")
    if opts.benchmark_csv:
        opts.benchmark_csv.parent.mkdir(parents=True, exist_ok=True)
        with opts.benchmark_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerow(result)


def _run(opts) -> None:
    cfg = _config(opts)
    if opts.q4_k_file:
        native = np.fromfile(opts.q4_k_file, dtype=np.uint8)
        if native.shape != (cfg.native_bytes,):
            raise ValueError(f"Q4_K file has {native.size} bytes; expected {cfg.native_bytes}")
        rng = np.random.default_rng(opts.seed)
        A = rng.uniform(-0.25, 0.25, size=(cfg.M, cfg.K)).astype(bfloat16)
    else:
        A, native = make_deterministic_native_q4_k(cfg, seed=opts.seed)
    start = time.perf_counter()
    prepared = prepare_q4ks_weights(native, cfg, "q4")
    prepare_ms = (time.perf_counter() - start) * 1000
    A_tensor = iron.tensor(A.reshape(-1), dtype=bfloat16, device="npu")
    B_tensor = iron.tensor(prepared, dtype=np.uint8, device="npu")
    C_tensor = iron.zeros(cfg.M * cfg.N, dtype=bfloat16, device="npu")
    rounds = []
    average_us = []
    for index in range(opts.benchmark_repeats):
        bench: BenchmarkResult = run_iters(
            whole_array_q4ks,
            A_tensor,
            B_tensor,
            C_tensor,
            **_kwargs(opts),
            warmup=opts.warmup,
            iters=opts.iters,
        )
        if bench.npu is None:
            raise RuntimeError("runtime returned no NPU timing")
        gflops = 2.0 * cfg.M * cfg.K * cfg.N / (1000.0 * bench.npu.avg_us)
        rounds.append(gflops)
        average_us.append(bench.npu.avg_us)
        print(f"round {index + 1}: {bench.npu.avg_us:.2f} us, {gflops:.2f} GFLOP/s")
    median = statistics.median(rounds)
    actual = C_tensor.numpy().reshape(cfg.M, cfg.N)
    errors = _verify(opts, cfg, A, native, actual)
    result = {
        "compute_type": cfg.compute_type,
        "cache_mode": cfg.cache_mode,
        "activation_input": cfg.activation_input,
        "M": cfg.M,
        "K": cfg.K,
        "N": cfg.N,
        "m_c": cfg.m_c,
        "m_a": cfg.m_a,
        "k": cfg.k,
        "n": cfg.n,
        "prepare_ms": prepare_ms,
        "conversion_npu_us": None,
        "dma_npu_us": None,
        "average_npu_us": statistics.mean(average_us),
        "total_npu_us": statistics.mean(average_us),
        "median_gflops": median,
        **errors,
    }
    _write_results(opts, result)
    print(f"prepared Q4_K in {prepare_ms:.2f} ms; median {median:.2f} GFLOP/s")
    if opts.min_gflops and median < opts.min_gflops:
        raise RuntimeError(f"performance gate failed: {median:.2f} < {opts.min_gflops:.2f} GFLOP/s")
    print("PASS!")


def main() -> None:
    opts = _parser().parse_args()
    run_design_cli(
        whole_array_q4ks,
        opts,
        compile_kwargs=_kwargs,
        run_and_verify=_run,
        device=lambda parsed: _device_for(parsed.dev, parsed.n_aie_cols),
        validate=_validate,
    )


if __name__ == "__main__":
    main()
