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
    Buffer,
    CascadeFlow,
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
from aie.iron.device import Tile, from_name
from aie.iron.dataflow import ObjectFifoLink
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
    ACCUMULATION_MODES,
    ACTIVATION_INPUTS,
    CASCADE_CHUNK_K,
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


def _kernels(config: Q4KSConfig, a_ty, b_ty, c_ty, accumulator_ty=None):
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
    if config.accumulation_mode == "fp32":
        flags.append("-DACCUM_FP32")
    symbol = (
        "q4ks_matmul_bfp16_fp32"
        if config.accumulation_mode == "fp32"
        else f"q4ks_matmul_{config.compute_type}"
    )
    matmul_c_ty = accumulator_ty if config.accumulation_mode == "fp32" else c_ty
    matmul = ExternalFunction(
        symbol,
        object_file_name=name,
        source_file=KERNEL_SOURCES[config.compute_type],
        arg_types=[a_ty, b_ty, matmul_c_ty, np.int32],
        include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
        compile_flags=flags,
        use_chess=True,
    )
    if config.accumulation_mode == "fp32":
        zero = Kernel("q4ks_zero_f32", matmul.object_file_name, [accumulator_ty])
        store = Kernel(
            "q4ks_store_bf16", matmul.object_file_name, [accumulator_ty, c_ty]
        )
    else:
        zero = Kernel("q4ks_zero_bf16", matmul.object_file_name, [c_ty])
        store = None
    return matmul, zero, store


def _cascade_kernels(config: Q4KSConfig, a_ty, b_ty, c_ty, accumulator_ty):
    name = f"q4ks_bfp16_cascade_{config.m_c}x{config.k}x{config.n}.o"
    flags = [
        f"-DDIM_M_C={config.m_c}",
        f"-DDIM_M_A={config.m_a}",
        f"-DDIM_K={config.k}",
        f"-DDIM_N={config.n}",
        f"-DPACKED_TILE_BYTES={config.tile_bytes}",
        "-DCOMPUTE_BFP16",
        "-DACCUM_CASCADE",
        f"-I{AIE_KERNEL_INCLUDE}",
    ]
    put_only = ExternalFunction(
        "q4ks_cascade_put_only",
        object_file_name=name,
        source_file=KERNEL_SOURCES["bfp16"],
        arg_types=[a_ty, b_ty],
        include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
        compile_flags=flags,
        use_chess=True,
    )
    put_get = Kernel("q4ks_cascade_put_get", name, [a_ty, b_ty])
    get_only = Kernel(
        "q4ks_cascade_get_only", name, [a_ty, b_ty, accumulator_ty]
    )
    zero = Kernel("q4ks_zero_f32", name, [accumulator_ty])
    store = Kernel("q4ks_store_bf16", name, [accumulator_ty, c_ty])
    return put_only, put_get, get_only, zero, store


def _cascade_resident_kernels(
    config: Q4KSConfig, a_ty, b_ty, c_ty, accumulator_ty
):
    name = (
        f"q4ks_bfp16_cascade_resident_"
        f"{config.m_c}x{config.k}x{config.n}.o"
    )
    flags = [
        f"-DDIM_M_C={config.m_c}",
        f"-DDIM_M_A={config.m_a}",
        f"-DDIM_K={config.k}",
        f"-DDIM_N={config.n}",
        f"-DPACKED_TILE_BYTES={config.runtime_tile_bytes}",
        "-DCOMPUTE_BFP16",
        "-DACCUM_CASCADE_RESIDENT",
        f"-I{AIE_KERNEL_INCLUDE}",
    ]
    accumulate = ExternalFunction(
        "q4ks_cascade_resident_accumulate",
        object_file_name=name,
        source_file=KERNEL_SOURCES["bfp16"],
        arg_types=[a_ty, b_ty, accumulator_ty],
        include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
        compile_flags=flags,
        use_chess=True,
    )
    zero = Kernel("q4ks_zero_f32", name, [accumulator_ty])
    put_only = Kernel("q4ks_cascade_resident_put", name, [accumulator_ty])
    put_get = Kernel("q4ks_cascade_resident_put_get", name, [accumulator_ty])
    get_stream = Kernel("q4ks_cascade_resident_get_stream", name, [accumulator_ty])
    return accumulate, zero, put_only, put_get, get_stream


def _cascade_register_kernels(config: Q4KSConfig, a_ty, b_ty):
    shard_k = config.K // N_AIE_ROWS
    name = (
        f"q4ks_bfp16_cascade_register_"
        f"m16xk{shard_k}xn16.o"
    )
    flags = [
        f"-DDIM_M_C={config.m_c}",
        f"-DDIM_M_A={config.m_a}",
        f"-DDIM_K={config.k}",
        f"-DDIM_N={config.n}",
        f"-DDIM_SHARD_K={shard_k}",
        f"-DPACKED_TILE_BYTES={config.runtime_tile_bytes}",
        "-DCOMPUTE_BFP16",
        "-DACCUM_CASCADE_REGISTER",
        f"-I{AIE_KERNEL_INCLUDE}",
    ]
    put_only = ExternalFunction(
        "q4ks_cascade_register_put",
        object_file_name=name,
        source_file=KERNEL_SOURCES["bfp16"],
        arg_types=[a_ty, b_ty],
        include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
        compile_flags=flags,
        use_chess=True,
    )
    put_get = Kernel(
        "q4ks_cascade_register_put_get", name, [a_ty, b_ty]
    )
    get_stream = Kernel(
        "q4ks_cascade_register_get_stream",
        name,
        [a_ty, b_ty, np.int32],
    )
    return put_only, put_get, get_stream


def _cascade_chunked_kernels(config: Q4KSConfig, a_ty, b_ty, accumulator_ty):
    name = (
        f"q4ks_bfp16_cascade_chunked_"
        f"m{config.m_c}xk{CASCADE_CHUNK_K}xn{config.n}.o"
    )
    flags = [
        f"-DDIM_M_C={config.m_c}",
        f"-DDIM_M_A={config.m_a}",
        f"-DDIM_K={config.k}",
        f"-DDIM_N={config.n}",
        f"-DDIM_CHUNK_K={CASCADE_CHUNK_K}",
        f"-DPACKED_TILE_BYTES={config.runtime_tile_bytes}",
        "-DCOMPUTE_BFP16",
        "-DACCUM_CASCADE_CHUNKED",
        f"-I{AIE_KERNEL_INCLUDE}",
    ]
    put_only = ExternalFunction(
        "q4ks_cascade_chunk_put",
        object_file_name=name,
        source_file=KERNEL_SOURCES["bfp16"],
        arg_types=[a_ty, b_ty],
        include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
        compile_flags=flags,
        use_chess=True,
    )
    put_get = Kernel(
        "q4ks_cascade_chunk_put_get", name, [a_ty, b_ty]
    )
    accumulate = Kernel(
        "q4ks_cascade_chunk_accumulate",
        name,
        [a_ty, b_ty, accumulator_ty, np.int32],
    )
    zero = Kernel("q4ks_zero_f32", name, [accumulator_ty])
    stream = Kernel("q4ks_cascade_chunk_stream", name, [accumulator_ty])
    return put_only, put_get, accumulate, zero, stream


def _cascade_shared_kernels(config: Q4KSConfig, a_ty, b_ty, accumulator_half_ty):
    name = (
        f"q4ks_bfp16_cascade_shared_"
        f"m{config.m_c}xk{config.k}xn{config.n}.o"
    )
    flags = [
        f"-DDIM_M_C={config.m_c}",
        f"-DDIM_M_A={config.m_a}",
        f"-DDIM_K={config.k}",
        f"-DDIM_N={config.n}",
        f"-DPACKED_TILE_BYTES={config.tile_bytes}",
        "-DCOMPUTE_BFP16",
        "-DACCUM_CASCADE_SHARED",
        f"-I{AIE_KERNEL_INCLUDE}",
    ]
    put_only = ExternalFunction(
        "q4ks_cascade_shared_put",
        object_file_name=name,
        source_file=KERNEL_SOURCES["bfp16"],
        arg_types=[a_ty, b_ty, np.int32],
        include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
        compile_flags=flags,
        use_chess=True,
    )
    put_get = Kernel(
        "q4ks_cascade_shared_put_get", name, [a_ty, b_ty, np.int32]
    )
    accumulate = Kernel(
        "q4ks_cascade_shared_accumulate",
        name,
        [a_ty, b_ty, accumulator_half_ty, accumulator_half_ty, np.int32],
    )
    zero = Kernel(
        "q4ks_cascade_shared_zero",
        name,
        [accumulator_half_ty, accumulator_half_ty],
    )
    stream = Kernel(
        "q4ks_cascade_shared_stream",
        name,
        [accumulator_half_ty, accumulator_half_ty],
    )
    return put_only, put_get, accumulate, zero, stream


def _cascade_hybrid_kernels(config: Q4KSConfig, a_ty, b_ty, c_ty):
    base_flags = [
        f"-DDIM_M_C={config.m_c}",
        f"-DDIM_M_A={config.m_a}",
        f"-DDIM_K={config.k}",
        f"-DDIM_N={config.n}",
        f"-DPACKED_TILE_BYTES={config.tile_bytes}",
        "-DCOMPUTE_BFP16",
        "-DACCUM_CASCADE_HYBRID",
        f"-I{AIE_KERNEL_INCLUDE}",
    ]

    def role_kernels(role: int, role_name: str):
        name = (
            f"q4ks_bfp16_cascade_hybrid_{role_name}_"
            f"m{config.m_c}a{config.m_a}xk{config.k}xn{config.n}.o"
        )
        final = ExternalFunction(
            f"q4ks_cascade_hybrid_{role_name}_final",
            object_file_name=name,
            source_file=KERNEL_SOURCES["bfp16"],
            arg_types=[a_ty, b_ty, c_ty, np.int32],
            include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
            compile_flags=base_flags + [f"-DCASCADE_ROLE={role}"],
            use_chess=True,
        )
        accumulate = Kernel(
            f"q4ks_cascade_hybrid_{role_name}_accumulate",
            name,
            [a_ty, b_ty, c_ty, np.int32],
        )
        zero = Kernel(f"q4ks_zero_bf16_{role_name}", name, [c_ty])
        return zero, accumulate, final

    zero_bottom, bottom_accumulate, bottom_final = role_kernels(0, "bottom")
    zero_middle, middle_accumulate, middle_final = role_kernels(1, "middle")
    zero_top, top_accumulate, top_final = role_kernels(2, "top")
    return (
        zero_bottom,
        bottom_accumulate,
        bottom_final,
        zero_middle,
        middle_accumulate,
        middle_final,
        zero_top,
        top_accumulate,
        top_final,
    )


def _build_cascade_design(
    dev,
    config: Q4KSConfig,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    """Split K over four rows and retain completed row-group sums in FP32."""

    M, K, N = config.M, config.K, config.N
    m, k, n = config.m_c, config.k, config.n
    n_cols = config.n_aie_cols
    cascade_rounds = K // (N_AIE_ROWS * k)
    n_rounds = N // (n * n_cols)
    output_tiles_per_column = (M // m) * n_rounds

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(config.prepared_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    A_l2_ty = np.ndarray[(m, k), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[
        (N_AIE_ROWS * config.packed_rows, k), np.dtype[np.uint8]
    ]
    B_l1_ty = np.ndarray[(config.packed_rows, k), np.dtype[np.uint8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[bfloat16]]
    C_l2_ty = np.ndarray[(m, n), np.dtype[bfloat16]]
    accumulator_ty = np.ndarray[(m * n,), np.dtype[np.float32]]

    kernel_a_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    kernel_b_ty = np.ndarray[(config.tile_bytes,), np.dtype[np.uint8]]
    kernel_c_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]
    put_only, put_get, get_only, zero, store = _cascade_kernels(
        config, kernel_a_ty, kernel_b_ty, kernel_c_ty, accumulator_ty
    )

    a_to_stream: StreamDims = [
        (k // 8, 8),
        (m, k),
        (8, 1),
    ]
    a_from_stream: StreamDims = [
        (k // 8, 64),
        (m // 8, 8 * k),
        (64, 1),
    ]
    A_l3l2: list[ObjectFifo] = []
    A_l2l1: list[ObjectFifo] = []
    for row in range(N_AIE_ROWS):
        parent = ObjectFifo(A_l2_ty, name=f"A_CASCADE_L3L2_{row}", depth=2)
        child = parent.cons().forward(
            tile=Tile(row, 1),
            obj_type=A_l1_ty,
            depth=config.a_fifo_depth,
            name=f"A_CASCADE_L2L1_{row}",
            dims_to_stream=a_to_stream,
            dims_from_stream=a_from_stream,
        )
        A_l3l2.append(parent)
        A_l2l1.append(child)

    B_l3l2: list[ObjectFifo] = []
    B_l2l1: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        parent = ObjectFifo(B_l2_ty, name=f"B_CASCADE_L3L2_{col}", depth=2)
        children = parent.cons().split(
            [row * config.tile_bytes for row in range(N_AIE_ROWS)],
            tile=Tile(col, 1),
            obj_types=[B_l1_ty] * N_AIE_ROWS,
            depths=[1] * N_AIE_ROWS,
            names=[f"B_CASCADE_L2L1_{col}_{row}" for row in range(N_AIE_ROWS)],
        )
        B_l3l2.append(parent)
        for row in range(N_AIE_ROWS):
            B_l2l1[row].append(children[row])

    c_dims: StreamDims = [
        (m // 8, 8 * n),
        (8, 8),
        (n // 8, 64),
        (8, 1),
    ]
    C_l1l2: list[ObjectFifo] = []
    C_l2l3: list[ObjectFifo] = []
    for col in range(n_cols):
        child = ObjectFifo(
            C_l1_ty, name=f"C_CASCADE_L1L2_{col}", depth=config.c_fifo_depth
        )
        parent = child.cons().forward(
            tile=Tile(col, 1),
            obj_type=C_l2_ty,
            depth=config.c_fifo_depth,
            name=f"C_CASCADE_L2L3_{col}",
            dims_to_stream=c_dims,
        )
        C_l1l2.append(child)
        C_l2l3.append(parent)

    def top_fn(in_a, in_b, out_c, accumulator, zero_fn, get_fn, store_fn):
        tile_loop = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tile_loop:
            elem_c = out_c.acquire(1)
            zero_fn(accumulator)
            rounds = range_(cascade_rounds) if cascade_rounds > 1 else range(1)
            for _ in rounds:
                elem_a = in_a.acquire(1)
                elem_b = in_b.acquire(1)
                get_fn(elem_a, elem_b, accumulator)
                in_a.release(1)
                in_b.release(1)
            store_fn(accumulator, elem_c)
            out_c.release(1)

    def middle_fn(in_a, in_b, put_get_fn):
        tile_loop = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tile_loop:
            rounds = range_(cascade_rounds) if cascade_rounds > 1 else range(1)
            for _ in rounds:
                elem_a = in_a.acquire(1)
                elem_b = in_b.acquire(1)
                put_get_fn(elem_a, elem_b)
                in_a.release(1)
                in_b.release(1)

    def bottom_fn(in_a, in_b, put_only_fn):
        tile_loop = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tile_loop:
            rounds = range_(cascade_rounds) if cascade_rounds > 1 else range(1)
            for _ in rounds:
                elem_a = in_a.acquire(1)
                elem_b = in_b.acquire(1)
                put_only_fn(elem_a, elem_b)
                in_a.release(1)
                in_b.release(1)

    workers: list[list[Worker]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        accumulator = Buffer(accumulator_ty, name=f"C_CASCADE_ACC_{col}")
        workers[0].append(
            Worker(
                top_fn,
                [
                    A_l2l1[0].cons(),
                    B_l2l1[0][col].cons(),
                    C_l1l2[col].prod(),
                    accumulator,
                    zero,
                    get_only,
                    store,
                ],
                tile=Tile(col, 2),
                stack_size=0xD00,
                trace=1 if trace_config and col == 1 else 0,
            )
        )
        for row in (1, 2):
            workers[row].append(
                Worker(
                    middle_fn,
                    [
                        A_l2l1[row].cons(),
                        B_l2l1[row][col].cons(),
                        put_get,
                    ],
                    tile=Tile(col, row + 2),
                    stack_size=0xD00,
                )
            )
        workers[3].append(
            Worker(
                bottom_fn,
                [A_l2l1[3].cons(), B_l2l1[3][col].cons(), put_only],
                tile=Tile(col, 5),
                stack_size=0xD00,
            )
        )

    for col in range(n_cols):
        for row in range(N_AIE_ROWS - 1, 0, -1):
            CascadeFlow(workers[row][col], workers[row - 1][col])
    flat_workers = [worker for row in workers for worker in row]

    A_prods = [fifo.prod(tile=Tile(row, 0)) for row, fifo in enumerate(A_l3l2)]
    B_prods = [fifo.prod(tile=Tile(col, 0)) for col, fifo in enumerate(B_l3l2)]
    C_conses = [fifo.cons(tile=Tile(col, 0)) for col, fifo in enumerate(C_l2l3)]
    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []
    n_k_tiles = K // k
    bytes_per_column = n_rounds * n_k_tiles * config.tile_bytes

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        for row_base in range(0, M // m, 2):
            row_count = min(2, M // m - row_base)
            task_group = TaskGroup()
            for tile_row in range(row_count):
                m_tile = row_base + tile_row
                for row in range(N_AIE_ROWS):
                    a_tap = TensorAccessPattern(
                        (M, K),
                        offset=m_tile * m * K + row * k,
                        sizes=[n_rounds, cascade_rounds, m, k],
                        strides=[0, N_AIE_ROWS * k, K, 1],
                    )
                    A_hs[row].fill(A, tap=a_tap, group=task_group)
                    A_taps.append(a_tap)
            for col in range(n_cols):
                c_tap = TensorAccessPattern(
                    (M, N),
                    offset=row_base * m * N + col * n,
                    sizes=[row_count, n_rounds, m, n],
                    strides=[m * N, n * n_cols, N, 1],
                )
                C_hs[col].drain(C, tap=c_tap, wait=True, group=task_group)
                C_taps.append(c_tap)
                for tile_row in range(row_count):
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=col * bytes_per_column,
                        sizes=[
                            n_rounds,
                            cascade_rounds,
                            N_AIE_ROWS * config.packed_rows,
                            k,
                        ],
                        strides=[
                            n_k_tiles * config.tile_bytes,
                            N_AIE_ROWS * config.tile_bytes,
                            k,
                            1,
                        ],
                    )
                    B_hs[col].fill(B, tap=b_tap, group=task_group)
                    B_taps.append(b_tap)
            task_group.finish()

    runtime = Runtime(
        sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses]
    )
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        if n_cols == 8:
            raise ValueError("trace requires a spare shim column; use fewer than 8 columns")
        program.enable_trace(
            trace_config.trace_size,
            workers=[workers[0][1]],
            egress_shim_col=n_cols,
        )
    module = program.resolve_program()
    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


def _build_cascade_resident_design(
    dev,
    config: Q4KSConfig,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    """Accumulate K/4 per row in FP32, then cascade-reduce each C tile once."""

    M, K, N = config.M, config.K, config.N
    m, k, n = config.m_c, config.k, config.n
    n_cols = config.n_aie_cols
    n_k_tiles = K // k
    shard_rounds = n_k_tiles // N_AIE_ROWS
    n_rounds = N // (n * n_cols)
    n_m_tiles = M // m
    output_tiles_per_column = n_m_tiles * n_rounds
    replay_rows = next(
        replay
        for replay in range(min(4, n_m_tiles), 0, -1)
        if n_m_tiles % replay == 0
    )

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(config.prepared_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    A_l2_ty = np.ndarray[(m, k), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[bfloat16]]

    expanded_rows = config.runtime_packed_rows
    B_l1_ty = np.ndarray[(expanded_rows, k), np.dtype[np.uint8]]
    B_row_panel_ty = np.ndarray[
        (shard_rounds * expanded_rows, k), np.dtype[np.uint8]
    ]
    B_panel_ty = np.ndarray[
        (n_k_tiles * expanded_rows, k), np.dtype[np.uint8]
    ]
    C_l1_ty = np.ndarray[(m, n), np.dtype[bfloat16]]
    C_l2_ty = np.ndarray[(m, n), np.dtype[bfloat16]]
    accumulator_ty = np.ndarray[(m * n,), np.dtype[np.float32]]

    kernel_a_ty = np.ndarray[(m * k,), np.dtype[bfloat16]]
    kernel_b_ty = np.ndarray[
        (config.runtime_tile_bytes,), np.dtype[np.uint8]
    ]
    kernel_c_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]
    accumulate, zero, put_only, put_get, get_stream = (
        _cascade_resident_kernels(
            config, kernel_a_ty, kernel_b_ty, kernel_c_ty, accumulator_ty
        )
    )

    a_to_stream: StreamDims = [
        (k // 8, 8),
        (m, k),
        (8, 1),
    ]
    a_from_stream: StreamDims = [
        (k // 8, 64),
        (m // 8, 8 * k),
        (64, 1),
    ]
    A_l3l2: list[ObjectFifo] = []
    A_l2l1: list[ObjectFifo] = []
    for row in range(N_AIE_ROWS):
        parent = ObjectFifo(
            A_l2_ty, name=f"A_RESIDENT_L3L2_{row}", depth=2
        )
        child = parent.cons().forward(
            tile=Tile(row, 1),
            obj_type=A_l1_ty,
            depth=config.a_fifo_depth,
            name=f"A_RESIDENT_L2L1_{row}",
            dims_to_stream=a_to_stream,
            dims_from_stream=a_from_stream,
        )
        A_l3l2.append(parent)
        A_l2l1.append(child)

    B_l3l2: list[ObjectFifo] = []
    B_l2l1: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        parent = ObjectFifo(
            B_panel_ty,
            name=f"B_RESIDENT_L3L2_{col}",
            depth=1,
        )
        children = [
            ObjectFifo(
                B_row_panel_ty,
                consumer_obj_type=B_l1_ty,
                name=f"B_RESIDENT_L2L1_{col}_{row}",
                depth=1,
                repeat_count=replay_rows,
            )
            for row in range(N_AIE_ROWS)
        ]
        ObjectFifoLink(
            parent.cons(),
            [child.prod() for child in children],
            tile=Tile(col, 1),
            dst_offsets=[
                row * shard_rounds * config.runtime_tile_bytes
                for row in range(N_AIE_ROWS)
            ],
        )
        B_l3l2.append(parent)
        for row in range(N_AIE_ROWS):
            B_l2l1[row].append(children[row])

    C_l1l2: list[ObjectFifo] = []
    C_l2l3: list[ObjectFifo] = []
    for col in range(n_cols):
        stream = ObjectFifo(
            C_l1_ty,
            name=f"C_RESIDENT_L1L3_{col}",
            depth=2,
            aie_stream=(0, 0),
        )
        C_l1l2.append(stream)
        C_l2l3.append(stream)

    def local_accumulate(in_a, in_b, accumulator, zero_fn, accumulate_fn):
        zero_fn(accumulator)
        rounds = range_(shard_rounds) if shard_rounds > 1 else range(1)
        for _ in rounds:
            elem_a = in_a.acquire(1)
            elem_b = in_b.acquire(1)
            accumulate_fn(elem_a, elem_b, accumulator)
            in_a.release(1)
            in_b.release(1)

    def top_fn(
        in_a,
        in_b,
        out_c,
        accumulator,
        zero_fn,
        accumulate_fn,
        get_stream_fn,
    ):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            local_accumulate(in_a, in_b, accumulator, zero_fn, accumulate_fn)
            get_stream_fn(accumulator)

    def middle_fn(
        in_a,
        in_b,
        accumulator,
        zero_fn,
        accumulate_fn,
        put_get_fn,
    ):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            local_accumulate(in_a, in_b, accumulator, zero_fn, accumulate_fn)
            put_get_fn(accumulator)

    def bottom_fn(
        in_a,
        in_b,
        accumulator,
        zero_fn,
        accumulate_fn,
        put_only_fn,
    ):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            local_accumulate(in_a, in_b, accumulator, zero_fn, accumulate_fn)
            put_only_fn(accumulator)

    workers: list[list[Worker]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        top_accumulator = Buffer(
            accumulator_ty, name=f"C_RESIDENT_ACC_0_{col}"
        )
        workers[0].append(
            Worker(
                top_fn,
                [
                    A_l2l1[0].cons(),
                    B_l2l1[0][col].cons(),
                    C_l1l2[col].prod(),
                    top_accumulator,
                    zero,
                    accumulate,
                    get_stream,
                ],
                tile=Tile(col, 2),
                stack_size=0xD00,
                trace=1 if trace_config and col == 1 else 0,
            )
        )
        for row in (1, 2):
            accumulator = Buffer(
                accumulator_ty, name=f"C_RESIDENT_ACC_{row}_{col}"
            )
            workers[row].append(
                Worker(
                    middle_fn,
                    [
                        A_l2l1[row].cons(),
                        B_l2l1[row][col].cons(),
                        accumulator,
                        zero,
                        accumulate,
                        put_get,
                    ],
                    tile=Tile(col, row + 2),
                    stack_size=0xD00,
                )
            )
        bottom_accumulator = Buffer(
            accumulator_ty, name=f"C_RESIDENT_ACC_3_{col}"
        )
        workers[3].append(
            Worker(
                bottom_fn,
                [
                    A_l2l1[3].cons(),
                    B_l2l1[3][col].cons(),
                    bottom_accumulator,
                    zero,
                    accumulate,
                    put_only,
                ],
                tile=Tile(col, 5),
                stack_size=0xD00,
            )
        )

    for col in range(n_cols):
        for row in range(N_AIE_ROWS - 1, 0, -1):
            CascadeFlow(workers[row][col], workers[row - 1][col])
    flat_workers = [worker for row in workers for worker in row]

    A_prods = [
        fifo.prod(tile=Tile(row, 0)) for row, fifo in enumerate(A_l3l2)
    ]
    B_prods = [
        fifo.prod(tile=Tile(col, 0)) for col, fifo in enumerate(B_l3l2)
    ]
    C_conses = [
        fifo.cons(tile=Tile(col, 0)) for col, fifo in enumerate(C_l2l3)
    ]
    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []
    panel_bytes = n_k_tiles * config.runtime_tile_bytes
    bytes_per_column = n_rounds * panel_bytes

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        for n_round in range(n_rounds):
            for slab_base in range(0, n_m_tiles, replay_rows):
                row_count = min(replay_rows, n_m_tiles - slab_base)
                task_group = TaskGroup()
                for col in range(n_cols):
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=col * bytes_per_column + n_round * panel_bytes,
                        sizes=[1, n_k_tiles, expanded_rows, k],
                        strides=[0, config.runtime_tile_bytes, k, 1],
                    )
                    B_hs[col].fill(B, tap=b_tap, group=task_group)
                    B_taps.append(b_tap)

                    n_tile = n_round * n_cols + col
                    c_tap = TensorAccessPattern(
                        (M, N),
                        offset=slab_base * m * N + n_tile * n,
                        sizes=[row_count, 1, m, n],
                        strides=[m * N, 0, N, 1],
                    )
                    C_hs[col].drain(
                        C, tap=c_tap, wait=True, group=task_group
                    )
                    C_taps.append(c_tap)

                for row in range(N_AIE_ROWS):
                    a_tap = TensorAccessPattern(
                        (M, K),
                        offset=slab_base * m * K + row * shard_rounds * k,
                        sizes=[row_count, shard_rounds, m, k],
                        strides=[m * K, k, K, 1],
                    )
                    A_hs[row].fill(A, tap=a_tap, group=task_group)
                    A_taps.append(a_tap)
                task_group.finish()

    runtime = Runtime(
        sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses]
    )
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        if n_cols == 8:
            raise ValueError("trace requires a spare shim column; use fewer than 8 columns")
        program.enable_trace(
            trace_config.trace_size,
            workers=[workers[0][1]],
            egress_shim_col=n_cols,
        )
    module = program.resolve_program()
    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


def _build_cascade_shared_design(
    dev,
    config: Q4KSConfig,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    """Build the shared-panel topology used by both cascade experiments.

    cascade-shared keeps a 128x128 FP32 C tile split across two L1s and
    cascades every 64-K contribution. cascade-hybrid instead accumulates each
    row's K/4 shard with the fast local BFP16/BF16 kernel, then performs one
    accfloat cascade reduction. Both keep compressed Q4_K panels in MemTile
    and assign interleaved 64-K slices to the four cascade rows.
    """

    hybrid = config.accumulation_mode == "cascade-hybrid"
    M, K, N = config.M, config.K, config.N
    m, k, n = config.m_c, config.k, config.n
    n_cols = config.n_aie_cols
    chunks_per_row = K // (N_AIE_ROWS * k)
    row_blocks = m // config.m_a
    row_block_unroll = 4
    if row_blocks % row_block_unroll:
        raise ValueError(
            "cascade-hybrid requires m_c / m_a to be divisible by 4"
        )
    row_block_groups = row_blocks // row_block_unroll
    n_rounds = N // (n * n_cols)
    n_m_tiles = M // m
    output_tiles_per_column = n_m_tiles * n_rounds
    dma_slab_rows = next(
        replay
        for replay in range(min(4, n_m_tiles), 0, -1)
        if n_m_tiles % replay == 0
    )
    # This split cascade consumes the complete row-local K stream before
    # advancing the parent object and supports eight reliable panel replays.
    # A repeat count of sixteen silently corrupts ordering on this NPU2, so a
    # larger M dimension is divided into independent eight-row residency slabs.
    weight_replay_rows = next(
        replay
        for replay in range(min(8, n_m_tiles), 0, -1)
        if n_m_tiles % replay == 0 and replay % dma_slab_rows == 0
    )

    a_panel_values = m * k
    a_block_values = config.m_a * k
    b_tile_bytes = config.tile_bytes
    b_row_bytes = chunks_per_row * b_tile_bytes
    b_panel_bytes = N_AIE_ROWS * b_row_bytes

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(config.prepared_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    A_panel_ty = np.ndarray[(a_panel_values,), np.dtype[bfloat16]]
    A_block_ty = np.ndarray[(a_block_values,), np.dtype[bfloat16]]
    B_panel_ty = np.ndarray[(b_panel_bytes,), np.dtype[np.uint8]]
    B_row_ty = np.ndarray[(b_row_bytes,), np.dtype[np.uint8]]
    B_tile_ty = np.ndarray[(b_tile_bytes,), np.dtype[np.uint8]]
    C_stream_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]
    accumulator_half_ty = np.ndarray[(m * n // 2,), np.dtype[np.float32]]
    local_c_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    if hybrid:
        (
            hybrid_zero_bottom,
            hybrid_bottom_accumulate,
            hybrid_bottom_final,
            hybrid_zero_middle,
            hybrid_middle_accumulate,
            hybrid_middle_final,
            hybrid_zero_top,
            hybrid_top_accumulate,
            hybrid_top_final,
        ) = _cascade_hybrid_kernels(config, A_block_ty, B_tile_ty, local_c_ty)
    else:
        put_only, put_get, accumulate, zero, stream = _cascade_shared_kernels(
            config, A_block_ty, B_tile_ty, accumulator_half_ty
        )

    a_to_stream: StreamDims = [
        (row_blocks, config.m_a * k),
        (k // 8, 8),
        (config.m_a, k),
        (8, 1),
    ]
    a_from_stream: StreamDims = [
        (k // 8, 64),
        (config.m_a // 8, 8 * k),
        (64, 1),
    ]

    A_l3l2: list[ObjectFifo] = []
    A_l2l1: list[ObjectFifo] = []
    for row in range(N_AIE_ROWS):
        parent = ObjectFifo(
            A_panel_ty,
            name=f"A_SHARED_L3L2_{row}",
            depth=2,
        )
        child = ObjectFifo(
            A_panel_ty,
            consumer_obj_type=A_block_ty,
            name=f"A_SHARED_L2L1_{row}",
            depth=1,
            dims_to_stream=a_to_stream,
            dims_from_stream_per_cons=a_from_stream,
        )
        ObjectFifoLink(parent.cons(), child.prod(), tile=Tile(row, 1))
        A_l3l2.append(parent)
        A_l2l1.append(child)

    B_l3l2: list[ObjectFifo] = []
    B_l2l1: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        parent = ObjectFifo(
            B_panel_ty,
            name=f"B_SHARED_L3L2_{col}",
            depth=1,
        )
        children = [
            ObjectFifo(
                B_row_ty,
                consumer_obj_type=B_tile_ty,
                name=f"B_SHARED_L2L1_{col}_{row}",
                depth=1,
                repeat_count=weight_replay_rows,
            )
            for row in range(N_AIE_ROWS)
        ]
        ObjectFifoLink(
            parent.cons(),
            [child.prod() for child in children],
            tile=Tile(col, 1),
            dst_offsets=[row * b_row_bytes for row in range(N_AIE_ROWS)],
        )
        B_l3l2.append(parent)
        for row in range(N_AIE_ROWS):
            B_l2l1[row].append(children[row])

    C_l1l2: list[ObjectFifo] = []
    C_l2l3: list[ObjectFifo] = []
    c_dims: StreamDims = [
        (m // 8, 8 * n),
        (8, 8),
        (n // 8, 64),
        (8, 1),
    ]
    for col in range(n_cols):
        if hybrid:
            child = ObjectFifo(
                C_stream_ty,
                name=f"C_HYBRID_L1L2_{col}",
                depth=1,
            )
            parent = child.cons().forward(
                obj_type=C_stream_ty,
                name=f"C_HYBRID_L2L3_{col}",
                depth=1,
                dims_to_stream=c_dims,
                tile=Tile(col, 1),
            )
        else:
            child = ObjectFifo(
                C_stream_ty,
                name=f"C_SHARED_L1L3_{col}",
                depth=2,
                aie_stream=(0, 0),
            )
            parent = child
        C_l1l2.append(child)
        C_l2l3.append(parent)

    def bottom_fn(in_a, in_b, put_fn):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            for _chunk in (range_(chunks_per_row) if chunks_per_row > 1 else range(1)):
                elem_b = in_b.acquire(1)
                for rb in range(row_blocks):
                    elem_a = in_a.acquire(1)
                    put_fn(elem_a, elem_b, rb)
                    in_a.release(1)
                in_b.release(1)

    def middle_fn(in_a, in_b, put_get_fn):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            for _chunk in (range_(chunks_per_row) if chunks_per_row > 1 else range(1)):
                elem_b = in_b.acquire(1)
                for rb in range(row_blocks):
                    elem_a = in_a.acquire(1)
                    put_get_fn(elem_a, elem_b, rb)
                    in_a.release(1)
                in_b.release(1)

    def top_fn(
        in_a,
        in_b,
        out_c,
        accumulator_low,
        accumulator_high,
        zero_fn,
        accumulate_fn,
        stream_fn,
    ):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            zero_fn(accumulator_low, accumulator_high)
            for _chunk in (range_(chunks_per_row) if chunks_per_row > 1 else range(1)):
                elem_b = in_b.acquire(1)
                for rb in range(row_blocks):
                    elem_a = in_a.acquire(1)
                    accumulate_fn(
                        elem_a,
                        elem_b,
                        accumulator_low,
                        accumulator_high,
                        rb,
                    )
                    in_a.release(1)
                in_b.release(1)
            stream_fn(accumulator_low, accumulator_high)

    prefix_chunks = chunks_per_row - 1

    def hybrid_bottom_fn(
        in_a, in_b, local_c, zero_fn, accumulate_fn, final_fn
    ):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            zero_fn(local_c)
            if prefix_chunks:
                prefix = (
                    range_(prefix_chunks)
                    if prefix_chunks > 1
                    else range(1)
                )
                for _chunk in prefix:
                    elem_b = in_b.acquire(1)
                    for rb_group in (
                        range_(row_block_groups)
                        if row_block_groups > 1
                        else range(1)
                    ):
                        for rb_offset in range(row_block_unroll):
                            rb = rb_group * row_block_unroll + rb_offset
                            elem_a = in_a.acquire(1)
                            accumulate_fn(elem_a, elem_b, local_c, rb)
                            in_a.release(1)
                    in_b.release(1)
            elem_b = in_b.acquire(1)
            for rb_group in (
                range_(row_block_groups)
                if row_block_groups > 1
                else range(1)
            ):
                for rb_offset in range(row_block_unroll):
                    rb = rb_group * row_block_unroll + rb_offset
                    elem_a = in_a.acquire(1)
                    final_fn(elem_a, elem_b, local_c, rb)
                    in_a.release(1)
            in_b.release(1)

    def hybrid_middle_fn(
        in_a, in_b, local_c, zero_fn, accumulate_fn, final_fn
    ):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            zero_fn(local_c)
            if prefix_chunks:
                prefix = (
                    range_(prefix_chunks)
                    if prefix_chunks > 1
                    else range(1)
                )
                for _chunk in prefix:
                    elem_b = in_b.acquire(1)
                    for rb_group in (
                        range_(row_block_groups)
                        if row_block_groups > 1
                        else range(1)
                    ):
                        for rb_offset in range(row_block_unroll):
                            rb = rb_group * row_block_unroll + rb_offset
                            elem_a = in_a.acquire(1)
                            accumulate_fn(elem_a, elem_b, local_c, rb)
                            in_a.release(1)
                    in_b.release(1)
            elem_b = in_b.acquire(1)
            for rb_group in (
                range_(row_block_groups)
                if row_block_groups > 1
                else range(1)
            ):
                for rb_offset in range(row_block_unroll):
                    rb = rb_group * row_block_unroll + rb_offset
                    elem_a = in_a.acquire(1)
                    final_fn(elem_a, elem_b, local_c, rb)
                    in_a.release(1)
            in_b.release(1)

    def hybrid_top_fn(
        in_a,
        in_b,
        out_c,
        zero_fn,
        accumulate_fn,
        final_fn,
    ):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            elem_c = out_c.acquire(1)
            zero_fn(elem_c)
            if prefix_chunks:
                prefix = (
                    range_(prefix_chunks)
                    if prefix_chunks > 1
                    else range(1)
                )
                for _chunk in prefix:
                    elem_b = in_b.acquire(1)
                    for rb_group in (
                        range_(row_block_groups)
                        if row_block_groups > 1
                        else range(1)
                    ):
                        for rb_offset in range(row_block_unroll):
                            rb = rb_group * row_block_unroll + rb_offset
                            elem_a = in_a.acquire(1)
                            accumulate_fn(elem_a, elem_b, elem_c, rb)
                            in_a.release(1)
                    in_b.release(1)
            elem_b = in_b.acquire(1)
            for rb_group in (
                range_(row_block_groups)
                if row_block_groups > 1
                else range(1)
            ):
                for rb_offset in range(row_block_unroll):
                    rb = rb_group * row_block_unroll + rb_offset
                    elem_a = in_a.acquire(1)
                    final_fn(elem_a, elem_b, elem_c, rb)
                    in_a.release(1)
            in_b.release(1)
            out_c.release(1)

    workers: list[list[Worker]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        if hybrid:
            for row in range(N_AIE_ROWS):
                tile = Tile(col, row + 2)
                common_args = [
                    A_l2l1[row].cons(),
                    B_l2l1[row][col].cons(),
                ]
                if row == 0:
                    worker_fn = hybrid_top_fn
                    worker_args = common_args + [
                        C_l1l2[col].prod(),
                        hybrid_zero_top,
                        hybrid_top_accumulate,
                        hybrid_top_final,
                    ]
                elif row == N_AIE_ROWS - 1:
                    local_c = Buffer(
                        local_c_ty,
                        name=f"C_HYBRID_LOCAL_{col}_{row}",
                        tile=tile,
                    )
                    worker_fn = hybrid_bottom_fn
                    worker_args = common_args + [
                        local_c,
                        hybrid_zero_bottom,
                        hybrid_bottom_accumulate,
                        hybrid_bottom_final,
                    ]
                else:
                    local_c = Buffer(
                        local_c_ty,
                        name=f"C_HYBRID_LOCAL_{col}_{row}",
                        tile=tile,
                    )
                    worker_fn = hybrid_middle_fn
                    worker_args = common_args + [
                        local_c,
                        hybrid_zero_middle,
                        hybrid_middle_accumulate,
                        hybrid_middle_final,
                    ]
                workers[row].append(
                    Worker(
                        worker_fn,
                        worker_args,
                        tile=tile,
                        stack_size=0xD00,
                        trace=(
                            1
                            if trace_config and row == 0 and col == 1
                            else 0
                        ),
                    )
                )
            continue
        top_tile = Tile(col, 2)
        neighbor_tile = Tile(col, 3)
        accumulator_low = Buffer(
            accumulator_half_ty,
            name=f"C_SHARED_ACC_LOW_{col}",
            tile=top_tile,
        )
        accumulator_high = Buffer(
            accumulator_half_ty,
            name=f"C_SHARED_ACC_HIGH_{col}",
            tile=neighbor_tile,
        )
        workers[0].append(
            Worker(
                top_fn,
                [
                    A_l2l1[0].cons(),
                    B_l2l1[0][col].cons(),
                    C_l1l2[col].prod(),
                    accumulator_low,
                    accumulator_high,
                    zero,
                    accumulate,
                    stream,
                ],
                tile=top_tile,
                stack_size=0xD00,
                trace=1 if trace_config and col == 1 else 0,
            )
        )
        for row in (1, 2):
            workers[row].append(
                Worker(
                    middle_fn,
                    [
                        A_l2l1[row].cons(),
                        B_l2l1[row][col].cons(),
                        put_get,
                    ],
                    tile=Tile(col, row + 2),
                    stack_size=0xD00,
                )
            )
        workers[3].append(
            Worker(
                bottom_fn,
                [A_l2l1[3].cons(), B_l2l1[3][col].cons(), put_only],
                tile=Tile(col, 5),
                stack_size=0xD00,
            )
        )

    for col in range(n_cols):
        for row in range(N_AIE_ROWS - 1, 0, -1):
            CascadeFlow(workers[row][col], workers[row - 1][col])
    flat_workers = [worker for row in workers for worker in row]

    A_prods = [
        fifo.prod(tile=Tile(row, 0)) for row, fifo in enumerate(A_l3l2)
    ]
    B_prods = [
        fifo.prod(tile=Tile(col, 0)) for col, fifo in enumerate(B_l3l2)
    ]
    C_conses = [
        fifo.cons(tile=Tile(col, 0)) for col, fifo in enumerate(C_l2l3)
    ]
    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []
    bytes_per_column = n_rounds * b_panel_bytes

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        for n_round in range(n_rounds):
            for weight_base in range(0, n_m_tiles, weight_replay_rows):
                weight_group = TaskGroup()
                for col in range(n_cols):
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=col * bytes_per_column + n_round * b_panel_bytes,
                        sizes=[
                            N_AIE_ROWS,
                            chunks_per_row,
                            config.packed_rows,
                            k,
                        ],
                        strides=[b_row_bytes, b_tile_bytes, k, 1],
                    )
                    B_hs[col].fill(B, tap=b_tap, group=weight_group)
                    B_taps.append(b_tap)

                weight_end = weight_base + weight_replay_rows
                for slab_base in range(weight_base, weight_end, dma_slab_rows):
                    task_group = TaskGroup()
                    for row in range(N_AIE_ROWS):
                        a_tap = TensorAccessPattern(
                            (M, K),
                            offset=slab_base * m * K + row * k,
                            sizes=[dma_slab_rows, chunks_per_row, m, k],
                            strides=[m * K, N_AIE_ROWS * k, K, 1],
                        )
                        A_hs[row].fill(A, tap=a_tap, group=task_group)
                        A_taps.append(a_tap)

                    for col in range(n_cols):
                        n_tile = n_round * n_cols + col
                        c_tap = TensorAccessPattern(
                            (M, N),
                            offset=slab_base * m * N + n_tile * n,
                            sizes=[dma_slab_rows, 1, m, n],
                            strides=[m * N, 0, N, 1],
                        )
                        C_hs[col].drain(
                            C, tap=c_tap, wait=True, group=task_group
                        )
                        C_taps.append(c_tap)
                    task_group.finish()
                weight_group.finish()

    runtime = Runtime(
        sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses]
    )
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        if n_cols == 8:
            raise ValueError("trace requires a spare shim column; use fewer than 8 columns")
        program.enable_trace(
            trace_config.trace_size,
            workers=[workers[0][1]],
            egress_shim_col=n_cols,
        )
    module = program.resolve_program()
    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


def _build_cascade_hybrid_2way_design(
    dev,
    config: Q4KSConfig,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    """Run two independent two-row K-split cascades in every NPU column."""

    M, K, N = config.M, config.K, config.N
    m, k, n = config.m_c, config.k, config.n
    n_cols = config.n_aie_cols
    cascade_rows = 2
    n_chains = N_AIE_ROWS // cascade_rows
    memtile_weight = config.cache_mode == "memtile-weight"
    chunks_per_shard = K // (cascade_rows * k)
    row_blocks = m // config.m_a
    row_block_unroll = 4 if row_blocks % 4 == 0 else 2
    if row_blocks % row_block_unroll:
        raise ValueError(
            "cascade-hybrid requires m_c / m_a to be divisible by 2"
        )
    row_block_groups = row_blocks // row_block_unroll
    n_rounds = N // (n * n_cols)
    n_m_tiles = M // m
    if n_m_tiles % n_chains:
        raise ValueError("cascade-hybrid requires an even number of M tiles")
    n_pair_tiles = n_m_tiles // n_chains
    output_tiles_per_chain = n_pair_tiles * n_rounds
    dma_slab_pairs = next(
        replay
        for replay in range(min(4, n_pair_tiles), 0, -1)
        if n_pair_tiles % replay == 0
    )
    weight_replay_pairs = next(
        replay
        for replay in range(min(8, n_pair_tiles), 0, -1)
        if n_pair_tiles % replay == 0 and replay % dma_slab_pairs == 0
    )

    a_panel_values = m * k
    a_block_values = config.m_a * k
    b_tile_bytes = config.tile_bytes
    b_shard_bytes = chunks_per_shard * b_tile_bytes
    b_panel_bytes = cascade_rows * b_shard_bytes

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(config.prepared_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    A_panel_ty = np.ndarray[(a_panel_values,), np.dtype[bfloat16]]
    A_block_ty = np.ndarray[(a_block_values,), np.dtype[bfloat16]]
    B_panel_ty = np.ndarray[(b_panel_bytes,), np.dtype[np.uint8]]
    B_shard_ty = np.ndarray[(b_shard_bytes,), np.dtype[np.uint8]]
    B_tile_ty = np.ndarray[(b_tile_bytes,), np.dtype[np.uint8]]
    C_stream_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]
    local_c_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    (
        hybrid_zero_bottom,
        hybrid_bottom_accumulate,
        hybrid_bottom_final,
        _,
        _,
        _,
        hybrid_zero_top,
        hybrid_top_accumulate,
        hybrid_top_final,
    ) = _cascade_hybrid_kernels(config, A_block_ty, B_tile_ty, local_c_ty)

    a_to_stream: StreamDims = [
        (row_blocks, config.m_a * k),
        (k // 8, 8),
        (config.m_a, k),
        (8, 1),
    ]
    a_from_stream: StreamDims = [
        (k // 8, 64),
        (config.m_a // 8, 8 * k),
        (64, 1),
    ]

    A_l3l2: list[ObjectFifo] = []
    A_l2l1: list[ObjectFifo] = []
    for row in range(N_AIE_ROWS):
        parent = ObjectFifo(
            A_panel_ty,
            name=f"A_HYBRID2_L3L2_{row}",
            depth=2,
        )
        child = ObjectFifo(
            A_panel_ty,
            consumer_obj_type=A_block_ty,
            name=f"A_HYBRID2_L2L1_{row}",
            depth=1,
            dims_to_stream=a_to_stream,
            dims_from_stream_per_cons=a_from_stream,
        )
        ObjectFifoLink(parent.cons(), child.prod(), tile=Tile(row, 1))
        A_l3l2.append(parent)
        A_l2l1.append(child)

    B_l3l2: list[ObjectFifo] = []
    B_l3l2_cols: list[int] = []
    B_l2l1: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        parent = ObjectFifo(
            B_panel_ty,
            name=f"B_HYBRID2_L3L2_{col}",
            depth=1,
        )
        shards = [
            ObjectFifo(
                B_shard_ty,
                consumer_obj_type=B_tile_ty,
                name=f"B_HYBRID2_L2L1_{col}_{shard}",
                depth=1,
                repeat_count=weight_replay_pairs if memtile_weight else 1,
            )
            for shard in range(cascade_rows)
        ]
        ObjectFifoLink(
            parent.cons(),
            [shard.prod() for shard in shards],
            tile=Tile(col, 1),
            dst_offsets=[
                shard * b_shard_bytes for shard in range(cascade_rows)
            ],
        )
        B_l3l2.append(parent)
        B_l3l2_cols.append(col)
        for row in range(N_AIE_ROWS):
            B_l2l1[row].append(shards[row % cascade_rows])

    c_dims: StreamDims = [
        (m // 8, 8 * n),
        (8, 8),
        (n // 8, 64),
        (8, 1),
    ]
    C_l1l2: list[list[ObjectFifo]] = [[] for _ in range(n_chains)]
    C_l2l3: list[list[ObjectFifo]] = [[] for _ in range(n_chains)]
    for chain in range(n_chains):
        for col in range(n_cols):
            child = ObjectFifo(
                C_stream_ty,
                name=f"C_HYBRID2_L1L2_{chain}_{col}",
                depth=1,
            )
            parent = child.cons().forward(
                obj_type=C_stream_ty,
                name=f"C_HYBRID2_L2L3_{chain}_{col}",
                depth=1,
                dims_to_stream=c_dims,
                tile=Tile(col, 1),
            )
            C_l1l2[chain].append(child)
            C_l2l3[chain].append(parent)

    prefix_chunks = chunks_per_shard - 1

    def hybrid_bottom_fn(
        in_a, in_b, local_c, zero_fn, accumulate_fn, final_fn
    ):
        tiles = (
            range_(output_tiles_per_chain)
            if output_tiles_per_chain > 1
            else range(1)
        )
        for _ in tiles:
            zero_fn(local_c)
            if prefix_chunks:
                prefix = (
                    range_(prefix_chunks)
                    if prefix_chunks > 1
                    else range(1)
                )
                for _chunk in prefix:
                    elem_b = in_b.acquire(1)
                    for rb_group in (
                        range_(row_block_groups)
                        if row_block_groups > 1
                        else range(1)
                    ):
                        for rb_offset in range(row_block_unroll):
                            rb = rb_group * row_block_unroll + rb_offset
                            elem_a = in_a.acquire(1)
                            accumulate_fn(elem_a, elem_b, local_c, rb)
                            in_a.release(1)
                    in_b.release(1)
            elem_b = in_b.acquire(1)
            for rb_group in (
                range_(row_block_groups)
                if row_block_groups > 1
                else range(1)
            ):
                for rb_offset in range(row_block_unroll):
                    rb = rb_group * row_block_unroll + rb_offset
                    elem_a = in_a.acquire(1)
                    final_fn(elem_a, elem_b, local_c, rb)
                    in_a.release(1)
            in_b.release(1)

    def hybrid_top_fn(
        in_a,
        in_b,
        out_c,
        zero_fn,
        accumulate_fn,
        final_fn,
    ):
        tiles = (
            range_(output_tiles_per_chain)
            if output_tiles_per_chain > 1
            else range(1)
        )
        for _ in tiles:
            elem_c = out_c.acquire(1)
            zero_fn(elem_c)
            if prefix_chunks:
                prefix = (
                    range_(prefix_chunks)
                    if prefix_chunks > 1
                    else range(1)
                )
                for _chunk in prefix:
                    elem_b = in_b.acquire(1)
                    for rb_group in (
                        range_(row_block_groups)
                        if row_block_groups > 1
                        else range(1)
                    ):
                        for rb_offset in range(row_block_unroll):
                            rb = rb_group * row_block_unroll + rb_offset
                            elem_a = in_a.acquire(1)
                            accumulate_fn(elem_a, elem_b, elem_c, rb)
                            in_a.release(1)
                    in_b.release(1)
            elem_b = in_b.acquire(1)
            for rb_group in (
                range_(row_block_groups)
                if row_block_groups > 1
                else range(1)
            ):
                for rb_offset in range(row_block_unroll):
                    rb = rb_group * row_block_unroll + rb_offset
                    elem_a = in_a.acquire(1)
                    final_fn(elem_a, elem_b, elem_c, rb)
                    in_a.release(1)
            in_b.release(1)
            out_c.release(1)

    workers: list[list[Worker]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        for row in range(N_AIE_ROWS):
            tile = Tile(col, row + 2)
            common_args = [
                A_l2l1[row].cons(),
                B_l2l1[row][col].cons(),
            ]
            if row % cascade_rows == 0:
                chain = row // cascade_rows
                worker_fn = hybrid_top_fn
                worker_args = common_args + [
                    C_l1l2[chain][col].prod(),
                    hybrid_zero_top,
                    hybrid_top_accumulate,
                    hybrid_top_final,
                ]
            else:
                local_c = Buffer(
                    local_c_ty,
                    name=f"C_HYBRID2_LOCAL_{col}_{row}",
                    tile=tile,
                )
                worker_fn = hybrid_bottom_fn
                worker_args = common_args + [
                    local_c,
                    hybrid_zero_bottom,
                    hybrid_bottom_accumulate,
                    hybrid_bottom_final,
                ]
            workers[row].append(
                Worker(
                    worker_fn,
                    worker_args,
                    tile=tile,
                    stack_size=0xD00,
                    trace=(
                        1
                        if trace_config and row == 0 and col == 1
                        else 0
                    ),
                )
            )

    for col in range(n_cols):
        CascadeFlow(workers[1][col], workers[0][col])
        CascadeFlow(workers[3][col], workers[2][col])
    flat_workers = [worker for row in workers for worker in row]

    A_prods = [
        fifo.prod(tile=Tile(row, 0)) for row, fifo in enumerate(A_l3l2)
    ]
    B_prods = [
        fifo.prod(tile=Tile(col, 0))
        for col, fifo in zip(B_l3l2_cols, B_l3l2)
    ]
    C_conses = [
        C_l2l3[chain][col].cons(tile=Tile(col, 0))
        for chain in range(n_chains)
        for col in range(n_cols)
    ]
    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []
    bytes_per_column = n_rounds * b_panel_bytes

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        if not memtile_weight:
            for slab_base in range(0, n_pair_tiles, dma_slab_pairs):
                current_pairs = min(dma_slab_pairs, n_pair_tiles - slab_base)
                task_group = TaskGroup()
                for pair in range(slab_base, slab_base + current_pairs):
                    for row in range(N_AIE_ROWS):
                        chain = row // cascade_rows
                        shard = row % cascade_rows
                        m_tile = pair * n_chains + chain
                        a_tap = TensorAccessPattern(
                            (M, K),
                            offset=m_tile * m * K + shard * k,
                            sizes=[n_rounds, chunks_per_shard, m, k],
                            strides=[0, cascade_rows * k, K, 1],
                        )
                        A_hs[row].fill(A, tap=a_tap, group=task_group)
                        A_taps.append(a_tap)

                    for col in range(n_cols):
                        b_tap = TensorAccessPattern(
                            (config.prepared_bytes,),
                            offset=col * bytes_per_column,
                            sizes=[
                                n_rounds,
                                cascade_rows * chunks_per_shard,
                                config.packed_rows,
                                k,
                            ],
                            strides=[
                                b_panel_bytes,
                                b_tile_bytes,
                                k,
                                1,
                            ],
                        )
                        B_hs[col].fill(B, tap=b_tap, group=task_group)
                        B_taps.append(b_tap)

                for chain in range(n_chains):
                    for col in range(n_cols):
                        m_tile = slab_base * n_chains + chain
                        c_tap = TensorAccessPattern(
                            (M, N),
                            offset=m_tile * m * N + col * n,
                            sizes=[current_pairs, n_rounds, m, n],
                            strides=[
                                n_chains * m * N,
                                n_cols * n,
                                N,
                                1,
                            ],
                        )
                        C_hs[chain * n_cols + col].drain(
                            C,
                            tap=c_tap,
                            wait=True,
                            group=task_group,
                        )
                        C_taps.append(c_tap)
                task_group.finish()
            return

        for n_round in range(n_rounds):
            for weight_base in range(
                0, n_pair_tiles, weight_replay_pairs
            ):
                weight_group = TaskGroup()
                for col in range(n_cols):
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=col * bytes_per_column + n_round * b_panel_bytes,
                        sizes=[
                            cascade_rows,
                            chunks_per_shard,
                            config.packed_rows,
                            k,
                        ],
                        strides=[
                            b_shard_bytes,
                            b_tile_bytes,
                            k,
                            1,
                        ],
                    )
                    B_hs[col].fill(B, tap=b_tap, group=weight_group)
                    B_taps.append(b_tap)

                weight_end = weight_base + weight_replay_pairs
                for slab_base in range(
                    weight_base, weight_end, dma_slab_pairs
                ):
                    task_group = TaskGroup()
                    for row in range(N_AIE_ROWS):
                        chain = row // cascade_rows
                        shard = row % cascade_rows
                        m_tile = slab_base * n_chains + chain
                        a_tap = TensorAccessPattern(
                            (M, K),
                            offset=m_tile * m * K + shard * k,
                            sizes=[
                                dma_slab_pairs,
                                chunks_per_shard,
                                m,
                                k,
                            ],
                            strides=[
                                n_chains * m * K,
                                cascade_rows * k,
                                K,
                                1,
                            ],
                        )
                        A_hs[row].fill(A, tap=a_tap, group=task_group)
                        A_taps.append(a_tap)

                    for chain in range(n_chains):
                        for col in range(n_cols):
                            n_tile = n_round * n_cols + col
                            m_tile = slab_base * n_chains + chain
                            c_tap = TensorAccessPattern(
                                (M, N),
                                offset=m_tile * m * N + n_tile * n,
                                sizes=[dma_slab_pairs, 1, m, n],
                                strides=[n_chains * m * N, 0, N, 1],
                            )
                            C_hs[chain * n_cols + col].drain(
                                C,
                                tap=c_tap,
                                wait=True,
                                group=task_group,
                            )
                            C_taps.append(c_tap)
                    task_group.finish()
                weight_group.finish()

    runtime = Runtime(
        sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses]
    )
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        if n_cols == 8:
            raise ValueError(
                "trace requires a spare shim column; use fewer than 8 columns"
            )
        program.enable_trace(
            trace_config.trace_size,
            workers=[workers[0][1]],
            egress_shim_col=n_cols,
        )
    module = program.resolve_program()
    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


def _build_cascade_chunked_design(
    dev,
    config: Q4KSConfig,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    """Cascade 256-K partials and retain the full result tile in FP32."""

    M, K, N = config.M, config.K, config.N
    m, n = config.m_c, config.n
    n_cols = config.n_aie_cols
    shard_k = K // N_AIE_ROWS
    chunks_per_shard = shard_k // CASCADE_CHUNK_K
    row_blocks = m // 16
    n_rounds = N // (n * n_cols)
    n_m_tiles = M // m
    output_tiles_per_column = n_m_tiles * n_rounds
    dma_slab_rows = next(
        replay
        for replay in range(min(4, n_m_tiles), 0, -1)
        if n_m_tiles % replay == 0
    )
    weight_replay_rows = next(
        replay
        for replay in range(min(32, n_m_tiles), 0, -1)
        if n_m_tiles % replay == 0 and replay % dma_slab_rows == 0
    )

    a_panel_values = m * CASCADE_CHUNK_K
    a_block_values = 16 * CASCADE_CHUNK_K
    b_chunk_bytes = CASCADE_CHUNK_K * n * 9 // 8
    b_row_bytes = shard_k * n * 9 // 8
    b_panel_bytes = K * n * 9 // 8

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(config.prepared_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    A_panel_ty = np.ndarray[(a_panel_values,), np.dtype[bfloat16]]
    A_block_ty = np.ndarray[(a_block_values,), np.dtype[bfloat16]]
    B_panel_ty = np.ndarray[(b_panel_bytes,), np.dtype[np.uint8]]
    B_row_ty = np.ndarray[(b_row_bytes,), np.dtype[np.uint8]]
    B_chunk_ty = np.ndarray[(b_chunk_bytes,), np.dtype[np.uint8]]
    C_stream_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]
    accumulator_ty = np.ndarray[(m * n,), np.dtype[np.float32]]

    put_only, put_get, accumulate, zero, stream = _cascade_chunked_kernels(
        config, A_block_ty, B_chunk_ty, accumulator_ty
    )

    a_to_stream: StreamDims = [
        (row_blocks, 16 * CASCADE_CHUNK_K),
        (CASCADE_CHUNK_K // 8, 8),
        (16, CASCADE_CHUNK_K),
        (8, 1),
    ]
    a_from_stream: StreamDims = [
        (CASCADE_CHUNK_K // 8, 64),
        (2, 8 * CASCADE_CHUNK_K),
        (64, 1),
    ]

    A_l3l2: list[ObjectFifo] = []
    A_l2l1: list[ObjectFifo] = []
    for row in range(N_AIE_ROWS):
        parent = ObjectFifo(
            A_panel_ty,
            name=f"A_CHUNK_L3L2_{row}",
            depth=2,
        )
        child = ObjectFifo(
            A_panel_ty,
            consumer_obj_type=A_block_ty,
            name=f"A_CHUNK_L2L1_{row}",
            depth=1,
            dims_to_stream=a_to_stream,
            dims_from_stream_per_cons=a_from_stream,
        )
        ObjectFifoLink(parent.cons(), child.prod(), tile=Tile(row, 1))
        A_l3l2.append(parent)
        A_l2l1.append(child)

    B_l3l2: list[ObjectFifo] = []
    B_l2l1: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        parent = ObjectFifo(
            B_panel_ty,
            name=f"B_CHUNK_L3L2_{col}",
            depth=1,
        )
        children = [
            ObjectFifo(
                B_row_ty,
                consumer_obj_type=B_chunk_ty,
                name=f"B_CHUNK_L2L1_{col}_{row}",
                depth=1,
                repeat_count=weight_replay_rows,
            )
            for row in range(N_AIE_ROWS)
        ]
        ObjectFifoLink(
            parent.cons(),
            [child.prod() for child in children],
            tile=Tile(col, 1),
            dst_offsets=[row * b_row_bytes for row in range(N_AIE_ROWS)],
        )
        B_l3l2.append(parent)
        for row in range(N_AIE_ROWS):
            B_l2l1[row].append(children[row])

    C_streams = [
        ObjectFifo(
            C_stream_ty,
            name=f"C_CHUNK_L1L3_{col}",
            depth=2,
            aie_stream=(0, 0),
        )
        for col in range(n_cols)
    ]

    def bottom_fn(in_a, in_b, put_fn):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            for _chunk in range(chunks_per_shard):
                elem_b = in_b.acquire(1)
                for _rb in range(row_blocks):
                    elem_a = in_a.acquire(1)
                    put_fn(elem_a, elem_b)
                    in_a.release(1)
                in_b.release(1)

    def middle_fn(in_a, in_b, put_get_fn):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            for _chunk in range(chunks_per_shard):
                elem_b = in_b.acquire(1)
                for _rb in range(row_blocks):
                    elem_a = in_a.acquire(1)
                    put_get_fn(elem_a, elem_b)
                    in_a.release(1)
                in_b.release(1)

    def top_fn(in_a, in_b, out_c, accumulator, zero_fn, accumulate_fn, stream_fn):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            zero_fn(accumulator)
            for _chunk in range(chunks_per_shard):
                elem_b = in_b.acquire(1)
                for rb in range(row_blocks):
                    elem_a = in_a.acquire(1)
                    accumulate_fn(elem_a, elem_b, accumulator, rb)
                    in_a.release(1)
                in_b.release(1)
            stream_fn(accumulator)

    workers: list[list[Worker]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        accumulator = Buffer(
            accumulator_ty, name=f"C_CHUNK_ACC_0_{col}"
        )
        workers[0].append(
            Worker(
                top_fn,
                [
                    A_l2l1[0].cons(),
                    B_l2l1[0][col].cons(),
                    C_streams[col].prod(),
                    accumulator,
                    zero,
                    accumulate,
                    stream,
                ],
                tile=Tile(col, 2),
                stack_size=0xD00,
                trace=1 if trace_config and col == 1 else 0,
            )
        )
        for row in (1, 2):
            workers[row].append(
                Worker(
                    middle_fn,
                    [
                        A_l2l1[row].cons(),
                        B_l2l1[row][col].cons(),
                        put_get,
                    ],
                    tile=Tile(col, row + 2),
                    stack_size=0xD00,
                )
            )
        workers[3].append(
            Worker(
                bottom_fn,
                [A_l2l1[3].cons(), B_l2l1[3][col].cons(), put_only],
                tile=Tile(col, 5),
                stack_size=0xD00,
            )
        )

    for col in range(n_cols):
        for row in range(N_AIE_ROWS - 1, 0, -1):
            CascadeFlow(workers[row][col], workers[row - 1][col])
    flat_workers = [worker for row in workers for worker in row]

    A_prods = [
        fifo.prod(tile=Tile(row, 0)) for row, fifo in enumerate(A_l3l2)
    ]
    B_prods = [
        fifo.prod(tile=Tile(col, 0)) for col, fifo in enumerate(B_l3l2)
    ]
    C_conses = [
        fifo.cons(tile=Tile(col, 0)) for col, fifo in enumerate(C_streams)
    ]
    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []
    bytes_per_column = n_rounds * b_panel_bytes

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        for n_round in range(n_rounds):
            for weight_base in range(0, n_m_tiles, weight_replay_rows):
                weight_group = TaskGroup()
                for col in range(n_cols):
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=col * bytes_per_column + n_round * b_panel_bytes,
                        sizes=[
                            N_AIE_ROWS,
                            chunks_per_shard,
                            CASCADE_CHUNK_K // 8,
                            n * 9,
                        ],
                        strides=[b_row_bytes, b_chunk_bytes, n * 9, 1],
                    )
                    B_hs[col].fill(B, tap=b_tap, group=weight_group)
                    B_taps.append(b_tap)

                weight_end = weight_base + weight_replay_rows
                for slab_base in range(weight_base, weight_end, dma_slab_rows):
                    task_group = TaskGroup()
                    for tile_offset in range(dma_slab_rows):
                        m_tile = slab_base + tile_offset
                        for row in range(N_AIE_ROWS):
                            a_tap = TensorAccessPattern(
                                (M, K),
                                offset=m_tile * m * K + row * shard_k,
                                sizes=[
                                    chunks_per_shard,
                                    m,
                                    CASCADE_CHUNK_K // 8,
                                    8,
                                ],
                                strides=[CASCADE_CHUNK_K, K, 8, 1],
                            )
                            A_hs[row].fill(A, tap=a_tap, group=task_group)
                            A_taps.append(a_tap)

                        for col in range(n_cols):
                            n_tile = n_round * n_cols + col
                            c_tap = TensorAccessPattern(
                                (M, N),
                                offset=m_tile * m * N + n_tile * n,
                                sizes=[1, 1, m, n],
                                strides=[0, 0, N, 1],
                            )
                            C_hs[col].drain(
                                C, tap=c_tap, wait=True, group=task_group
                            )
                            C_taps.append(c_tap)
                    task_group.finish()
                weight_group.finish()

    runtime = Runtime(
        sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses]
    )
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        if n_cols == 8:
            raise ValueError("trace requires a spare shim column; use fewer than 8 columns")
        program.enable_trace(
            trace_config.trace_size,
            workers=[workers[0][1]],
            egress_shim_col=n_cols,
        )
    module = program.resolve_program()
    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


def _build_cascade_register_design(
    dev,
    config: Q4KSConfig,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    """Keep each K/4 partial C microtile in accfloat registers.

    The four AIE rows compute their K shards concurrently.  A hardware
    cascade reduces four register-resident 16x16 partial tiles, after which
    the top row converts directly to the BF16 output stream.  MemTiles cache
    both the expanded B panel and one activation shard, so neither operand is
    reloaded merely to preserve the register lifetime.
    """

    M, K, N = config.M, config.K, config.N
    m, n = config.m_c, config.n
    n_cols = config.n_aie_cols
    shard_k = K // N_AIE_ROWS
    row_blocks = m // 16
    column_blocks = n // 16
    n_rounds = N // (n * n_cols)
    n_m_tiles = M // m
    output_tiles_per_column = n_m_tiles * n_rounds
    replay_rows = next(
        replay
        for replay in range(min(4, n_m_tiles), 0, -1)
        if n_m_tiles % replay == 0
    )

    a_panel_values = m * shard_k
    a_block_values = 16 * shard_k
    b_block_bytes = shard_k * 16 * 9 // 8
    b_row_bytes = shard_k * n * 9 // 8
    b_panel_bytes = K * n * 9 // 8

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(config.prepared_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    A_panel_ty = np.ndarray[(a_panel_values,), np.dtype[bfloat16]]
    A_block_ty = np.ndarray[(a_block_values,), np.dtype[bfloat16]]
    B_panel_ty = np.ndarray[(b_panel_bytes,), np.dtype[np.uint8]]
    B_row_ty = np.ndarray[(b_row_bytes,), np.dtype[np.uint8]]
    B_block_ty = np.ndarray[(b_block_bytes,), np.dtype[np.uint8]]
    C_stream_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]

    put_only, put_get, get_stream = _cascade_register_kernels(
        config, A_block_ty, B_block_ty
    )

    # The MemTile reads a row-major activation shard one 16-row group at a
    # time.  The compute-tile DMA writes each group in 8x8 A microtiles.
    a_to_stream: StreamDims = [
        (row_blocks, 16 * shard_k),
        (shard_k // 8, 8),
        (16, shard_k),
        (8, 1),
    ]
    a_from_stream: StreamDims = [
        (shard_k // 8, 64),
        (2, 8 * shard_k),
        (64, 1),
    ]

    A_l3l2: list[ObjectFifo] = []
    A_l2l1: list[ObjectFifo] = []
    for row in range(N_AIE_ROWS):
        parent = ObjectFifo(
            A_panel_ty,
            name=f"A_REGISTER_L3L2_{row}",
            depth=1,
        )
        child = ObjectFifo(
            A_panel_ty,
            consumer_obj_type=A_block_ty,
            name=f"A_REGISTER_L2L1_{row}",
            depth=1,
            dims_to_stream=a_to_stream,
            dims_from_stream_per_cons=a_from_stream,
        )
        ObjectFifoLink(parent.cons(), child.prod(), tile=Tile(row, 1))
        A_l3l2.append(parent)
        A_l2l1.append(child)

    B_l3l2: list[ObjectFifo] = []
    B_l2l1: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        parent = ObjectFifo(
            B_panel_ty,
            name=f"B_REGISTER_L3L2_{col}",
            depth=1,
        )
        children = [
            ObjectFifo(
                B_row_ty,
                consumer_obj_type=B_block_ty,
                name=f"B_REGISTER_L2L1_{col}_{row}",
                depth=1,
                repeat_count=replay_rows * row_blocks,
            )
            for row in range(N_AIE_ROWS)
        ]
        ObjectFifoLink(
            parent.cons(),
            [child.prod() for child in children],
            tile=Tile(col, 1),
            dst_offsets=[row * b_row_bytes for row in range(N_AIE_ROWS)],
        )
        B_l3l2.append(parent)
        for row in range(N_AIE_ROWS):
            B_l2l1[row].append(children[row])

    C_streams = [
        ObjectFifo(
            C_stream_ty,
            name=f"C_REGISTER_L1L3_{col}",
            depth=2,
            aie_stream=(0, 0),
        )
        for col in range(n_cols)
    ]

    def bottom_fn(in_a, in_b, put_fn):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            for _rb in range(row_blocks):
                elem_a = in_a.acquire(1)
                for _nb in range(column_blocks):
                    elem_b = in_b.acquire(1)
                    put_fn(elem_a, elem_b)
                    in_b.release(1)
                in_a.release(1)

    def middle_fn(in_a, in_b, put_get_fn):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            for _rb in range(row_blocks):
                elem_a = in_a.acquire(1)
                for _nb in range(column_blocks):
                    elem_b = in_b.acquire(1)
                    put_get_fn(elem_a, elem_b)
                    in_b.release(1)
                in_a.release(1)

    def top_fn(in_a, in_b, out_c, get_fn):
        tiles = (
            range_(output_tiles_per_column)
            if output_tiles_per_column > 1
            else range(1)
        )
        for _ in tiles:
            for rb in range(row_blocks):
                elem_a = in_a.acquire(1)
                for nb in range(column_blocks):
                    elem_b = in_b.acquire(1)
                    last = int(
                        rb + 1 == row_blocks and nb + 1 == column_blocks
                    )
                    get_fn(elem_a, elem_b, last)
                    in_b.release(1)
                in_a.release(1)

    workers: list[list[Worker]] = [[] for _ in range(N_AIE_ROWS)]
    for col in range(n_cols):
        workers[0].append(
            Worker(
                top_fn,
                [
                    A_l2l1[0].cons(),
                    B_l2l1[0][col].cons(),
                    C_streams[col].prod(),
                    get_stream,
                ],
                tile=Tile(col, 2),
                stack_size=0xD00,
                trace=1 if trace_config and col == 1 else 0,
            )
        )
        for row in (1, 2):
            workers[row].append(
                Worker(
                    middle_fn,
                    [
                        A_l2l1[row].cons(),
                        B_l2l1[row][col].cons(),
                        put_get,
                    ],
                    tile=Tile(col, row + 2),
                    stack_size=0xD00,
                )
            )
        workers[3].append(
            Worker(
                bottom_fn,
                [A_l2l1[3].cons(), B_l2l1[3][col].cons(), put_only],
                tile=Tile(col, 5),
                stack_size=0xD00,
            )
        )

    for col in range(n_cols):
        for row in range(N_AIE_ROWS - 1, 0, -1):
            CascadeFlow(workers[row][col], workers[row - 1][col])
    flat_workers = [worker for row in workers for worker in row]

    A_prods = [
        fifo.prod(tile=Tile(row, 0)) for row, fifo in enumerate(A_l3l2)
    ]
    B_prods = [
        fifo.prod(tile=Tile(col, 0)) for col, fifo in enumerate(B_l3l2)
    ]
    C_conses = [
        fifo.cons(tile=Tile(col, 0)) for col, fifo in enumerate(C_streams)
    ]
    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []
    bytes_per_column = n_rounds * b_panel_bytes

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        for n_round in range(n_rounds):
            for slab_base in range(0, n_m_tiles, replay_rows):
                task_group = TaskGroup()
                for col in range(n_cols):
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=col * bytes_per_column + n_round * b_panel_bytes,
                        sizes=[N_AIE_ROWS, column_blocks, shard_k // 8, 144],
                        strides=[b_row_bytes, b_block_bytes, 144, 1],
                    )
                    B_hs[col].fill(B, tap=b_tap, group=task_group)
                    B_taps.append(b_tap)

                for tile_offset in range(replay_rows):
                    m_tile = slab_base + tile_offset
                    for row in range(N_AIE_ROWS):
                        a_tap = TensorAccessPattern(
                            (M, K),
                            offset=m_tile * m * K + row * shard_k,
                            sizes=[1, m, shard_k // 8, 8],
                            strides=[0, K, 8, 1],
                        )
                        A_hs[row].fill(A, tap=a_tap, group=task_group)
                        A_taps.append(a_tap)

                    for col in range(n_cols):
                        n_tile = n_round * n_cols + col
                        c_tap = TensorAccessPattern(
                            (M, N),
                            offset=m_tile * m * N + n_tile * n,
                            sizes=[row_blocks, column_blocks, 16, 16],
                            strides=[16 * N, 16, N, 1],
                        )
                        C_hs[col].drain(
                            C, tap=c_tap, wait=True, group=task_group
                        )
                        C_taps.append(c_tap)
                task_group.finish()

    runtime = Runtime(
        sequence, [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses]
    )
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        if n_cols == 8:
            raise ValueError("trace requires a spare shim column; use fewer than 8 columns")
        program.enable_trace(
            trace_config.trace_size,
            workers=[workers[0][1]],
            egress_shim_col=n_cols,
        )
    module = program.resolve_program()
    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


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
    accumulation_mode: str,
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
        accumulation_mode=accumulation_mode,
        cache_mode=cache_mode,
        activation_input=activation_input,
        cache_k=cache_k,
    )
    if accumulation_mode == "cascade-hybrid":
        return _build_cascade_hybrid_2way_design(
            dev, config, trace_config, generate_taps=generate_taps
        )
    if accumulation_mode == "cascade-shared":
        return _build_cascade_shared_design(
            dev, config, trace_config, generate_taps=generate_taps
        )
    if accumulation_mode == "cascade-chunked":
        return _build_cascade_chunked_design(
            dev, config, trace_config, generate_taps=generate_taps
        )
    if accumulation_mode == "cascade-register":
        return _build_cascade_register_design(
            dev, config, trace_config, generate_taps=generate_taps
        )
    if accumulation_mode == "cascade-resident":
        return _build_cascade_resident_design(
            dev, config, trace_config, generate_taps=generate_taps
        )
    if accumulation_mode == "cascade":
        return _build_cascade_design(
            dev, config, trace_config, generate_taps=generate_taps
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
    # Eight hardware replays currently stall on NPU2.  Counts through four
    # are validated, so larger M dimensions use multiple replay slabs.  A
    # divisor gives every producer object the one static repeat count required
    # by ObjectFifo lowering.
    memtile_replay_rows = next(
        replay
        for replay in range(min(4, n_row_blocks), 0, -1)
        if n_row_blocks % replay == 0
    )
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
    accumulator_ty = (
        np.ndarray[(m_c * n,), np.dtype[np.float32]]
        if accumulation_mode == "fp32"
        else None
    )
    matmul_kernel, zero_kernel, store_kernel = _kernels(
        config, kernel_a_ty, kernel_b_ty, kernel_c_ty, accumulator_ty
    )

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
            obj_types=[A_l1_ty] * a_rows_per_shim,
            names=[f"A_L2L1_{row}" for row in range(start, stop)],
            depths=[config.a_fifo_depth] * a_rows_per_shim,
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
        if cache_mode == "memtile-weight":
            # One producer object is the entire compressed full-K panel while
            # each consumer acquisition is one packed Q4_K compute tile.  The
            # repeat therefore applies to a panel (not to each K tile).  Large
            # M dimensions reload the panel between bounded replay slabs.
            child_b = ObjectFifo(
                B_l2_ty,
                consumer_obj_type=B_l1_ty,
                depth=1,
                repeat_count=memtile_replay_rows,
                name=f"B_L2L1_{col}",
            )
            ObjectFifoLink(
                parent_b.cons(), child_b.prod(), tile=Tile(col, 1)
            )
        else:
            child_b = parent_b.cons().forward(
                obj_type=B_l1_ty,
                depth=1,
                name=f"B_L2L1_{col}",
            )
        B_l2l1.append(child_b)
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

    def core_fn_fp32(in_a, in_b, out_c, accumulator, zero, matmul, store):
        tile_loop = range_(n_tiles_per_core) if n_tiles_per_core > 1 else range(1)
        for _ in tile_loop:
            elem_c = out_c.acquire(1)
            zero(accumulator)
            k_loop = range_(K // k) if K // k > 1 else range(1)
            for _ in k_loop:
                elem_b = in_b.acquire(1)
                for subtile in range(config.a_subtiles):
                    elem_a = in_a.acquire(1)
                    matmul(elem_a, elem_b, accumulator, subtile)
                    in_a.release(1)
                in_b.release(1)
            store(accumulator, elem_c)
            out_c.release(1)

    def make_worker(row, col):
        common = [
            A_l2l1[row].cons(),
            B_l2l1[col].cons(),
            C_l1l2[row][col].prod(),
        ]
        trace = 1 if trace_config and row * n_aie_cols + col == 1 else 0
        if accumulation_mode == "fp32":
            accumulator = Buffer(
                accumulator_ty, name=f"C_accumulator_{row}_{col}"
            )
            return Worker(
                core_fn_fp32,
                common
                + [accumulator, zero_kernel, matmul_kernel, store_kernel],
                stack_size=0xD00,
                trace=trace,
            )
        return Worker(
            core_fn,
            common + [zero_kernel, matmul_kernel],
            stack_size=0xD00,
            trace=trace,
        )

    workers = Worker.grid(N_AIE_ROWS, n_aie_cols, make_worker)
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
            tiles_per_col = config.n_n_tiles // n_aie_cols
            for n_round in range(tiles_per_col):
                for slab_base in range(0, n_row_blocks, memtile_replay_rows):
                    task_group = TaskGroup()
                    # Load each full-K compressed panel once per bounded slab;
                    # the MemTile DMA replays its K descriptors for each row.
                    for col in range(n_aie_cols):
                        b_tap = B_tiles[n_round * n_aie_cols + col]
                        B_hs[col].fill(B, tap=b_tap, group=task_group)
                        B_taps.append(b_tap)

                    slab_end = slab_base + memtile_replay_rows
                    for row_base in range(slab_base, slab_end, 2):
                        current_rows = min(2, slab_end - row_base)
                        for col in range(n_aie_cols):
                            n_tile = n_round * n_aie_cols + col
                            c_tap = TensorAccessPattern(
                                (M, N),
                                offset=(
                                    row_base * N_AIE_ROWS * m_c * N
                                    + n_tile * n
                                ),
                                sizes=[current_rows, 1, N_AIE_ROWS * m_c, n],
                                strides=[N_AIE_ROWS * m_c * N, 0, N, 1],
                            )
                            C_hs[col].drain(
                                C, tap=c_tap, wait=True, group=task_group
                            )
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
                                            + col
                                            * a_rows_per_shim
                                            * m_c
                                            * K
                                        ),
                                        sizes=[
                                            1,
                                            config.n_k_tiles,
                                            m_c * a_rows_per_shim,
                                            k,
                                        ],
                                        strides=[0, k, K, 1],
                                    )
                                    A_hs[col].fill(
                                        A, tap=a_tap, group=task_group
                                    )
                                    A_taps.append(a_tap)
                    task_group.finish()
            return

        c_index = 0
        tg = TaskGroup()
        # A joined C object has N_AIE_ROWS * m_c rows.  NPU DMA dimensions
        # are limited to 1023, so large m_c tiles use one row block per drain
        # and expose the four joined compute rows as a separate dimension.
        c_row_batch = 1 if N_AIE_ROWS * m_c > 1023 else 2
        n_rounds = N // n // n_aie_cols
        for row_base in range(0, n_row_blocks, c_row_batch):
            current_rows = min(c_row_batch, n_row_blocks - row_base)
            for col in range(n_aie_cols):
                if c_row_batch == 1:
                    c_tap = TensorAccessPattern(
                        (M, N),
                        offset=row_base * N_AIE_ROWS * m_c * N + col * n,
                        sizes=[n_rounds, N_AIE_ROWS, m_c, n],
                        strides=[n * n_aie_cols, m_c * N, N, 1],
                    )
                else:
                    c_tap = C_tiles[c_index]
                    c_index += 1
                C_hs[col].drain(C, tap=c_tap, wait=True, group=tg)
                C_taps.append(c_tap)
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
    accumulation_mode: CompileTime[str] = "bf16",
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
        accumulation_mode,
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
        accumulation_mode="bf16",
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
    parser.add_argument(
        "--accumulation-mode", choices=ACCUMULATION_MODES, default="bf16"
    )
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
        accumulation_mode=opts.accumulation_mode,
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
        accumulation_mode=cfg.accumulation_mode,
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
        if cfg.accumulation_mode == "cascade-hybrid":
            cascade_rows = 2
            chunks_per_row = cfg.K // (cascade_rows * cfg.k)
            partials = []
            for cascade_row in range(cascade_rows):
                partial = 0.0
                for chunk in range(chunks_per_row):
                    k0 = (chunk * cascade_rows + cascade_row) * cfg.k
                    ks = slice(k0, k0 + cfg.k)
                    partial += float(
                        activation[ks] @ weights[ks].astype(np.float32)
                    )
                    if chunk + 1 != chunks_per_row:
                        partial = float(
                            np.asarray(partial, dtype=bfloat16)
                        )
                partials.append(partial)
            stored = partials[-1]
            for cascade_row in range(cascade_rows - 2, -1, -1):
                stored = float(np.float32(stored) + np.float32(partials[cascade_row]))
            values[i] = float(np.asarray(stored, dtype=bfloat16))
            continue

        stored = 0.0
        for k0 in range(0, cfg.K, cfg.k):
            ks = slice(k0, k0 + cfg.k)
            stored += float(activation[ks] @ weights[ks].astype(np.float32))
            if cfg.accumulation_mode == "bf16":
                stored = float(np.asarray(stored, dtype=bfloat16))
        values[i] = float(np.asarray(stored, dtype=bfloat16))
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
    prepared = prepare_q4ks_weights(native, cfg, cfg.weight_storage_type)
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
        "accumulation_mode": cfg.accumulation_mode,
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
        raise RuntimeError(
            f"performance gate failed: {median:.2f} < "
            f"{opts.min_gflops:.2f} GFLOP/s"
        )
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
