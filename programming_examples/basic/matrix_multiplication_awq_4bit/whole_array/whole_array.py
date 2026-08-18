# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Configurable NPU2 whole-array BF16 x AWQ-INT4 matrix multiplication.

This is the current ``@iron.jit`` whole-array dataflow with a packed uint4 B
operand.  The host-visible B buffer follows ``../packing.py``.  Each core
dequantizes one packed B tile to BF16 scratch, then consumes two streamed A
halves with an explicit half index; no mutable call-order state is used.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie import ir
from aie.helpers.taplib import (
    TensorAccessPattern,
    TensorAccessSequence,
    TensorTiler2D,
)
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
from aie.utils.hostruntime.argparse import (
    add_benchmark_args,
    add_compile_args,
    add_trace_arg,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.trace import TraceConfig
from ml_dtypes import bfloat16

_EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
if str(_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_ROOT))

from packing import (  # noqa: E402
    AWQConfig,
    N_AIE_ROWS,
    b_partition_taps,
    make_deterministic_inputs,
    pack_awq_weights,
    reference_matmul,
    reference_samples,
)

_KERNEL_SOURCE = str(Path(__file__).resolve().parent / "awq_4bit.cc")
_VERIFY_SCALAR_PRODUCT_THRESHOLD = 1024 * 1024 * 1024
_REFERENCE_BF16_BASELINE_GFLOPS = 3250.3


def _use_local_trace_triggers(module):
    """Start one representative trace locally without a broadcast overlay."""

    text = str(module)
    start = "aie.trace.start broadcast = 15"
    stop = "aie.trace.stop broadcast = 14"
    if text.count(start) != 1 or text.count(stop) != 1:
        raise RuntimeError("expected exactly one representative worker trace")
    text = text.replace(start, "aie.trace.start event = <\"TRUE\">")
    text = text.replace(stop, "aie.trace.stop event = <\"NONE\">")
    return ir.Module.parse(text, context=module.context)


def _device_for(dev_str: str, n_aie_cols: int):
    if dev_str != "npu2":
        raise ValueError("matrix_multiplication_awq_4bit is NPU2-only")
    if n_aie_cols not in (1, 2, 4, 8):
        raise ValueError("n_aie_cols must be one of 1, 2, 4, or 8")
    device = from_name("npu2", n_cols=None)
    if resolve_target_arch(device) != "aie2p":
        raise ValueError("the selected target is not an NPU2 device")
    return device


def _dtype_out(dtype_out_str: str):
    if dtype_out_str == "bf16":
        return bfloat16
    if dtype_out_str == "f32":
        return np.float32
    raise ValueError("dtype_out_str must be 'bf16' or 'f32'")


def _awq_kernels(
    *,
    m: int,
    k: int,
    n: int,
    group_size: int,
    tile_bytes: int,
    dtype_out_str: str,
    use_chess: bool,
    a_ty,
    b_ty,
    c_ty,
) -> tuple[ExternalFunction, Kernel]:
    suffix = "bf16" if dtype_out_str == "bf16" else "f32"
    object_name = f"awq_4bit_{m}x{k}x{n}_g{group_size}_{suffix}.o"
    compile_flags = [
        f"-DDIM_M={m}",
        f"-DDIM_K={k}",
        f"-DDIM_N={n}",
        f"-DGROUP_SIZE={group_size}",
        f"-DPACKED_TILE_BYTES={tile_bytes}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        f"-DAWQ_OUTPUT_{suffix.upper()}",
    ]
    matmul = ExternalFunction(
        f"awq_matmul_bf16_{suffix}",
        object_file_name=object_name,
        source_file=_KERNEL_SOURCE,
        arg_types=[a_ty, b_ty, c_ty, np.int32],
        include_dirs=[aie_config.cxx_header_path()],
        compile_flags=compile_flags,
        use_chess=use_chess,
    )
    zero = Kernel(f"awq_zero_{suffix}", matmul.object_file_name, [c_ty])
    return matmul, zero


def _build_design(
    dev,
    M: int,
    K: int,
    N: int,
    m: int,
    k: int,
    n: int,
    group_size: int,
    n_aie_cols: int,
    dtype_out_str: str,
    use_chess: bool,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    if resolve_target_arch(dev) != "aie2p":
        raise ValueError("matrix_multiplication_awq_4bit is NPU2-only")
    if not use_chess:
        raise ValueError("this example supports the Chess kernel backend only")

    awq = AWQConfig(
        M=M,
        K=K,
        N=N,
        m=m,
        k=k,
        n=n,
        group_size=group_size,
        n_aie_cols=n_aie_cols,
        dtype_out=dtype_out_str,
    )
    dtype_out = _dtype_out(dtype_out_str)
    fifo_depth = 2
    c_fifo_depth = awq.output_fifo_depth
    n_aie_cores = N_AIE_ROWS * n_aie_cols
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
    n_shim_mem_A = min(N_AIE_ROWS, n_aie_cols)
    n_A_tiles_per_shim = N_AIE_ROWS // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(awq.packed_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[
        (m * k * n_A_tiles_per_shim,), np.dtype[bfloat16]
    ]
    A_l1_ty = np.ndarray[(m // 2, k), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[(awq.packed_rows, k), np.dtype[np.uint8]]
    B_l1_ty = B_l2_ty
    C_l2_ty = np.ndarray[(m * n * N_AIE_ROWS,), np.dtype[dtype_out]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    kernel_A_ty = np.ndarray[((m // 2) * k,), np.dtype[bfloat16]]
    kernel_B_ty = np.ndarray[(awq.tile_bytes,), np.dtype[np.uint8]]
    kernel_C_ty = np.ndarray[(m * n,), np.dtype[dtype_out]]
    matmul_kernel, zero_kernel = _awq_kernels(
        m=m,
        k=k,
        n=n,
        group_size=group_size,
        tile_bytes=awq.tile_bytes,
        dtype_out_str=dtype_out_str,
        use_chess=use_chess,
        a_ty=kernel_A_ty,
        b_ty=kernel_B_ty,
        c_ty=kernel_C_ty,
    )

    A_l3l2_fifos: list[ObjectFifo] = []
    A_l2l1_fifos: list[ObjectFifo] = []
    B_l3l2_fifos: list[ObjectFifo] = []
    B_l2l1_fifos: list[ObjectFifo] = []
    C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    C_l2l3_fifos: list[ObjectFifo] = []

    # One full (m, k) A tile stays in each MemTile object.  Its stream emits
    # two (m/2, k) objects and scatters each into the 8x8 microtile layout
    # consumed by the AIE2P MMUL kernel.
    a_to_stream: StreamDims = [
        (2, m * k // 2),
        (k // 8, 8),
        (m // 2, k),
        (8, 1),
    ]
    a_from_stream: StreamDims = [
        (k // 8, 64),
        (m // 16, 8 * k),
        (64, 1),
    ]
    for shim in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{shim}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = shim * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        row_count = stop_row - start_row
        children = a_l3l2.cons().split(
            [m * k * row for row in range(row_count)],
            depths=[fifo_depth] * row_count,
            obj_types=[A_l1_ty] * row_count,
            names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
            dims_to_stream=[a_to_stream] * row_count,
            dims_from_stream=[a_from_stream] * row_count,
        )
        A_l2l1_fifos.extend(children)

    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(
                obj_type=B_l1_ty,
                depth=fifo_depth,
                name=f"B_L2L1_{col}",
            )
        )

        c_dims: StreamDims = [
            (m // 8, 8 * n),
            (8, 8),
            (n // 8, 64),
            (8, 1),
        ]
        c_l2l3 = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=c_fifo_depth,
            dims_to_stream=c_dims,
        )
        C_l2l3_fifos.append(c_l2l3)
        children = c_l2l3.prod().join(
            [m * n * row for row in range(N_AIE_ROWS)],
            depths=[c_fifo_depth] * N_AIE_ROWS,
            obj_types=[C_l1_ty] * N_AIE_ROWS,
            names=[f"C_L1L2_{col}_{row}" for row in range(N_AIE_ROWS)],
        )
        for row in range(N_AIE_ROWS):
            C_l1l2_fifos[row].append(children[row])

    def core_fn(in_a, in_b, out_c, zero, matmul):
        tile_loop = range(1)
        if n_tiles_per_core > 1:
            tile_loop = range_(n_tiles_per_core)
        for _ in tile_loop:
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k) if K // k > 1 else range(1):
                elem_in_b = in_b.acquire(1)
                elem_in_a = in_a.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out, 0)
                in_a.release(1)
                elem_in_a = in_a.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out, 1)
                in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        N_AIE_ROWS,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
            trace=1 if trace_config and row * n_aie_cols + col == 1 else 0,
        ),
    )
    flat_workers = [worker for row in workers for worker in row]
    trace_workers = [flat_workers[1]] if trace_config else []

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    B_tiles = [
        TensorAccessPattern(
            (awq.packed_bytes,),
            offset=tap.offset,
            sizes=list(tap.sizes),
            strides=list(tap.strides),
        )
        for tap in b_partition_taps(awq)
    ]
    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (m * N_AIE_ROWS, n),
        tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
        prune_step=False,
    )

    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []
    A_prods = [fifo.prod() for fifo in A_l3l2_fifos]
    B_prods = [fifo.prod() for fifo in B_l3l2_fifos]
    C_conses = [fifo.cons() for fifo in C_l2l3_fifos]

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        c_index = 0
        task_group = TaskGroup()
        n_row_blocks = M // m // N_AIE_ROWS
        for transfer_block in range(iron.ceildiv(n_row_blocks, tb_max_n_rows)):
            for pingpong in (0, 1):
                if c_index >= len(C_tiles):
                    break
                row_base = (
                    transfer_block * tb_max_n_rows
                    + pingpong * tb_max_n_rows // 2
                )
                current_rows = min(tb_max_n_rows // 2, n_row_blocks - row_base)

                for col in range(n_aie_cols):
                    C_hs[col].drain(
                        C, tap=C_tiles[c_index], wait=True, group=task_group
                    )
                    C_taps.append(C_tiles[c_index])
                    c_index += 1

                    for tile_row in range(current_rows):
                        tile_offset = (
                            (row_base + tile_row) * n_shim_mem_A + col
                        ) % len(A_tiles)
                        if col < n_shim_mem_A:
                            A_hs[col].fill(
                                A, tap=A_tiles[tile_offset], group=task_group
                            )
                            A_taps.append(A_tiles[tile_offset])
                        B_hs[col].fill(B, tap=B_tiles[col], group=task_group)
                        B_taps.append(B_tiles[col])

                if transfer_block > 0 or pingpong > 0:
                    task_group.finish()
                    task_group = TaskGroup()
        task_group.finish()

    runtime = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses],
    )
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        program.enable_trace(
            trace_config.trace_size,
            workers=trace_workers,
            egress_shim_col=n_aie_cols if n_aie_cols < 8 else 0,
        )
    module = program.resolve_program()
    if trace_config:
        module = _use_local_trace_triggers(module)

    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array_awq_4bit(
    A: In,
    packed_B: In,
    C: Out,
    *,
    M: CompileTime[int] = 1024,
    K: CompileTime[int] = 1024,
    N: CompileTime[int] = 2048,
    m: CompileTime[int] = 64,
    k: CompileTime[int] = 128,
    n: CompileTime[int] = 64,
    group_size: CompileTime[int] = 128,
    n_aie_cols: CompileTime[int] = 8,
    dtype_out_str: CompileTime[str] = "bf16",
    use_chess: CompileTime[bool] = True,
    trace_config: CompileTime[TraceConfig | None] = None,
):
    return _build_design(
        iron.get_current_device(),
        M,
        K,
        N,
        m,
        k,
        n,
        group_size,
        n_aie_cols,
        dtype_out_str,
        use_chess,
        trace_config,
    )


# Friendly alias for code that expects the current example's ``whole_array`` name.
whole_array = whole_array_awq_4bit


def generate_taps(
    M: int = 1024,
    K: int = 1024,
    N: int = 2048,
    m: int = 64,
    k: int = 128,
    n: int = 64,
    group_size: int = 128,
    n_aie_cols: int = 8,
    dtype_out_str: str = "bf16",
):
    """Return ``(A, packed-B, C)`` TAP sequences for visualization/tests."""

    dev = _device_for("npu2", n_aie_cols)
    iron.set_current_device(dev)
    return _build_design(
        dev,
        M,
        K,
        N,
        m,
        k,
        n,
        group_size,
        n_aie_cols,
        dtype_out_str,
        True,
        None,
        generate_taps=True,
    )


def _make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="NPU2 Whole-Array AWQ INT4 Matrix Multiplication"
    )
    add_compile_args(
        parser,
        short_dev=None,
        dev_choices=("npu2",),
        default_dev="npu2",
    )
    parser.add_argument("-M", type=int, default=1024)
    parser.add_argument("-K", type=int, default=1024)
    parser.add_argument("-N", type=int, default=2048)
    parser.add_argument("-m", type=int, default=64)
    parser.add_argument("-k", type=int, default=128)
    parser.add_argument("-n", type=int, default=64)
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128], default=128)
    parser.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=8)
    parser.add_argument("--dtype_out", choices=["bf16", "f32"], default="bf16")
    parser.add_argument("--use-chess", type=int, choices=[1], default=1)
    parser.add_argument(
        "--verify-mode",
        choices=["auto", "full", "sampled", "none"],
        default="auto",
    )
    parser.add_argument("--verify-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1726250518)
    parser.add_argument(
        "--benchmark-repeats",
        type=int,
        default=1,
        help="independent warmup/timed benchmark rounds",
    )
    parser.add_argument(
        "--min-gflops",
        type=float,
        default=0.0,
        help="fail when median round-average throughput is below this value",
    )
    add_trace_arg(parser, with_short=False)
    add_benchmark_args(parser, default_warmup=1, default_iters=1)
    return parser


def _awq_config_from_opts(opts) -> AWQConfig:
    return AWQConfig(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        group_size=opts.group_size,
        n_aie_cols=opts.n_aie_cols,
        dtype_out=opts.dtype_out,
    )


def _trace_config(opts) -> TraceConfig | None:
    return TraceConfig(trace_size=opts.trace_size) if opts.trace_size > 0 else None


def _compile_kwargs(opts) -> dict:
    return dict(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        group_size=opts.group_size,
        n_aie_cols=opts.n_aie_cols,
        dtype_out_str=opts.dtype_out,
        use_chess=bool(opts.use_chess),
        trace_config=_trace_config(opts),
    )


def _validate_cli(opts) -> None:
    if opts.dev != "npu2":
        sys.exit("matrix_multiplication_awq_4bit is NPU2-only")
    if opts.use_chess != 1:
        sys.exit("matrix_multiplication_awq_4bit supports Chess only")
    try:
        _awq_config_from_opts(opts)
    except (TypeError, ValueError) as error:
        sys.exit(str(error))
    if opts.warmup < 0:
        sys.exit("--warmup must be >= 0")
    if opts.iters < 1:
        sys.exit("--iters must be >= 1")
    if opts.benchmark_repeats < 1:
        sys.exit("--benchmark-repeats must be >= 1")
    if opts.min_gflops < 0:
        sys.exit("--min-gflops must be >= 0")
    if opts.verify_samples < 1:
        sys.exit("--verify-samples must be >= 1")


def _throughput_gflops(awq: AWQConfig, bench: BenchmarkResult) -> float:
    if bench.npu is None:
        raise RuntimeError("NPU timing was not returned by the runtime")
    return 2.0 * awq.M * awq.K * awq.N / (1000.0 * bench.npu.avg_us)


def _verify(
    opts,
    awq: AWQConfig,
    A: np.ndarray,
    qweight: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    actual: np.ndarray,
) -> None:
    mode = opts.verify_mode
    if mode == "none":
        print("Verification skipped (--verify-mode=none).")
        return
    if mode == "auto":
        mode = (
            "sampled"
            if awq.M * awq.K * awq.N > _VERIFY_SCALAR_PRODUCT_THRESHOLD
            else "full"
        )

    actual_f32 = actual.astype(np.float32)
    if mode == "full":
        expected = reference_matmul(
            A,
            qweight,
            scales,
            zeros,
            awq.group_size,
            dtype_out=awq.dtype_out,
            tile_k=awq.k,
        )
        np.testing.assert_allclose(
            actual_f32,
            expected.astype(np.float32),
            rtol=0.05,
            atol=0.5,
            err_msg="NPU output does not match the full AWQ CPU reference",
        )
        print(f"Full verification passed ({awq.M * awq.N} outputs).")
        return

    rng = np.random.default_rng(opts.seed ^ 0x4A17)
    rows = rng.integers(0, awq.M, size=opts.verify_samples)
    cols = rng.integers(0, awq.N, size=opts.verify_samples)
    indices = list(zip(rows.tolist(), cols.tolist()))
    expected = reference_samples(
        A,
        qweight,
        scales,
        zeros,
        awq.group_size,
        indices,
        dtype_out=awq.dtype_out,
        tile_k=awq.k,
    )
    expected_values = np.array([expected[index] for index in indices], dtype=np.float32)
    if awq.dtype_out == "bf16":
        expected_values = expected_values.astype(bfloat16).astype(np.float32)
    actual_values = np.array([actual_f32[index] for index in indices])
    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=0.05,
        atol=0.5,
        err_msg="NPU output does not match sampled AWQ CPU reference",
    )
    print(f"Sampled verification passed ({len(indices)} outputs).")


def _run_and_verify(opts) -> None:
    awq = _awq_config_from_opts(opts)
    A, qweight, scales, zeros = make_deterministic_inputs(awq, seed=opts.seed)
    packed_B = pack_awq_weights(qweight, scales, zeros, awq)
    dtype_out = _dtype_out(awq.dtype_out)

    A_tensor = iron.tensor(A.reshape(-1), dtype=bfloat16, device="npu")
    B_tensor = iron.tensor(packed_B, dtype=np.uint8, device="npu")
    C_tensor = iron.zeros(awq.M * awq.N, dtype=dtype_out, device="npu")

    round_gflops: list[float] = []
    for round_index in range(opts.benchmark_repeats):
        bench = run_iters(
            whole_array_awq_4bit,
            A_tensor,
            B_tensor,
            C_tensor,
            **_compile_kwargs(opts),
            warmup=opts.warmup,
            iters=opts.iters,
        )
        gflops = _throughput_gflops(awq, bench)
        round_gflops.append(gflops)
        assert bench.npu is not None
        print(
            f"Benchmark round {round_index + 1}/{opts.benchmark_repeats}: "
            f"{bench.npu.avg_us:.2f} us average, {gflops:.2f} GFLOP/s"
        )

    median_gflops = statistics.median(round_gflops)
    print(f"Median round-average throughput: {median_gflops:.2f} GFLOP/s")
    if (awq.M, awq.K, awq.N) == (4096, 4096, 4096):
        print(
            "Speedup over measured current BF16 baseline "
            f"({_REFERENCE_BF16_BASELINE_GFLOPS:.1f} GFLOP/s): "
            f"{median_gflops / _REFERENCE_BF16_BASELINE_GFLOPS:.3f}x"
        )

    actual = C_tensor.numpy().reshape(awq.M, awq.N)
    _verify(opts, awq, A, qweight, scales, zeros, actual)
    if opts.min_gflops and median_gflops < opts.min_gflops:
        raise RuntimeError(
            f"performance gate failed: {median_gflops:.2f} GFLOP/s < "
            f"{opts.min_gflops:.2f} GFLOP/s"
        )
    print("PASS!")


def main() -> None:
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array_awq_4bit,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda parsed: _device_for(parsed.dev, parsed.n_aie_cols),
        validate=_validate_cli,
    )


if __name__ == "__main__":
    main()
