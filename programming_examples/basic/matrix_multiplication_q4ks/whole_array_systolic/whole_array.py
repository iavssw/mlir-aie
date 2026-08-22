# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Eight-stage stationary-activation Q4_K systolic matrix multiplication."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.dialects.aie import EndOp, dma_bd, dma_start, memtile_dma, next_bd
from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence
from aie.iron import (
    Buffer,
    CascadeFlow,
    CompileTime,
    Flow,
    In,
    Kernel,
    Lock,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.dataflow import ObjectFifoLink
from aie.iron.device import Tile, from_name
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

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from packing import (  # noqa: E402
    Q4KSConfig,
    bfp16ebs8_to_float,
    decode_q4_k,
    float_to_bfp16ebs8,
    make_deterministic_native_q4_k,
)
from systolic_packing import (  # noqa: E402
    A_CHUNK_K,
    M_A,
    N_AIE_COLS,
    N_AIE_ROWS,
    TRANSPORTS,
    WEIGHT_FLOWS,
    SystolicConfig,
    prepare_systolic_bfp_activations,
    prepare_systolic_q4ks_weights,
)

KERNEL_SOURCE = str(HERE / "q4ks_systolic.cc")
AIE_KERNEL_INCLUDE = str(HERE.parents[3] / "aie_kernels")
VERIFY_SCALAR_PRODUCT_THRESHOLD = 1024 * 1024 * 1024


class _ColumnCacheDma:
    """Two-channel MemTile program with native MM2S replay.

    ``TileDma`` intentionally exposes the common one-transfer-per-BD form.
    This experiment needs the MemTile DMA's hardware ``repeat_count`` so a
    4096-row matrix does not consume 32 of channel 4's 24 available BD IDs.
    Keeping this tiny specialization local avoids changing the shared IRON
    interface merely for an experimental transport.
    """

    def __init__(self, tile, cache, cache_empty, cache_ready, length, repeats):
        self._tile = tile
        self._cache = cache
        self._cache_empty = cache_empty
        self._cache_ready = cache_ready
        self._length = length
        self._repeats = repeats
        self._resolved = False

    @property
    def tile(self):
        return self._tile

    def all_tiles(self):
        return [self._tile]

    def all_buffers_and_locks(self):
        return [self._cache], [self._cache_empty, self._cache_ready]

    def resolve(self, loc=None, ip=None):
        if self._resolved:
            return
        self._resolved = True

        @memtile_dma(self._tile.op)
        def _body(block):
            dma_start(
                DMAChannelDir.S2MM,
                5,
                dest=block[1],
                chain=block[2],
            )
            with block[1]:
                self._cache_empty.acquire(self._repeats)
                dma_bd(self._cache.op, transfer_len=self._length)
                self._cache_ready.release(self._repeats)
                next_bd(block[1])
            with block[2]:
                dma_start(
                    DMAChannelDir.MM2S,
                    4,
                    dest=block[3],
                    chain=block[4],
                    repeat_count=self._repeats - 1,
                )
            with block[3]:
                self._cache_ready.acquire(1)
                dma_bd(self._cache.op, transfer_len=self._length)
                self._cache_empty.release(1)
                next_bd(block[3])
            with block[4]:
                EndOp()


def _device_for(dev: str):
    if dev != "npu2":
        raise ValueError("whole_array_systolic is NPU2-only")
    device = from_name("npu2", n_cols=None)
    if resolve_target_arch(device) != "aie2p":
        raise ValueError("selected device is not AIE2P")
    return device


def _kernel_set(
    config: SystolicConfig,
    a_chunk_ty,
    a_bfp_ty,
    b_q4_ty,
    b_bfp_ty,
    c_ty,
    c_slab_ty,
):
    if config.weight_flow == "q4-expand-once":
        if config.transport == "q4-column-dequant-cache":
            storages = ("bfpstream",)
            helper_storage = "bfp"
        elif config.transport == "q4-expand-broadcast-persistent":
            storages = ("q4expandstream", "bfpstream")
            helper_storage = "helper"
        else:
            storages = ("q4expand", "bfp")
            helper_storage = "bfp"
    else:
        selected = {
            "q4-local": "q4",
            "q4-direct": (
                "q4stream"
                if config.transport
                in (
                    "memtile-fused-stream",
                    "memtile-fused-cache",
                    "memtile-fused-persistent",
                    "joint-fused",
                )
                else "q4direct"
            ),
            "bfp-prepared": (
                "bfpstream"
                if config.transport
                in (
                    "bfp-memtile-persistent",
                    "bfp-core-multicast-persistent",
                )
                else "bfp"
            ),
            "resident": "resident",
        }[config.weight_flow]
        storages = (selected,)
        helper_storage = selected

    storage_ids = {
        "q4": 0,
        "bfp": 1,
        "resident": 2,
        "q4expand": 3,
        "q4direct": 4,
        "q4stream": 5,
        "helper": 6,
        "bfpstream": 7,
        "q4expandstream": 8,
        "columnexpand": 9,
    }
    resident_repeats = config.n_panels * (config.M // config.m_wave)
    role_ids = {"all": 0, "west": 1, "middle": 2, "east": 3, "helpers": 4}
    direct_c_stream = config.transport in (
        "bfp-direct-c-preconverted-a",
        "q4-direct-c-preconverted-a",
    )

    def object_config(storage: str, role: str):
        # Only the synthetic resident ceiling loops over N_PANELS inside the
        # kernel.  Every eligible/native path consumes one panel per call, so
        # specializing those objects by the matrix-wide panel count is both
        # unnecessary and, because N was not part of the object name, caused
        # ExternalFunction registry collisions between otherwise compatible
        # designs.
        kernel_panels = (
            resident_repeats if storage == "resident" else 1
        )
        c_suffix = (
            f"_c{config.c_panel_slab}" if config.c_panel_slab > 1 else ""
        )
        direct_suffix = "_dc" if direct_c_stream else ""
        panel_suffix = (
            f"_p{kernel_panels}" if storage == "resident" else ""
        )
        object_name = (
            f"q4ks_systolic_v8_m{config.m_a}_a{config.a_chunk_k}_"
            f"k{config.k_stage}_n{config.n}{c_suffix}{direct_suffix}"
            f"{panel_suffix}_"
            f"{storage}_{role}.o"
        )
        flags = [
            f"-DDIM_M={config.m_a}",
            f"-DDIM_K_STAGE={config.k_stage}",
            f"-DDIM_N={config.n}",
            f"-DA_CHUNK_K={config.a_chunk_k}",
            f"-DQ4_TILE_BYTES={config.q4_tile_bytes}",
            f"-DN_PANELS={kernel_panels}",
            f"-DC_PANEL_SLAB={config.c_panel_slab}",
            f"-DKERNEL_STORAGE={storage_ids[storage]}",
            f"-DKERNEL_ROLE={role_ids[role]}",
            f"-DDIRECT_C_STREAM={1 if direct_c_stream else 0}",
            f"-I{AIE_KERNEL_INCLUDE}",
        ]
        return object_name, flags

    prebuilt_root = os.environ.get("Q4KS_SYSTOLIC_PREBUILT_DIR")

    def declare(symbol: str, object_name: str, flags, arg_types):
        if prebuilt_root:
            object_path = (Path(prebuilt_root).expanduser() / object_name).resolve()
            if not object_path.is_file():
                raise FileNotFoundError(
                    f"missing prebuilt systolic kernel object: {object_path}"
                )
            return Kernel(symbol, object_name, arg_types=arg_types)
        return ExternalFunction(
            symbol,
            object_file_name=object_name,
            source_file=KERNEL_SOURCE,
            arg_types=arg_types,
            include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
            compile_flags=flags,
            use_chess=True,
        )

    helper_object, helper_flags = object_config(helper_storage, "helpers")
    convert = ExternalFunction(
        "q4ks_systolic_convert_a",
        object_file_name=helper_object,
        source_file=KERNEL_SOURCE,
        arg_types=[a_chunk_ty, a_bfp_ty, np.int32],
        include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
        compile_flags=helper_flags,
        use_chess=True,
    )
    copy_a = Kernel(
        "q4ks_systolic_copy_a",
        helper_object,
        [a_chunk_ty, a_chunk_ty],
    )
    expand_q4 = None
    if config.transport == "q4-column-dequant-cache":
        expander_object, expander_flags = object_config(
            "columnexpand", "helpers"
        )
        expand_q4 = ExternalFunction(
            "q4ks_systolic_expand_q4_to_bfp_stream",
            object_file_name=expander_object,
            source_file=KERNEL_SOURCE,
            arg_types=[b_q4_ty, np.int32],
            include_dirs=[aie_config.cxx_header_path(), AIE_KERNEL_INCLUDE],
            compile_flags=expander_flags,
            use_chess=True,
        )

    roles = {}
    for storage in storages:
        b_ty = (
            b_q4_ty
            if storage in ("q4", "q4expand", "q4direct", "q4stream")
            else b_bfp_ty
        )
        for role in ("west", "middle", "east"):
            object_name, _flags = object_config(storage, role)
            forward_variants = (False,) if direct_c_stream else (False, True)
            for forward in forward_variants:
                args = [a_bfp_ty]
                if storage not in (
                    "q4stream",
                    "bfpstream",
                    "q4expandstream",
                ):
                    args.append(b_ty)
                if role == "east" and not (
                    direct_c_stream
                    and storage in ("bfp", "q4direct")
                    and not forward
                ):
                    if config.c_panel_slab > 1:
                        args.append(c_slab_ty)
                        if storage not in (
                            "q4stream",
                            "bfpstream",
                            "q4expandstream",
                        ):
                            args.append(np.int32)
                    else:
                        args.append(c_ty)
                suffix = "_forward" if forward else ""
                symbol = f"q4ks_systolic_{role}_{storage}{suffix}"
                if (
                    direct_c_stream
                    and storage in ("bfp", "q4direct")
                    and role == "east"
                    and not forward
                ):
                    symbol = (
                        "q4ks_systolic_east_bfp_stream_c"
                        if storage == "bfp"
                        else "q4ks_systolic_east_q4direct_stream_c"
                    )
                if role == "east" and config.c_panel_slab > 1:
                    symbol += "_slab"
                roles[(storage, role, forward)] = declare(
                    symbol,
                    object_name,
                    _flags,
                    args,
                )
    return convert, copy_a, expand_q4, roles


def _build_design(
    dev,
    M: int,
    K: int,
    N: int,
    n: int,
    m_a: int,
    weight_flow: str,
    transport: str,
    panel_slab: int,
    c_panel_slab: int,
    trace_config: TraceConfig | None,
    *,
    generate_taps: bool = False,
):
    if transport == "auto":
        transport = "core-stream"
    config = SystolicConfig(
        M=M,
        K=K,
        N=N,
        n=n,
        m_a=m_a,
        weight_flow=weight_flow,
        transport=transport,
        panel_slab=panel_slab,
        c_panel_slab=c_panel_slab,
    )
    nominal_waves = M // config.m_wave
    # The resident ceiling replays one stationary A/B pair entirely in-core.
    waves = 1 if weight_flow == "resident" else nominal_waves
    memtile_weight_cache = transport in (
        "memtile-cache",
        "memtile-fused-cache",
        "memtile-fused-persistent",
        "bfp-memtile-persistent",
        "bfp-memtile-broadcast-persistent",
        "bfp-core-multicast-persistent",
        "q4-memtile-broadcast-persistent",
        "q4-expand-broadcast-persistent",
        "q4-column-dequant-cache",
        "bfp-broadcast-preconverted-a",
        "bfp-direct-c-preconverted-a",
        "q4-direct-c-preconverted-a",
    )
    memtile_slab_stream = transport in (
        "memtile-slab",
        "memtile-slab-direct-a",
        "memtile-fused-stream",
    )
    fused_stream = transport in (
        "memtile-fused-stream",
        "memtile-fused-cache",
        "memtile-fused-persistent",
        "bfp-memtile-persistent",
        "bfp-core-multicast-persistent",
        "joint-fused",
        "q4-expand-broadcast-persistent",
    )
    persistent_cache = transport in (
        "memtile-fused-persistent",
        "bfp-memtile-persistent",
        "bfp-memtile-broadcast-persistent",
        "bfp-core-multicast-persistent",
        "q4-memtile-broadcast-persistent",
        "q4-expand-broadcast-persistent",
        "q4-column-dequant-cache",
        "bfp-broadcast-preconverted-a",
        "bfp-direct-c-preconverted-a",
        "q4-direct-c-preconverted-a",
    )
    expand_broadcast = transport == "q4-expand-broadcast-persistent"
    column_dequant_cache = transport == "q4-column-dequant-cache"
    core_multicast = transport in (
        "q4-expand-broadcast-persistent",
        "bfp-core-multicast-persistent",
    )
    preconverted_a = transport in (
        "bfp-broadcast-preconverted-a",
        "bfp-direct-c-preconverted-a",
        "q4-direct-c-preconverted-a",
    )
    direct_c_stream = transport in (
        "bfp-direct-c-preconverted-a",
        "q4-direct-c-preconverted-a",
    )
    memtile_broadcast_cache = transport in (
        "bfp-memtile-broadcast-persistent",
        "q4-memtile-broadcast-persistent",
        "bfp-broadcast-preconverted-a",
        "bfp-direct-c-preconverted-a",
        "q4-direct-c-preconverted-a",
    )
    direct_memtile_a = transport in (
        "memtile-slab-direct-a",
        "bfp-memtile-broadcast-persistent",
        "bfp-core-multicast-persistent",
        "q4-memtile-broadcast-persistent",
        "q4-column-dequant-cache",
        "bfp-broadcast-preconverted-a",
        "bfp-direct-c-preconverted-a",
        "q4-direct-c-preconverted-a",
    ) and not (column_dequant_cache and config.m_a > M_A)
    joint_slab = transport == "joint-slab"
    joint_fused = transport == "joint-fused"
    chunks = config.k_stage // config.a_chunk_k
    panels = config.n_panels
    # Keep every shim DMA BD at or below four prepared panels.  The complete
    # cached object is still one contiguous ObjectFIFO token; the leading TAP
    # dimension lowers to a repeated BD that fills successive 40-KiB regions
    # for the default K-stage instead of one unsupported 160/320-KiB BD.
    b_dma_panels = min(4, config.panel_slab)
    if column_dequant_cache:
        row_storages = ["bfpstream"] * N_AIE_ROWS
    elif expand_broadcast:
        row_storages = ["q4expandstream"] + ["bfpstream"] * (
            N_AIE_ROWS - 1
        )
    elif weight_flow == "resident":
        row_storages = ["resident"] * N_AIE_ROWS
    elif weight_flow == "bfp-prepared":
        bfp_storage = "bfpstream" if fused_stream else "bfp"
        row_storages = [bfp_storage] * N_AIE_ROWS
    elif weight_flow == "q4-direct":
        direct_storage = "q4stream" if fused_stream else "q4direct"
        row_storages = [direct_storage] * N_AIE_ROWS
    elif weight_flow == "q4-expand-once":
        row_storages = ["q4expand"] + ["bfp"] * (N_AIE_ROWS - 1)
    else:
        row_storages = ["q4"] * N_AIE_ROWS
    cache_slabs = (
        config.weight_cache_slabs
        if memtile_weight_cache or joint_slab or joint_fused
        else 1
    )
    worker_waves = waves * cache_slabs
    worker_panels = (
        config.panel_slab
        if memtile_weight_cache or joint_slab or joint_fused
        else (1 if weight_flow == "resident" else panels)
    )

    prepared_a_bytes = (
        N_AIE_COLS * waves * N_AIE_ROWS * config.a_panel_bfp_bytes
    )
    A_ty = (
        np.ndarray[(prepared_a_bytes,), np.dtype[np.uint8]]
        if preconverted_a
        else np.ndarray[(M * K,), np.dtype[bfloat16]]
    )
    B_ty = np.ndarray[(config.prepared_bytes,), np.dtype[np.uint8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    row_stream_a = config.m_a > M_A
    cache_c_tasks = (
        config.weight_replay_waves
        * config.panel_slab
        // config.c_panel_slab
    )
    fused_cache_group = (
        memtile_weight_cache
        and not row_stream_a
        and cache_c_tasks <= 16
        # Each shim carries one A and one B producer on its two MM2S channels.
        and 1 + config.weight_replay_waves <= 16
    )
    a_parent_rows = (
        2 * N_AIE_ROWS
        if joint_slab or joint_fused
        else (1 if row_stream_a else N_AIE_ROWS)
    )
    A_wave_ty = (
        np.ndarray[
            (N_AIE_ROWS * config.a_panel_bfp_bytes,), np.dtype[np.uint8]
        ]
        if preconverted_a
        else np.ndarray[
            (a_parent_rows * config.m_a * config.k_stage,), np.dtype[bfloat16]
        ]
    )
    A_row_ty = (
        np.ndarray[(config.a_panel_bfp_bytes,), np.dtype[np.uint8]]
        if preconverted_a
        else np.ndarray[
            (config.m_a * config.k_stage,), np.dtype[bfloat16]
        ]
    )
    A_chunk_ty = np.ndarray[(config.m_a * config.a_chunk_k,), np.dtype[bfloat16]]
    A_bfp_ty = np.ndarray[(config.a_panel_bfp_bytes,), np.dtype[np.uint8]]
    B_q4_ty = np.ndarray[(config.q4_tile_bytes,), np.dtype[np.uint8]]
    B_bfp_ty = np.ndarray[(config.bfp_tile_bytes,), np.dtype[np.uint8]]
    B_runtime_ty = (
        B_bfp_ty
        if weight_flow in ("bfp-prepared", "resident")
        else B_q4_ty
    )
    B_slab_ty = np.ndarray[
        (config.panel_slab * config.runtime_tile_bytes,), np.dtype[np.uint8]
    ]
    B_bfp_slab_ty = np.ndarray[
        (config.panel_slab * config.bfp_tile_bytes,), np.dtype[np.uint8]
    ]
    B_row_types = (
        [B_q4_ty] + [B_bfp_ty] * (N_AIE_ROWS - 1)
        if weight_flow == "q4-expand-once"
        else [B_runtime_ty] * N_AIE_ROWS
    )
    C_tile_ty = np.ndarray[(config.m_a * n,), np.dtype[bfloat16]]
    C_slab_ty = np.ndarray[
        (config.c_panel_slab * config.m_a * n,), np.dtype[bfloat16]
    ]

    convert, copy_a, expand_q4, role_kernels = _kernel_set(
        config,
        A_chunk_ty,
        A_bfp_ty,
        B_q4_ty,
        B_bfp_ty,
        C_tile_ty,
        C_slab_ty,
    )

    # One host/MemTile object per column is split into four row shards. Using
    # chunk-sized child objects lets dims_to_stream emit successive 32x64
    # objects with the supported split() ownership model.
    A_l3l2: list[ObjectFifo] = []
    A_l2l1: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    a_to_stream = [
        (a_parent_rows, config.m_a * config.k_stage),
        (config.k_stage // config.a_chunk_k, config.a_chunk_k),
        (config.m_a, config.k_stage),
        (config.a_chunk_k, 1),
    ]
    a_row_to_stream = [
        (config.k_stage // config.a_chunk_k, config.a_chunk_k),
        (config.m_a, config.k_stage),
        (config.a_chunk_k, 1),
    ]
    for col in range(N_AIE_COLS):
        parent = ObjectFifo(A_wave_ty, name=f"A_SYS_L3L2_{col}", depth=1)
        if preconverted_a:
            row_fifos = [
                ObjectFifo(
                    A_bfp_ty,
                    name=f"A_SYS_PRECONVERTED_ROW_{col}_{row}",
                    depth=1,
                )
                for row in range(N_AIE_ROWS)
            ]
            ObjectFifoLink(
                parent.cons(),
                [fifo.prod() for fifo in row_fifos],
                tile=Tile(col, 1),
                dst_offsets=[
                    row * config.a_panel_bfp_bytes
                    for row in range(N_AIE_ROWS)
                ],
            )
        elif direct_memtile_a:
            row_fifos = [
                ObjectFifo(
                    A_row_ty,
                    consumer_obj_type=A_chunk_ty,
                    name=f"A_SYS_DIRECT_ROW_{col}_{row}",
                    depth=1,
                    dims_to_stream=a_row_to_stream,
                )
                for row in range(N_AIE_ROWS)
            ]
            ObjectFifoLink(
                parent.cons(),
                [fifo.prod() for fifo in row_fifos],
                tile=Tile(col, 1),
                dst_offsets=[
                    row * config.m_a * config.k_stage
                    for row in range(N_AIE_ROWS)
                ],
            )
        else:
            entry = ObjectFifo(
                A_wave_ty,
                consumer_obj_type=A_chunk_ty,
                depth=1,
                name=f"A_SYS_ROW_{col}_0",
                dims_to_stream=a_to_stream,
                repeat_count=(
                    (
                        4
                        if joint_fused
                        else config.weight_cache_slabs
                        if joint_slab
                        else None
                    )
                ),
            )
            ObjectFifoLink(
                parent.cons(),
                entry.prod(),
                tile=Tile(col, 1),
            )
            row_fifos = [entry]
            for row in range(1, N_AIE_ROWS):
                row_fifos.append(
                    ObjectFifo(
                        A_chunk_ty,
                        name=f"A_SYS_ROW_{col}_{row}",
                        depth=1,
                    )
                )
        A_l3l2.append(parent)
        for row, fifo in enumerate(row_fifos):
            A_l2l1[row].append(fifo)

    # Most systolic transports inject B into row zero, which forwards it south
    # with put_ms().  Broadcast paths use one MemTile ObjectFIFO with four
    # consumers.  The eligible column-dequant path streams Q4 directly from
    # L3 into row zero, streams the once-expanded BFP slab back to a MemTile
    # Buffer, and replays that slab concurrently to all four rows.  Its replay
    # is intentionally expressed as an explicit TileDma rather than a dummy
    # ObjectFIFO: the latter is conservatively charged as a third core DMA
    # input by the tile placer even though the kernel reads a direct stream.
    B_l3l2: list[ObjectFifo] = []
    B_rows: list[list[ObjectFifo]] = [[] for _ in range(N_AIE_ROWS)]
    B_expand_inputs: list[ObjectFifo | None] = []
    B_expand_streams: list[ObjectFifo | None] = []
    B_cache_tiles: list[Tile] = []
    B_cache_dmas: list[_ColumnCacheDma] = []
    B_cache_locks: list[Lock] = []
    stream_transport = transport in (
        "core-stream",
        "shim-stream",
        "memtile-cache",
        "memtile-slab",
        "memtile-slab-direct-a",
        "joint-slab",
    )
    b_to_stream = [
        (config.panel_slab, config.runtime_tile_bytes),
        (config.runtime_tile_bytes // 64, 64),
        (64, 1),
    ]
    for col in range(N_AIE_COLS):
        if column_dequant_cache:
            parent = ObjectFifo(
                B_slab_ty,
                consumer_obj_type=B_q4_ty,
                depth=1,
                name=f"B_SYS_Q4_L3L1_{col}",
            )
            memtile = Tile(col, 1, tile_type=AIETileType.MemTile)
            cache = Buffer(
                B_bfp_slab_ty,
                tile=memtile,
                name=f"B_SYS_EXPANDED_CACHE_{col}",
            )
            cache_empty = Lock(
                memtile,
                init=config.weight_replay_waves,
                name=f"B_SYS_CACHE_EMPTY_{col}",
            )
            cache_ready = Lock(
                memtile, init=0, name=f"B_SYS_CACHE_READY_{col}"
            )

            # One S2MM BD receives a fully expanded slab.  The MM2S channel's
            # native repeat_count then replays the same descriptor once per
            # four-row M wave, using only one of channel 4's 24 BD IDs.
            B_cache_dmas.append(
                _ColumnCacheDma(
                    memtile,
                    cache,
                    cache_empty,
                    cache_ready,
                    config.panel_slab * config.bfp_tile_bytes,
                    config.weight_replay_waves,
                )
            )
            B_cache_locks.extend([cache_empty, cache_ready])
            B_cache_tiles.append(memtile)
            B_l3l2.append(parent)
            B_expand_inputs.append(parent)
            B_expand_streams.append(None)
            for row in range(N_AIE_ROWS):
                B_rows[row].append(None)
            continue

        B_expand_inputs.append(None)
        B_expand_streams.append(None)
        if transport == "shim-stream":
            parent = ObjectFifo(
                B_slab_ty,
                consumer_obj_type=B_row_types[0],
                depth=1,
                name=f"B_SYS_L3L1_{col}",
            )
            entry = parent
        else:
            parent = ObjectFifo(
                B_slab_ty,
                name=f"B_SYS_L3L2_{col}",
                depth=(
                    1
                    if memtile_weight_cache or joint_slab or joint_fused
                    else 2
                ),
            )

        if memtile_broadcast_cache:
            entry = ObjectFifo(
                B_slab_ty,
                consumer_obj_type=B_runtime_ty,
                depth=1,
                name=f"B_SYS_BROADCAST_{col}",
                repeat_count=config.weight_replay_waves,
            )
            ObjectFifoLink(parent.cons(), entry.prod(), tile=Tile(col, 1))
            row_fifos = [entry] * N_AIE_ROWS
        elif fused_stream:
            entry = ObjectFifo(
                B_slab_ty,
                consumer_obj_type=B_runtime_ty,
                depth=1,
                name=f"B_SYS_FUSED_ROW0_{col}",
                aie_stream=(1, 0),
                repeat_count=(
                    2
                    if joint_fused
                    else config.weight_replay_waves
                    if (
                        memtile_weight_cache
                        and config.weight_replay_waves > 1
                    )
                    else None
                ),
            )
            ObjectFifoLink(parent.cons(), entry.prod(), tile=Tile(col, 1))
            row_fifos = [entry, None, None, None]
        elif transport == "tile-dma":
            row_fifos = [
                ObjectFifo(
                    B_runtime_ty,
                    name=f"B_SYS_DMA_LOCAL_{col}_{row}",
                    depth=1,
                )
                for row in range(N_AIE_ROWS)
            ]
            ObjectFifoLink(
                parent.cons(),
                [fifo.prod() for fifo in row_fifos],
                tile=Tile(col, 1),
                dst_offsets=[0] * N_AIE_ROWS,
            )
        elif transport in (
            "core-stream",
            "memtile-cache",
            "memtile-slab",
            "memtile-slab-direct-a",
            "joint-slab",
        ):
            entry = ObjectFifo(
                B_slab_ty,
                consumer_obj_type=B_row_types[0],
                depth=1,
                name=f"B_SYS_ROW_{col}_0",
                # The cached slab is already contiguous in tile order.  Keep
                # the replay BD one-dimensional, matching the proven parent
                # memtile-weight design, rather than combining repeat_count
                # with a redundant identity dimensionsToStream descriptor.
                dims_to_stream=(
                    None
                    if memtile_weight_cache or memtile_slab_stream or joint_slab
                    else b_to_stream
                ),
                repeat_count=(
                    (
                        2
                        if joint_slab
                        else config.weight_replay_waves
                    )
                    if (
                        joint_slab
                        or (
                            memtile_weight_cache
                            and config.weight_replay_waves > 1
                        )
                    )
                    else None
                ),
            )
            ObjectFifoLink(parent.cons(), entry.prod(), tile=Tile(col, 1))

        if stream_transport:
            row_fifos = [entry]
            for row in range(1, N_AIE_ROWS):
                row_fifos.append(
                    ObjectFifo(
                        B_row_types[row],
                        name=f"B_SYS_ROW_{col}_{row}",
                        depth=1,
                        aie_stream=(0, 0),
                    )
                )
        B_l3l2.append(parent)
        for row, fifo in enumerate(row_fifos):
            B_rows[row].append(fifo)

    # Only the eastern edge materializes C.  A grouped path writes several
    # panels into one token-major slab, then double-buffers that slab in the
    # MemTile before one large L3 drain.  The single-panel path is retained as
    # an exact baseline.
    if transport == "q4-direct-c-preconverted-a":
        c_to_stream = [
            (config.m_a // 32, 32 * n),
            (32, 8),
            (n // 8, 256),
            (8, 1),
        ]
        C_l1l3 = []
        C_l2l3 = []
        for row in range(N_AIE_ROWS):
            child = ObjectFifo(
                C_tile_ty,
                name=f"C_SYS_STREAM_{row}",
                depth=2,
                aie_stream=(0, 0),
            )
            parent = ObjectFifo(
                C_tile_ty,
                name=f"C_SYS_STREAM_L2L3_{row}",
                depth=2,
                dims_to_stream=c_to_stream,
            )
            ObjectFifoLink(child.cons(), parent.prod(), tile=Tile(row, 1))
            C_l1l3.append(child)
            C_l2l3.append(parent)
    elif direct_c_stream:
        C_l1l3 = [
            ObjectFifo(
                C_tile_ty,
                name=f"C_SYS_STREAM_{row}",
                depth=2,
                aie_stream=(0, 0),
            )
            for row in range(N_AIE_ROWS)
        ]
        C_l2l3 = C_l1l3
    elif config.c_panel_slab == 1 and not persistent_cache:
        c_depth = 1 if row_stream_a else 2
        C_l1l3 = [
            ObjectFifo(C_tile_ty, name=f"C_SYS_{row}", depth=c_depth)
            for row in range(N_AIE_ROWS)
        ]
        C_l2l3 = C_l1l3
    else:
        C_l1l3 = []
        C_l2l3 = []
        c_slab_cols = config.c_panel_slab * n
        c_slab_to_stream = [
            (config.m_a // 16, 16 * c_slab_cols),
            (16, 16),
            (c_slab_cols // 16, 256),
            (16, 1),
        ]
        for row in range(N_AIE_ROWS):
            parent = ObjectFifo(
                C_slab_ty,
                name=f"C_SYS_L2L3_{row}",
                depth=2,
                dims_to_stream=(
                    c_slab_to_stream
                    if (
                        memtile_slab_stream
                        or joint_slab
                        or joint_fused
                        or persistent_cache
                    )
                    else None
                ),
            )
            child = ObjectFifo(
                C_slab_ty,
                name=f"C_SYS_L1L2_{row}",
                depth=1,
            )
            ObjectFifoLink(child.cons(), parent.prod(), tile=Tile(row, 1))
            C_l1l3.append(child)
            C_l2l3.append(parent)

    def _convert_stationary_a(in_a, a_bfp, convert_fn):
        chunk_loop = range_(chunks) if chunks > 1 else range(1)
        for chunk in chunk_loop:
            elem_a = in_a.acquire(1)
            convert_fn(elem_a, a_bfp, chunk)
            in_a.release(1)

    def _forward_later_a(in_a, out_a, copy_fn, count):
        forward_loop = range_(count) if count > 1 else range(1)
        for _chunk in forward_loop:
            elem_a = in_a.acquire(1)
            elem_out = out_a.acquire(1)
            copy_fn(elem_a, elem_out)
            in_a.release(1)
            out_a.release(1)

    def _compute_east_panels(in_b, out_c, a_bfp, compute_fn):
        if config.c_panel_slab == 1:
            panel_loop = (
                range_(worker_panels) if worker_panels > 1 else range(1)
            )
            for _panel in panel_loop:
                elem_b = in_b.acquire(1)
                elem_c = out_c.acquire(1)
                compute_fn(a_bfp, elem_b, elem_c)
                in_b.release(1)
                out_c.release(1)
            return

        slab_count = worker_panels // config.c_panel_slab
        slab_loop = range_(slab_count) if slab_count > 1 else range(1)
        for _slab in slab_loop:
            elem_c = out_c.acquire(1)
            for c_panel in range(config.c_panel_slab):
                elem_b = in_b.acquire(1)
                compute_fn(a_bfp, elem_b, elem_c, c_panel)
                in_b.release(1)
            out_c.release(1)

    def make_non_east_forward_fn(physical_row):
        forward_chunks = (N_AIE_ROWS - physical_row - 1) * chunks

        def body(
            in_a,
            out_a,
            in_b,
            a_bfp,
            convert_fn,
            copy_fn,
            compute_fn,
            b_forward,
        ):
            wave_loop = (
                range_(worker_waves) if worker_waves > 1 else range(1)
            )
            for _wave in wave_loop:
                _convert_stationary_a(in_a, a_bfp, convert_fn)
                _forward_later_a(in_a, out_a, copy_fn, forward_chunks)
                panel_loop = range_(worker_panels) if worker_panels > 1 else range(1)
                for _panel in panel_loop:
                    elem_b = in_b.acquire(1)
                    compute_fn(a_bfp, elem_b)
                    in_b.release(1)

        return body

    def make_non_east_dma_fn(physical_row):
        forward_chunks = (N_AIE_ROWS - physical_row - 1) * chunks

        def body(
            in_a,
            out_a,
            in_b,
            a_bfp,
            convert_fn,
            copy_fn,
            compute_fn,
        ):
            wave_loop = (
                range_(worker_waves) if worker_waves > 1 else range(1)
            )
            for _wave in wave_loop:
                _convert_stationary_a(in_a, a_bfp, convert_fn)
                _forward_later_a(in_a, out_a, copy_fn, forward_chunks)
                panel_loop = (
                    range_(worker_panels) if worker_panels > 1 else range(1)
                )
                for _panel in panel_loop:
                    elem_b = in_b.acquire(1)
                    compute_fn(a_bfp, elem_b)
                    in_b.release(1)

        return body

    def non_east_direct_a_forward_fn(
        in_a,
        in_b,
        a_bfp,
        convert_fn,
        compute_fn,
        _b_forward,
    ):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            panel_loop = (
                range_(worker_panels) if worker_panels > 1 else range(1)
            )
            for _panel in panel_loop:
                elem_b = in_b.acquire(1)
                compute_fn(a_bfp, elem_b)
                in_b.release(1)

    def east_direct_a_forward_fn(
        in_a,
        in_b,
        out_c,
        a_bfp,
        convert_fn,
        compute_fn,
        _b_forward,
    ):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            _compute_east_panels(in_b, out_c, a_bfp, compute_fn)

    def last_row_non_east_fn(in_a, in_b, a_bfp, convert_fn, compute_fn):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            panel_loop = range_(worker_panels) if worker_panels > 1 else range(1)
            for _panel in panel_loop:
                elem_b = in_b.acquire(1)
                compute_fn(a_bfp, elem_b)
                in_b.release(1)

    def make_east_forward_fn(physical_row):
        forward_chunks = (N_AIE_ROWS - physical_row - 1) * chunks

        def body(
            in_a,
            out_a,
            in_b,
            out_c,
            a_bfp,
            convert_fn,
            copy_fn,
            compute_fn,
            b_forward,
        ):
            wave_loop = (
                range_(worker_waves) if worker_waves > 1 else range(1)
            )
            for _wave in wave_loop:
                _convert_stationary_a(in_a, a_bfp, convert_fn)
                _forward_later_a(in_a, out_a, copy_fn, forward_chunks)
                _compute_east_panels(in_b, out_c, a_bfp, compute_fn)

        return body

    def make_east_dma_fn(physical_row):
        forward_chunks = (N_AIE_ROWS - physical_row - 1) * chunks

        def body(
            in_a,
            out_a,
            in_b,
            out_c,
            a_bfp,
            convert_fn,
            copy_fn,
            compute_fn,
        ):
            wave_loop = (
                range_(worker_waves) if worker_waves > 1 else range(1)
            )
            for _wave in wave_loop:
                _convert_stationary_a(in_a, a_bfp, convert_fn)
                _forward_later_a(in_a, out_a, copy_fn, forward_chunks)
                _compute_east_panels(in_b, out_c, a_bfp, compute_fn)

        return body

    def east_last_row_fn(in_a, in_b, out_c, a_bfp, convert_fn, compute_fn):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            _compute_east_panels(in_b, out_c, a_bfp, compute_fn)

    def _compute_fused_non_east(a_bfp, compute_fn):
        slab_count = worker_panels // config.c_panel_slab
        slab_loop = range_(slab_count) if slab_count > 1 else range(1)
        for _slab in slab_loop:
            compute_fn(a_bfp)

    def _compute_fused_east(out_c, a_bfp, compute_fn):
        slab_count = worker_panels // config.c_panel_slab
        slab_loop = range_(slab_count) if slab_count > 1 else range(1)
        for _slab in slab_loop:
            elem_c = out_c.acquire(1)
            compute_fn(a_bfp, elem_c)
            out_c.release(1)

    def fused_direct_non_east_input_fn(
        in_a, _stream_in, a_bfp, convert_fn, compute_fn
    ):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            _compute_fused_non_east(a_bfp, compute_fn)

    def fused_direct_non_east_fn(in_a, a_bfp, convert_fn, compute_fn):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            _compute_fused_non_east(a_bfp, compute_fn)

    def fused_direct_east_input_fn(
        in_a, _stream_in, out_c, a_bfp, convert_fn, compute_fn
    ):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            _compute_fused_east(out_c, a_bfp, compute_fn)

    def fused_direct_east_fn(in_a, out_c, a_bfp, convert_fn, compute_fn):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            _compute_fused_east(out_c, a_bfp, compute_fn)

    def preconverted_non_east_fn(in_a, in_b, compute_fn):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            elem_a = in_a.acquire(1)
            panel_loop = range_(worker_panels) if worker_panels > 1 else range(1)
            for _panel in panel_loop:
                elem_b = in_b.acquire(1)
                compute_fn(elem_a, elem_b)
                in_b.release(1)
            in_a.release(1)

    def preconverted_east_fn(in_a, in_b, out_c, compute_fn):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            elem_a = in_a.acquire(1)
            _compute_east_panels(in_b, out_c, elem_a, compute_fn)
            in_a.release(1)

    def preconverted_east_stream_fn(in_a, in_b, _out_c, compute_fn):
        wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
        for _wave in wave_loop:
            elem_a = in_a.acquire(1)
            panel_loop = range_(worker_panels) if worker_panels > 1 else range(1)
            for _panel in panel_loop:
                elem_b = in_b.acquire(1)
                compute_fn(elem_a, elem_b)
                in_b.release(1)
            in_a.release(1)

    def _expand_column_slab(in_q4, expand_fn):
        # The direct stream represents one complete BFP slab.  Only the last
        # tile asserts TLAST, allowing the MemTile consumer to lock/replay the
        # slab as one object after all panels have been produced.
        for panel in range(config.panel_slab):
            elem_q4 = in_q4.acquire(1)
            expand_fn(elem_q4, 1 if panel + 1 == config.panel_slab else 0)
            in_q4.release(1)

    def _compute_column_cache_non_east(
        in_a, a_bfp, convert_fn, compute_fn
    ):
        wave_loop = range_(waves) if waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            slab_count = worker_panels // config.c_panel_slab
            slab_loop = range_(slab_count) if slab_count > 1 else range(1)
            for _c_slab in slab_loop:
                compute_fn(a_bfp)

    def _compute_column_cache_east(
        in_a, out_c, a_bfp, convert_fn, compute_fn
    ):
        wave_loop = range_(waves) if waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            slab_count = worker_panels // config.c_panel_slab
            slab_loop = range_(slab_count) if slab_count > 1 else range(1)
            for _c_slab in slab_loop:
                elem_c = out_c.acquire(1)
                compute_fn(a_bfp, elem_c)
                out_c.release(1)

    def column_cache_top_non_east_fn(
        in_a,
        in_q4,
        a_bfp,
        convert_fn,
        expand_fn,
        compute_fn,
    ):
        slab_loop = range_(cache_slabs) if cache_slabs > 1 else range(1)
        for _slab in slab_loop:
            _expand_column_slab(in_q4, expand_fn)
            _compute_column_cache_non_east(
                in_a, a_bfp, convert_fn, compute_fn
            )

    def column_cache_top_east_fn(
        in_a,
        in_q4,
        out_c,
        a_bfp,
        convert_fn,
        expand_fn,
        compute_fn,
    ):
        slab_loop = range_(cache_slabs) if cache_slabs > 1 else range(1)
        for _slab in slab_loop:
            _expand_column_slab(in_q4, expand_fn)
            _compute_column_cache_east(
                in_a, out_c, a_bfp, convert_fn, compute_fn
            )

    def column_cache_non_east_fn(in_a, a_bfp, convert_fn, compute_fn):
        slab_loop = range_(cache_slabs) if cache_slabs > 1 else range(1)
        for _slab in slab_loop:
            _compute_column_cache_non_east(
                in_a, a_bfp, convert_fn, compute_fn
            )

    def column_cache_east_fn(
        in_a, out_c, a_bfp, convert_fn, compute_fn
    ):
        slab_loop = range_(cache_slabs) if cache_slabs > 1 else range(1)
        for _slab in slab_loop:
            _compute_column_cache_east(
                in_a, out_c, a_bfp, convert_fn, compute_fn
            )

    def _compute_column_cache_forward_non_east(
        physical_row,
        in_a,
        out_a,
        a_bfp,
        convert_fn,
        copy_fn,
        compute_fn,
    ):
        forward_chunks = (N_AIE_ROWS - physical_row - 1) * chunks
        wave_loop = range_(waves) if waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            _forward_later_a(in_a, out_a, copy_fn, forward_chunks)
            slab_count = worker_panels // config.c_panel_slab
            slab_loop = range_(slab_count) if slab_count > 1 else range(1)
            for _c_slab in slab_loop:
                compute_fn(a_bfp)

    def _compute_column_cache_forward_east(
        physical_row,
        in_a,
        out_a,
        out_c,
        a_bfp,
        convert_fn,
        copy_fn,
        compute_fn,
    ):
        forward_chunks = (N_AIE_ROWS - physical_row - 1) * chunks
        wave_loop = range_(waves) if waves > 1 else range(1)
        for _wave in wave_loop:
            _convert_stationary_a(in_a, a_bfp, convert_fn)
            _forward_later_a(in_a, out_a, copy_fn, forward_chunks)
            slab_count = worker_panels // config.c_panel_slab
            slab_loop = range_(slab_count) if slab_count > 1 else range(1)
            for _c_slab in slab_loop:
                elem_c = out_c.acquire(1)
                compute_fn(a_bfp, elem_c)
                out_c.release(1)

    def make_column_cache_forward_fn(physical_row, top, east):
        if top and east:
            def body(
                in_a,
                out_a,
                in_q4,
                out_c,
                a_bfp,
                convert_fn,
                copy_fn,
                expand_fn,
                compute_fn,
            ):
                slab_loop = range_(cache_slabs) if cache_slabs > 1 else range(1)
                for _slab in slab_loop:
                    _expand_column_slab(in_q4, expand_fn)
                    _compute_column_cache_forward_east(
                        physical_row,
                        in_a,
                        out_a,
                        out_c,
                        a_bfp,
                        convert_fn,
                        copy_fn,
                        compute_fn,
                    )
            return body

        if top:
            def body(
                in_a,
                out_a,
                in_q4,
                a_bfp,
                convert_fn,
                copy_fn,
                expand_fn,
                compute_fn,
            ):
                slab_loop = range_(cache_slabs) if cache_slabs > 1 else range(1)
                for _slab in slab_loop:
                    _expand_column_slab(in_q4, expand_fn)
                    _compute_column_cache_forward_non_east(
                        physical_row,
                        in_a,
                        out_a,
                        a_bfp,
                        convert_fn,
                        copy_fn,
                        compute_fn,
                    )
            return body

        if east:
            def body(
                in_a,
                out_a,
                out_c,
                a_bfp,
                convert_fn,
                copy_fn,
                compute_fn,
            ):
                slab_loop = range_(cache_slabs) if cache_slabs > 1 else range(1)
                for _slab in slab_loop:
                    _compute_column_cache_forward_east(
                        physical_row,
                        in_a,
                        out_a,
                        out_c,
                        a_bfp,
                        convert_fn,
                        copy_fn,
                        compute_fn,
                    )
            return body

        def body(
            in_a,
            out_a,
            a_bfp,
            convert_fn,
            copy_fn,
            compute_fn,
        ):
            slab_loop = range_(cache_slabs) if cache_slabs > 1 else range(1)
            for _slab in slab_loop:
                _compute_column_cache_forward_non_east(
                    physical_row,
                    in_a,
                    out_a,
                    a_bfp,
                    convert_fn,
                    copy_fn,
                    compute_fn,
                )
        return body

    def make_fused_non_east_fn(physical_row, input_route_marker):
        forward_chunks = (N_AIE_ROWS - physical_row - 1) * chunks
        if physical_row + 1 < N_AIE_ROWS:
            if input_route_marker:
                def body(
                    in_a,
                    out_a,
                    _stream_in,
                    a_bfp,
                    convert_fn,
                    copy_fn,
                    compute_fn,
                ):
                    wave_loop = (
                        range_(worker_waves) if worker_waves > 1 else range(1)
                    )
                    for _wave in wave_loop:
                        _convert_stationary_a(in_a, a_bfp, convert_fn)
                        _forward_later_a(
                            in_a, out_a, copy_fn, forward_chunks
                        )
                        _compute_fused_non_east(a_bfp, compute_fn)
                return body

            def body(
                in_a, out_a, a_bfp, convert_fn, copy_fn, compute_fn
            ):
                wave_loop = (
                    range_(worker_waves) if worker_waves > 1 else range(1)
                )
                for _wave in wave_loop:
                    _convert_stationary_a(in_a, a_bfp, convert_fn)
                    _forward_later_a(in_a, out_a, copy_fn, forward_chunks)
                    _compute_fused_non_east(a_bfp, compute_fn)
            return body

        def body(in_a, a_bfp, convert_fn, compute_fn):
            wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
            for _wave in wave_loop:
                _convert_stationary_a(in_a, a_bfp, convert_fn)
                _compute_fused_non_east(a_bfp, compute_fn)
        return body

    def make_fused_east_fn(physical_row, input_route_marker):
        forward_chunks = (N_AIE_ROWS - physical_row - 1) * chunks
        if physical_row + 1 < N_AIE_ROWS:
            if input_route_marker:
                def body(
                    in_a,
                    out_a,
                    _stream_in,
                    out_c,
                    a_bfp,
                    convert_fn,
                    copy_fn,
                    compute_fn,
                ):
                    wave_loop = (
                        range_(worker_waves) if worker_waves > 1 else range(1)
                    )
                    for _wave in wave_loop:
                        _convert_stationary_a(in_a, a_bfp, convert_fn)
                        _forward_later_a(
                            in_a, out_a, copy_fn, forward_chunks
                        )
                        _compute_fused_east(out_c, a_bfp, compute_fn)
                return body

            def body(
                in_a,
                out_a,
                out_c,
                a_bfp,
                convert_fn,
                copy_fn,
                compute_fn,
            ):
                wave_loop = (
                    range_(worker_waves) if worker_waves > 1 else range(1)
                )
                for _wave in wave_loop:
                    _convert_stationary_a(in_a, a_bfp, convert_fn)
                    _forward_later_a(in_a, out_a, copy_fn, forward_chunks)
                    _compute_fused_east(out_c, a_bfp, compute_fn)
            return body

        def body(in_a, out_c, a_bfp, convert_fn, compute_fn):
            wave_loop = range_(worker_waves) if worker_waves > 1 else range(1)
            for _wave in wave_loop:
                _convert_stationary_a(in_a, a_bfp, convert_fn)
                _compute_fused_east(out_c, a_bfp, compute_fn)
        return body

    workers: list[list[Worker]] = [[] for _ in range(N_AIE_ROWS)]
    for row in range(N_AIE_ROWS):
        for col in range(N_AIE_COLS):
            role = "west" if col == 0 else "east" if col == 7 else "middle"
            forwards = (
                row == 0
                if core_multicast
                else (stream_transport or fused_stream)
                and row + 1 < N_AIE_ROWS
            )
            compute_storage = row_storages[row]
            compute = role_kernels[(compute_storage, role, forwards)]
            if preconverted_a:
                args = [
                    A_l2l1[row][col].cons(channel=1),
                    B_rows[row][col].cons(),
                ]
                if role == "east":
                    args.append(C_l1l3[row].prod())
                args.append(compute)
                workers[row].append(
                    Worker(
                        (
                            preconverted_east_stream_fn
                            if direct_c_stream and role == "east"
                            else preconverted_east_fn
                            if role == "east"
                            else preconverted_non_east_fn
                        ),
                        args,
                        tile=Tile(col, row + 2),
                        stack_size=0x800 if row_stream_a else 0xD00,
                        trace=(
                            1
                            if trace_config and row == 0 and col == 4
                            else 0
                        ),
                    )
                )
                continue
            stationary_a = Buffer(A_bfp_ty, name=f"A_SYS_BFP_{col}_{row}")
            if column_dequant_cache:
                args = [
                    A_l2l1[row][col].cons(channel=1),
                ]
                if row_stream_a and row + 1 < N_AIE_ROWS:
                    args.append(A_l2l1[row + 1][col].prod(channel=1))
                if row == 0:
                    args.append(B_expand_inputs[col].cons())
                if role == "east":
                    args.append(C_l1l3[row].prod())
                args.extend([stationary_a, convert])
                if row_stream_a and row + 1 < N_AIE_ROWS:
                    args.append(copy_a)
                if row == 0:
                    args.append(expand_q4)
                args.append(compute)
                if row_stream_a and row + 1 < N_AIE_ROWS:
                    body = make_column_cache_forward_fn(
                        row, row == 0, role == "east"
                    )
                else:
                    body = (
                        column_cache_top_east_fn
                        if row == 0 and role == "east"
                        else column_cache_top_non_east_fn
                        if row == 0
                        else column_cache_east_fn
                        if role == "east"
                        else column_cache_non_east_fn
                    )
                workers[row].append(
                    Worker(
                        body,
                        args,
                        tile=Tile(col, row + 2),
                        stack_size=0x800,
                        trace=(
                            1
                            if trace_config and row == 0 and col == 4
                            else 0
                        ),
                    )
                )
                continue
            if fused_stream:
                has_input_marker = row == 0
                if direct_memtile_a:
                    args = [A_l2l1[row][col].cons(channel=1)]
                    if has_input_marker:
                        args.append(B_rows[row][col].cons())
                    if role == "east":
                        args.append(C_l1l3[row].prod())
                    args.extend([stationary_a, convert, compute])
                    if role == "east":
                        body = (
                            fused_direct_east_input_fn
                            if has_input_marker
                            else fused_direct_east_fn
                        )
                    else:
                        body = (
                            fused_direct_non_east_input_fn
                            if has_input_marker
                            else fused_direct_non_east_fn
                        )
                    workers[row].append(
                        Worker(
                            body,
                            args,
                            tile=Tile(col, row + 2),
                            stack_size=0x800,
                            trace=(
                                1
                                if trace_config and row == 0 and col == 4
                                else 0
                            ),
                        )
                    )
                    continue
                args = [A_l2l1[row][col].cons(channel=1)]
                if row + 1 < N_AIE_ROWS:
                    args.append(A_l2l1[row + 1][col].prod(channel=1))
                if has_input_marker:
                    args.append(B_rows[row][col].cons())
                if role == "east":
                    args.append(C_l1l3[row].prod())
                args.append(stationary_a)
                args.append(convert)
                if row + 1 < N_AIE_ROWS:
                    args.append(copy_a)
                args.append(compute)
                body = (
                    make_fused_east_fn(row, has_input_marker)
                    if role == "east"
                    else make_fused_non_east_fn(row, has_input_marker)
                )
                workers[row].append(
                    Worker(
                        body,
                        args,
                        tile=Tile(col, row + 2),
                        stack_size=0xD00,
                        trace=(
                            1
                            if trace_config and row == 0 and col == 4
                            else 0
                        ),
                    )
                )
                continue
            common = [
                A_l2l1[row][col].cons(channel=1),
            ]
            if not direct_memtile_a and row + 1 < N_AIE_ROWS:
                common.append(A_l2l1[row + 1][col].prod(channel=1))
            common.append(B_rows[row][col].cons())
            if direct_memtile_a and role == "east":
                args = common + [
                    C_l1l3[row].prod(),
                    stationary_a,
                    convert,
                    compute,
                ]
                if forwards:
                    args.append(B_rows[row + 1][col].prod())
                    body = east_direct_a_forward_fn
                else:
                    body = east_last_row_fn
            elif direct_memtile_a:
                args = common + [stationary_a, convert, compute]
                if forwards:
                    args.append(B_rows[row + 1][col].prod())
                    body = non_east_direct_a_forward_fn
                else:
                    body = last_row_non_east_fn
            elif role == "east" and row + 1 < N_AIE_ROWS:
                args = common + [
                    C_l1l3[row].prod(),
                    stationary_a,
                    convert,
                    copy_a,
                    compute,
                ]
                if forwards:
                    args.append(B_rows[row + 1][col].prod())
                    body = make_east_forward_fn(row)
                else:
                    body = make_east_dma_fn(row)
            elif role == "east":
                args = common + [C_l1l3[row].prod(), stationary_a, convert, compute]
                body = east_last_row_fn
            elif row + 1 < N_AIE_ROWS:
                args = common + [
                    stationary_a,
                    convert,
                    copy_a,
                    compute,
                ]
                if forwards:
                    args.append(B_rows[row + 1][col].prod())
                    body = make_non_east_forward_fn(row)
                else:
                    body = make_non_east_dma_fn(row)
            else:
                args = common + [stationary_a, convert, compute]
                body = last_row_non_east_fn
            workers[row].append(
                Worker(
                    body,
                    args,
                    tile=Tile(col, row + 2),
                    stack_size=0x800 if row_stream_a else 0xD00,
                    trace=1 if trace_config and row == 0 and col == 4 else 0,
                )
            )

    for row in range(N_AIE_ROWS):
        for col in range(N_AIE_COLS - 1):
            CascadeFlow(workers[row][col], workers[row][col + 1])

    flat_workers = [worker for row in workers for worker in row]
    # Each shim has two host-to-device channels, used by one A and one B
    # producer.  C drains on columns 0..3 use the independent device-to-host
    # direction, so they do not consume either MM2S channel.
    input_shim = lambda col: col
    A_prods = [
        fifo.prod(tile=Tile(input_shim(col), 0))
        for col, fifo in enumerate(A_l3l2)
    ]
    B_prods = [
        fifo.prod(tile=Tile(input_shim(col), 0))
        for col, fifo in enumerate(B_l3l2)
    ]
    C_conses = [fifo.cons(tile=Tile(row, 0)) for row, fifo in enumerate(C_l2l3)]

    A_taps: list[TensorAccessPattern] = []
    B_taps: list[TensorAccessPattern] = []
    C_taps: list[TensorAccessPattern] = []

    def sequence(A, B, C, A_hs, B_hs, C_hs):
        if persistent_cache:
            c_objects = config.panel_slab // config.c_panel_slab

            # Keep one compressed Q4 slab resident for the complete M sweep.
            # A and C each use one repeated task, so the group consumes only
            # one A, one B, and one C descriptor per participating shim.
            for cache_base in range(0, panels, config.panel_slab):
                transfer_block = TaskGroup()
                for col in range(N_AIE_COLS):
                    tile = col * panels + cache_base
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=tile * config.runtime_tile_bytes,
                        sizes=[
                            config.panel_slab // b_dma_panels,
                            b_dma_panels,
                            config.runtime_tile_bytes // 64,
                            64,
                        ],
                        strides=[
                            b_dma_panels * config.runtime_tile_bytes,
                            config.runtime_tile_bytes,
                            64,
                            1,
                        ],
                    )
                    B_hs[col].fill(B, tap=b_tap, group=transfer_block)
                    B_taps.append(b_tap)

                    if preconverted_a:
                        a_wave_bytes = N_AIE_ROWS * config.a_panel_bfp_bytes
                        a_tap = TensorAccessPattern(
                            (prepared_a_bytes,),
                            offset=col * waves * a_wave_bytes,
                            sizes=[1, waves, a_wave_bytes // 64, 64],
                            strides=[0, a_wave_bytes, 64, 1],
                        )
                    else:
                        a_tap = TensorAccessPattern(
                            (M, K),
                            offset=col * config.k_stage,
                            sizes=[
                                waves,
                                config.m_wave,
                                config.k_stage // 2,
                                2,
                            ],
                            strides=[
                                config.m_wave * K,
                                K,
                                2,
                                1,
                            ],
                        )
                    A_hs[col].fill(A, tap=a_tap, group=transfer_block)
                    A_taps.append(a_tap)

                for row in range(N_AIE_ROWS):
                    c_tap = TensorAccessPattern(
                        (M, N),
                        offset=row * config.m_a * N + cache_base * n,
                        sizes=[
                            waves,
                            c_objects,
                            config.m_a,
                            config.c_panel_slab * n,
                        ],
                        strides=[
                            config.m_wave * N,
                            config.c_panel_slab * n,
                            N,
                            1,
                        ],
                    )
                    C_hs[row].drain(
                        C,
                        tap=c_tap,
                        wait=True,
                        group=transfer_block,
                    )
                    C_taps.append(c_tap)
                transfer_block.finish()
            return

        if joint_fused:
            group_waves = 2
            slabs_per_group = 4
            c_objects_per_slab = config.panel_slab // config.c_panel_slab

            # Cache two activation waves and replay them across four resident
            # Q4 slabs.  Every B slab is injected once then direct-streamed to
            # both waves, so this retains the streaming path's 32 group
            # boundaries while halving L3 weight traffic at 4096^3.
            for wave_base in range(0, waves, group_waves):
                for slab_group in range(
                    0, config.weight_cache_slabs, slabs_per_group
                ):
                    transfer_block = TaskGroup()
                    panel_base = slab_group * config.panel_slab

                    for col in range(N_AIE_COLS):
                        a_tap = TensorAccessPattern(
                            (M, K),
                            offset=wave_base * config.m_wave * K
                            + col * config.k_stage,
                            sizes=[
                                group_waves * N_AIE_ROWS,
                                config.m_a,
                                config.k_stage // 2,
                                2,
                            ],
                            strides=[config.m_a * K, K, 2, 1],
                        )
                        A_hs[col].fill(A, tap=a_tap, group=transfer_block)
                        A_taps.append(a_tap)

                    for col in range(N_AIE_COLS):
                        tile = col * panels + panel_base
                        b_tap = TensorAccessPattern(
                            (config.prepared_bytes,),
                            offset=tile * config.runtime_tile_bytes,
                            sizes=[
                                (
                                    slabs_per_group
                                    * config.panel_slab
                                    // b_dma_panels
                                ),
                                b_dma_panels,
                                config.runtime_tile_bytes // 64,
                                64,
                            ],
                            strides=[
                                b_dma_panels * config.runtime_tile_bytes,
                                config.runtime_tile_bytes,
                                64,
                                1,
                            ],
                        )
                        B_hs[col].fill(B, tap=b_tap, group=transfer_block)
                        B_taps.append(b_tap)

                    for local_slab in range(slabs_per_group):
                        slab_base = (
                            panel_base + local_slab * config.panel_slab
                        )
                        for row in range(N_AIE_ROWS):
                            c_tap = TensorAccessPattern(
                                (M, N),
                                offset=(
                                    wave_base * config.m_wave
                                    + row * config.m_a
                                )
                                * N
                                + slab_base * n,
                                sizes=[
                                    group_waves,
                                    c_objects_per_slab,
                                    config.m_a,
                                    config.c_panel_slab * n,
                                ],
                                strides=[
                                    config.m_wave * N,
                                    config.c_panel_slab * n,
                                    N,
                                    1,
                                ],
                            )
                            C_hs[row].drain(
                                C,
                                tap=c_tap,
                                wait=True,
                                group=transfer_block,
                            )
                            C_taps.append(c_tap)
                    transfer_block.finish()
            return

        if joint_slab:
            group_waves = 2
            c_objects_per_slab = (
                config.panel_slab // config.c_panel_slab
            )

            # Two complete four-row activation waves share every resident Q4
            # slab.  A is loaded once per pair, the full stage-major B sweep
            # is one long task per column, and one C task per slab scatters
            # [wave, C8-object] order back into the two host row bands.
            for wave_base in range(0, waves, group_waves):
                transfer_block = TaskGroup()
                for col in range(N_AIE_COLS):
                    a_tap = TensorAccessPattern(
                        (M, K),
                        offset=wave_base * config.m_wave * K
                        + col * config.k_stage,
                        sizes=[
                            group_waves * N_AIE_ROWS,
                            config.m_a,
                            config.k_stage // 2,
                            2,
                        ],
                        strides=[config.m_a * K, K, 2, 1],
                    )
                    A_hs[col].fill(
                        A, tap=a_tap, group=transfer_block
                    )
                    A_taps.append(a_tap)

                for col in range(N_AIE_COLS):
                    tile = col * panels
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=tile * config.runtime_tile_bytes,
                        sizes=[
                            panels // b_dma_panels,
                            b_dma_panels,
                            config.runtime_tile_bytes // 64,
                            64,
                        ],
                        strides=[
                            b_dma_panels * config.runtime_tile_bytes,
                            config.runtime_tile_bytes,
                            64,
                            1,
                        ],
                    )
                    B_hs[col].fill(
                        B, tap=b_tap, group=transfer_block
                    )
                    B_taps.append(b_tap)

                for slab_base in range(
                    0, panels, config.panel_slab
                ):
                    for row in range(N_AIE_ROWS):
                        c_tap = TensorAccessPattern(
                            (M, N),
                            offset=(
                                wave_base * config.m_wave
                                + row * config.m_a
                            )
                            * N
                            + slab_base * n,
                            sizes=[
                                group_waves,
                                c_objects_per_slab,
                                config.m_a,
                                config.c_panel_slab * n,
                            ],
                            strides=[
                                config.m_wave * N,
                                config.c_panel_slab * n,
                                N,
                                1,
                            ],
                        )
                        C_hs[row].drain(
                            C,
                            tap=c_tap,
                            wait=True,
                            group=transfer_block,
                        )
                        C_taps.append(c_tap)
                transfer_block.finish()
            return

        if memtile_slab_stream:
            c_objects = panels // config.c_panel_slab

            # M-wave outer keeps one converted A shard stationary while
            # ping-pong compressed-Q4 slabs cover the complete N dimension.
            # MemTile reorders each token-major C object on its outbound DMA;
            # one repeated shim task per row then drains every C object into a
            # contiguous full-width host band.  The whole wave therefore fits
            # in one TaskGroup and never reprograms a live FIFO mid-sweep.
            for wave in range(waves):
                transfer_block = TaskGroup()
                if row_stream_a:
                    for row in range(N_AIE_ROWS):
                        for col in range(N_AIE_COLS):
                            a_tap = TensorAccessPattern(
                                (M, K),
                                offset=(
                                    wave * config.m_wave
                                    + row * config.m_a
                                )
                                * K
                                + col * config.k_stage,
                                sizes=[
                                    1,
                                    config.m_a,
                                    config.k_stage // 2,
                                    2,
                                ],
                                strides=[0, K, 2, 1],
                            )
                            A_hs[col].fill(
                                A, tap=a_tap, group=transfer_block
                            )
                            A_taps.append(a_tap)
                else:
                    for col in range(N_AIE_COLS):
                        a_tap = TensorAccessPattern(
                            (M, K),
                            offset=wave * config.m_wave * K
                            + col * config.k_stage,
                            sizes=[
                                N_AIE_ROWS,
                                config.m_a,
                                config.k_stage // 2,
                                2,
                            ],
                            strides=[config.m_a * K, K, 2, 1],
                        )
                        A_hs[col].fill(
                            A, tap=a_tap, group=transfer_block
                        )
                        A_taps.append(a_tap)

                # Prepared weights are stage-major then N-panel-major, hence
                # every column's complete N sweep is contiguous in L3.  Use
                # one long repeated task rather than starting one task per
                # MemTile slab.  Each 40-KiB BD remains hardware-safe; four
                # repeats form one 160-KiB FIFO object and the depth-two parent
                # naturally ping-pongs while workers consume the N sweep.
                for col in range(N_AIE_COLS):
                    tile = col * panels
                    b_tap = TensorAccessPattern(
                        (config.prepared_bytes,),
                        offset=tile * config.runtime_tile_bytes,
                        sizes=[
                            panels // b_dma_panels,
                            b_dma_panels,
                            config.runtime_tile_bytes // 64,
                            64,
                        ],
                        strides=[
                            b_dma_panels * config.runtime_tile_bytes,
                            config.runtime_tile_bytes,
                            64,
                            1,
                        ],
                    )
                    B_hs[col].fill(
                        B, tap=b_tap, group=transfer_block
                    )
                    B_taps.append(b_tap)

                for row in range(N_AIE_ROWS):
                    c_tap = TensorAccessPattern(
                        (M, N),
                        offset=(
                            wave * config.m_wave + row * config.m_a
                        )
                        * N,
                        sizes=[
                            c_objects,
                            1,
                            config.m_a,
                            config.c_panel_slab * n,
                        ],
                        strides=[
                            config.c_panel_slab * n,
                            0,
                            N,
                            1,
                        ],
                    )
                    C_hs[row].drain(
                        C,
                        tap=c_tap,
                        wait=True,
                        group=transfer_block,
                    )
                    C_taps.append(c_tap)
                transfer_block.finish()
            return

        if fused_cache_group:
            replay_waves = config.weight_replay_waves
            for cache_base in range(0, panels, config.panel_slab):
                for wave_base in range(0, waves, replay_waves):
                    transfer_block = TaskGroup()

                    # Match the proven whole-array schedule: the large B load,
                    # all activation waves that consume it, and their grouped
                    # C drains are one bounded transfer block.  This lets L3
                    # DMA overlap MemTile replay and compute without nested
                    # task groups or a barrier between load and consumption.
                    for col in range(N_AIE_COLS):
                        tile = col * panels + cache_base
                        b_tap = TensorAccessPattern(
                            (config.prepared_bytes,),
                            offset=tile * config.runtime_tile_bytes,
                            sizes=[
                                config.panel_slab // b_dma_panels,
                                b_dma_panels,
                                config.runtime_tile_bytes // 64,
                                64,
                            ],
                            strides=[
                                b_dma_panels * config.runtime_tile_bytes,
                                config.runtime_tile_bytes,
                                64,
                                1,
                            ],
                        )
                        B_hs[col].fill(B, tap=b_tap, group=transfer_block)
                        B_taps.append(b_tap)

                    for wave in range(wave_base, wave_base + replay_waves):
                        for col in range(N_AIE_COLS):
                            a_tap = TensorAccessPattern(
                                (M, K),
                                offset=wave * config.m_wave * K
                                + col * config.k_stage,
                                sizes=[
                                    N_AIE_ROWS,
                                    config.m_a,
                                    config.k_stage // 2,
                                    2,
                                ],
                                strides=[config.m_a * K, K, 2, 1],
                            )
                            A_hs[col].fill(
                                A, tap=a_tap, group=transfer_block
                            )
                            A_taps.append(a_tap)

                        if config.c_panel_slab == 1:
                            for panel in range(
                                cache_base,
                                cache_base + config.panel_slab,
                            ):
                                for row in range(N_AIE_ROWS):
                                    c_tap = TensorAccessPattern(
                                        (M, N),
                                        offset=(
                                            wave * config.m_wave
                                            + row * config.m_a
                                        )
                                        * N
                                        + panel * n,
                                        sizes=[
                                            config.m_a // 16,
                                            n // 16,
                                            16,
                                            16,
                                        ],
                                        strides=[16 * N, 16, N, 1],
                                    )
                                    C_hs[row].drain(
                                        C,
                                        tap=c_tap,
                                        wait=True,
                                        group=transfer_block,
                                    )
                                    C_taps.append(c_tap)
                        else:
                            for c_base in range(
                                cache_base,
                                cache_base + config.panel_slab,
                                config.c_panel_slab,
                            ):
                                for row in range(N_AIE_ROWS):
                                    c_tap = TensorAccessPattern(
                                        (M, N),
                                        offset=(
                                            wave * config.m_wave
                                            + row * config.m_a
                                        )
                                        * N
                                        + c_base * n,
                                        sizes=[
                                            config.m_a // 16,
                                            config.c_panel_slab * n // 16,
                                            16,
                                            16,
                                        ],
                                        strides=[16 * N, 16, N, 1],
                                    )
                                    C_hs[row].drain(
                                        C,
                                        tap=c_tap,
                                        wait=True,
                                        group=transfer_block,
                                    )
                                    C_taps.append(c_tap)
                    transfer_block.finish()
            return

        if memtile_weight_cache:
            replay_waves = config.weight_replay_waves
            for cache_base in range(0, panels, config.panel_slab):
                for wave_base in range(0, waves, replay_waves):
                    load_group = TaskGroup()

                    # Transfer one contiguous KxN compressed slab per column.
                    # The child MemTile DMA replays this slab for each wave in
                    # the bounded group without another L3 weight transfer.
                    for col in range(N_AIE_COLS):
                        tile = col * panels + cache_base
                        b_tap = TensorAccessPattern(
                            (config.prepared_bytes,),
                            offset=tile * config.runtime_tile_bytes,
                            sizes=[
                                config.panel_slab // b_dma_panels,
                                b_dma_panels,
                                config.runtime_tile_bytes // 64,
                                64,
                            ],
                            strides=[
                                b_dma_panels * config.runtime_tile_bytes,
                                config.runtime_tile_bytes,
                                64,
                                1,
                            ],
                        )
                        B_hs[col].fill(B, tap=b_tap, group=load_group)
                        B_taps.append(b_tap)

                    # Complete the shim-to-MemTile transfer before starting
                    # any A/C task group.  The ObjectFIFO consumer locks keep
                    # the slab resident while the MemTile MM2S BD replays it;
                    # the host DMA task itself does not need to remain live.
                    # This avoids nested TaskGroups and releases the four B
                    # shim BDs before A/C descriptors are allocated.
                    load_group.finish()

                    for wave in range(wave_base, wave_base + replay_waves):
                        # B remains resident under ObjectFIFO locks while this
                        # bounded wave group allocates and frees A/C shim BDs.
                        wave_group = TaskGroup()
                        if row_stream_a:
                            for row in range(N_AIE_ROWS):
                                for col in range(N_AIE_COLS):
                                    a_tap = TensorAccessPattern(
                                        (M, K),
                                        offset=(
                                            wave * config.m_wave
                                            + row * config.m_a
                                        )
                                        * K
                                        + col * config.k_stage,
                                        sizes=[
                                            1,
                                            config.m_a,
                                            config.k_stage // 2,
                                            2,
                                        ],
                                        strides=[0, K, 2, 1],
                                    )
                                    A_hs[col].fill(
                                        A, tap=a_tap, group=wave_group
                                    )
                                    A_taps.append(a_tap)
                        else:
                            for col in range(N_AIE_COLS):
                                a_tap = TensorAccessPattern(
                                    (M, K),
                                    offset=wave * config.m_wave * K
                                    + col * config.k_stage,
                                    sizes=[
                                        N_AIE_ROWS,
                                        config.m_a,
                                        config.k_stage // 2,
                                        2,
                                    ],
                                    strides=[config.m_a * K, K, 2, 1],
                                )
                                A_hs[col].fill(
                                    A, tap=a_tap, group=wave_group
                                )
                                A_taps.append(a_tap)

                        c_slabs = config.panel_slab // config.c_panel_slab
                        split_output = c_slabs > 8
                        if split_output:
                            # Free the four A descriptors before allocating
                            # bounded C groups. B remains resident in MemTile.
                            wave_group.finish()
                        output_span = (
                            8 * config.c_panel_slab
                            if split_output
                            else config.panel_slab
                        )
                        for output_base in range(
                            cache_base,
                            cache_base + config.panel_slab,
                            output_span,
                        ):
                            output_end = min(
                                output_base + output_span,
                                cache_base + config.panel_slab,
                            )
                            output_group = (
                                TaskGroup() if split_output else wave_group
                            )
                            if config.c_panel_slab == 1:
                                for panel in range(output_base, output_end):
                                    for row in range(N_AIE_ROWS):
                                        c_tap = TensorAccessPattern(
                                            (M, N),
                                            offset=(
                                                wave * config.m_wave
                                                + row * config.m_a
                                            )
                                            * N
                                            + panel * n,
                                            sizes=[
                                                config.m_a // 16,
                                                n // 16,
                                                16,
                                                16,
                                            ],
                                            strides=[16 * N, 16, N, 1],
                                        )
                                        C_hs[row].drain(
                                            C,
                                            tap=c_tap,
                                            wait=True,
                                            group=output_group,
                                        )
                                        C_taps.append(c_tap)
                            else:
                                for c_base in range(
                                    output_base,
                                    output_end,
                                    config.c_panel_slab,
                                ):
                                    for row in range(N_AIE_ROWS):
                                        c_tap = TensorAccessPattern(
                                            (M, N),
                                            offset=(
                                                wave * config.m_wave
                                                + row * config.m_a
                                            )
                                            * N
                                            + c_base * n,
                                            sizes=[
                                                config.m_a // 16,
                                                config.c_panel_slab * n // 16,
                                                16,
                                                16,
                                            ],
                                            strides=[16 * N, 16, N, 1],
                                        )
                                        C_hs[row].drain(
                                            C,
                                            tap=c_tap,
                                            wait=True,
                                            group=output_group,
                                        )
                                        C_taps.append(c_tap)
                            if split_output:
                                output_group.finish()
                        if not split_output:
                            wave_group.finish()
            return

        schedule_span = max(config.panel_slab, config.c_panel_slab)
        for wave in range(waves):
            for schedule_base in range(0, worker_panels, schedule_span):
                weight_group = TaskGroup()
                if schedule_base == 0:
                    if row_stream_a:
                        for row in range(N_AIE_ROWS):
                            for col in range(N_AIE_COLS):
                                a_tap = TensorAccessPattern(
                                    (M, K),
                                    offset=(
                                        wave * config.m_wave + row * config.m_a
                                    )
                                    * K
                                    + col * config.k_stage,
                                    sizes=[
                                        1,
                                        config.m_a,
                                        config.k_stage // 2,
                                        2,
                                    ],
                                    strides=[0, K, 2, 1],
                                )
                                A_hs[col].fill(A, tap=a_tap, group=weight_group)
                                A_taps.append(a_tap)
                    else:
                        for col in range(N_AIE_COLS):
                            a_tap = TensorAccessPattern(
                                (M, K),
                                offset=wave * config.m_wave * K
                                + col * config.k_stage,
                                sizes=[
                                    N_AIE_ROWS,
                                    config.m_a,
                                    config.k_stage // 2,
                                    2,
                                ],
                                strides=[config.m_a * K, K, 2, 1],
                            )
                            A_hs[col].fill(A, tap=a_tap, group=weight_group)
                            A_taps.append(a_tap)
                schedule_end = schedule_base + schedule_span
                for slab_base in range(
                    schedule_base, schedule_end, config.panel_slab
                ):
                    for col in range(N_AIE_COLS):
                        tile = col * panels + slab_base
                        b_tap = TensorAccessPattern(
                            (config.prepared_bytes,),
                            offset=tile * config.runtime_tile_bytes,
                            sizes=[
                                1,
                                config.panel_slab,
                                config.runtime_tile_bytes // 64,
                                64,
                            ],
                            strides=[0, config.runtime_tile_bytes, 64, 1],
                        )
                        B_hs[col].fill(B, tap=b_tap, group=weight_group)
                        B_taps.append(b_tap)

                if config.c_panel_slab == 1:
                    for panel in range(schedule_base, schedule_end):
                        for row in range(N_AIE_ROWS):
                            c_tap = TensorAccessPattern(
                                (M, N),
                                offset=(wave * config.m_wave + row * config.m_a) * N
                                + panel * n,
                                sizes=[config.m_a // 16, n // 16, 16, 16],
                                strides=[16 * N, 16, N, 1],
                            )
                            C_hs[row].drain(
                                C, tap=c_tap, wait=True, group=weight_group
                            )
                            C_taps.append(c_tap)
                else:
                    for c_base in range(
                        schedule_base, schedule_end, config.c_panel_slab
                    ):
                        for row in range(N_AIE_ROWS):
                            c_tap = TensorAccessPattern(
                                (M, N),
                                offset=(
                                    wave * config.m_wave + row * config.m_a
                                )
                                * N
                                + c_base * n,
                                sizes=[
                                    config.m_a // 16,
                                    config.c_panel_slab * n // 16,
                                    16,
                                    16,
                                ],
                                strides=[16 * N, 16, N, 1],
                            )
                            C_hs[row].drain(
                                C, tap=c_tap, wait=True, group=weight_group
                            )
                            C_taps.append(c_tap)
                weight_group.finish()

    runtime = Runtime(
        sequence,
        [A_ty, B_ty, C_ty, A_prods, B_prods, C_conses],
    )
    if column_dequant_cache:
        # The north core expands Q4 into MemTile S2MM channel 5.  MM2S channel
        # 4 replays the resident BFP slab and the switch duplicates every beat
        # to all four Core slave-stream port-zero endpoints.  There is one
        # dequantization and one MemTile read per column, not per compute row.
        for col in range(N_AIE_COLS):
            runtime.add_flow(
                Flow(
                    src=workers[0][col].tile,
                    dst=B_cache_tiles[col],
                    src_port=WireBundle.Core,
                    src_channel=0,
                    dst_port=WireBundle.DMA,
                    dst_channel=5,
                )
            )
            for row in range(N_AIE_ROWS):
                runtime.add_flow(
                    Flow(
                        src=B_cache_tiles[col],
                        dst=workers[row][col].tile,
                        src_port=WireBundle.DMA,
                        src_channel=4,
                        dst_port=WireBundle.Core,
                        dst_channel=0,
                    )
                )
        for lock in B_cache_locks:
            runtime.add_lock(lock)
        for tile_dma in B_cache_dmas:
            runtime.add_tile_dma(tile_dma)
    elif core_multicast:
        for col in range(N_AIE_COLS):
            for row in range(1, N_AIE_ROWS):
                runtime.add_flow(
                    Flow(
                        src=Tile(col, 2),
                        dst=Tile(col, row + 2),
                        src_port=WireBundle.Core,
                        src_channel=0,
                        dst_port=WireBundle.Core,
                        dst_channel=0,
                    )
                )
    elif fused_stream:
        for col in range(N_AIE_COLS):
            for row in range(N_AIE_ROWS - 1):
                runtime.add_flow(
                    Flow(
                        src=Tile(col, row + 2),
                        dst=Tile(col, row + 3),
                        src_port=WireBundle.Core,
                        src_channel=0,
                        dst_port=WireBundle.Core,
                        dst_channel=0,
                    )
                )
    program = Program(dev, runtime, workers=flat_workers)
    if trace_config:
        program.enable_trace(
            trace_config.trace_size,
            workers=[workers[0][4]],
            egress_shim_col=4,
        )
    module = program.resolve_program()
    if generate_taps:
        return (
            TensorAccessSequence.from_taps(A_taps),
            TensorAccessSequence.from_taps(B_taps),
            TensorAccessSequence.from_taps(C_taps),
        )
    return module


@iron.jit(
    aiecc_flags=["--alloc-scheme=basic-sequential"],
    source_files=[KERNEL_SOURCE, __file__],
)
def whole_array_q4ks_systolic(
    A: In,
    prepared_B: In,
    C: Out,
    *,
    M: CompileTime[int] = 4096,
    K: CompileTime[int] = 4096,
    N: CompileTime[int] = 4096,
    n: CompileTime[int] = 32,
    m_a: CompileTime[int] = 32,
    weight_flow: CompileTime[str] = "q4-local",
    transport: CompileTime[str] = "core-stream",
    panel_slab: CompileTime[int] = 1,
    c_panel_slab: CompileTime[int] = 1,
    trace_config: CompileTime[TraceConfig | None] = None,
):
    return _build_design(
        iron.get_current_device(),
        M,
        K,
        N,
        n,
        m_a,
        weight_flow,
        transport,
        panel_slab,
        c_panel_slab,
        trace_config,
    )


whole_array_systolic = whole_array_q4ks_systolic


def generate_taps(**kwargs):
    dev = _device_for("npu2")
    iron.set_current_device(dev)
    values = dict(
        M=256,
        K=2048,
        N=256,
        n=32,
        m_a=32,
        weight_flow="q4-local",
        transport="core-stream",
        panel_slab=1,
        c_panel_slab=1,
        trace_config=None,
    )
    values.update(kwargs)
    return _build_design(dev, **values, generate_taps=True)


def _source_config(config: SystolicConfig) -> Q4KSConfig:
    return Q4KSConfig(
        M=config.M,
        K=config.K,
        N=config.N,
        m_c=32,
        m_a=16,
        k=256,
        n=16,
        n_aie_cols=8,
        compute_type="bfp16",
        accumulation_mode="bf16",
        cache_mode="l1-weight",
    )


def _round_bfp_rows(values: np.ndarray) -> np.ndarray:
    """Round independent K rows to the native 8-value BFP16 ABI."""

    values = np.ascontiguousarray(values, dtype=np.float32)
    packed = float_to_bfp16ebs8(values.reshape(-1))
    return bfp16ebs8_to_float(packed).reshape(values.shape)


def _selected_products(lhs, rhs, row_ids, col_ids):
    """Evaluate selected row/column products without a huge sampled GEMM."""

    pair_count = row_ids.size
    is_cartesian = (
        pair_count == lhs.shape[0] * rhs.shape[0]
        and np.array_equal(
            row_ids, np.repeat(np.arange(lhs.shape[0]), rhs.shape[0])
        )
        and np.array_equal(
            col_ids, np.tile(np.arange(rhs.shape[0]), lhs.shape[0])
        )
    )
    if is_cartesian:
        return (lhs @ rhs.T).reshape(-1)

    result = np.empty(pair_count, np.float32)
    for start in range(0, pair_count, 256):
        stop = min(start + 256, pair_count)
        result[start:stop] = np.einsum(
            "ik,ik->i",
            lhs[row_ids[start:stop]],
            rhs[col_ids[start:stop]],
            optimize=True,
        )
    return result


def _sample_references(A, native, config, rows, cols):
    q, scales_native, biases_native = decode_q4_k(
        native, K=config.K, N=config.N
    )
    scales = scales_native.astype(bfloat16).astype(np.float32)
    biases = biases_native.astype(bfloat16).astype(np.float32)
    groups = np.arange(config.K) // 32
    unique_rows, row_ids = np.unique(rows, return_inverse=True)
    unique_cols, col_ids = np.unique(cols, return_inverse=True)

    activation = A[unique_rows].astype(np.float32)
    weights = (
        q[:, unique_cols].astype(np.float32) * scales[groups][:, unique_cols]
        - biases[groups][:, unique_cols]
    ).astype(bfloat16).astype(np.float32).T

    native = _selected_products(activation, weights, row_ids, col_ids)
    rounded_a = _round_bfp_rows(activation)
    rounded_b = _round_bfp_rows(weights)
    bfp = _selected_products(rounded_a, rounded_b, row_ids, col_ids)
    return (
        np.asarray(native, dtype=bfloat16).astype(np.float32),
        np.asarray(bfp, dtype=bfloat16).astype(np.float32),
    )


def _error_metrics(observed: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.abs(observed - expected)
    return {
        "max_abs": float(delta.max(initial=0)),
        "max_rel": float(
            (delta / np.maximum(np.abs(expected), 1.0e-12)).max(initial=0)
        ),
        "nrmse": float(
            np.sqrt(np.mean(delta * delta))
            / max(np.sqrt(np.mean(expected * expected)), 1.0e-12)
        ),
    }


def _max_bf16_ulp(observed: np.ndarray, expected: np.ndarray) -> int:
    def ordered(values):
        bits = np.asarray(values, dtype=bfloat16).view(np.uint16).astype(np.int32)
        return np.where(
            bits & 0x8000,
            0x8000 - (bits & 0x7FFF),
            0x8000 + bits,
        )

    return int(np.max(np.abs(ordered(observed) - ordered(expected)), initial=0))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="Q4_K stationary-A systolic matmul")
    add_compile_args(
        parser, short_dev=None, dev_choices=("npu2",), default_dev="npu2"
    )
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("--n-tile", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--m-tile", type=int, choices=(32, 64), default=32)
    parser.add_argument("--weight-flow", choices=WEIGHT_FLOWS, default="q4-local")
    parser.add_argument("--transport", choices=TRANSPORTS, default="auto")
    parser.add_argument(
        "--panel-slab", type=int, choices=(1, 2, 4, 8, 16, 32), default=1
    )
    parser.add_argument(
        "--c-panel-slab", type=int, choices=(1, 2, 4, 8), default=1
    )
    parser.add_argument("--q4-k-file", type=Path)
    parser.add_argument(
        "--verify-mode", choices=("auto", "full", "sampled", "none"), default="auto"
    )
    parser.add_argument("--verify-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0x53595354)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--min-gflops", type=float, default=0.0)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--benchmark-csv", type=Path)
    add_trace_arg(parser, with_short=False)
    add_benchmark_args(parser, default_warmup=1, default_iters=1)
    return parser


def _config(opts) -> SystolicConfig:
    return SystolicConfig(
        m_a=opts.m_tile,
        M=opts.M,
        K=opts.K,
        N=opts.N,
        n=opts.n_tile,
        weight_flow=opts.weight_flow,
        transport=opts.transport,
        panel_slab=opts.panel_slab,
        c_panel_slab=opts.c_panel_slab,
    )


def _kwargs(opts):
    config = _config(opts)
    return dict(
        M=config.M,
        K=config.K,
        N=config.N,
        m_a=config.m_a,
        n=config.n,
        weight_flow=config.weight_flow,
        transport=(
            "core-stream" if config.transport == "auto" else config.transport
        ),
        panel_slab=config.panel_slab,
        c_panel_slab=config.c_panel_slab,
        trace_config=TraceConfig(trace_size=opts.trace_size) if opts.trace_size else None,
    )


def _validate(opts):
    try:
        _config(opts)
    except (TypeError, ValueError) as error:
        sys.exit(str(error))
    if opts.warmup < 0 or opts.iters < 1 or opts.benchmark_repeats < 1:
        sys.exit("warmup must be >= 0 and iterations/repeats must be positive")
    if opts.min_gflops < 0 or opts.verify_samples < 1:
        sys.exit("min-gflops must be non-negative and verify-samples positive")


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


def _run(opts):
    config = _config(opts)
    if opts.q4_k_file:
        native = np.fromfile(opts.q4_k_file, np.uint8)
        if native.shape != (config.native_bytes,):
            raise ValueError(
                f"Q4_K file contains {native.size} bytes; expected {config.native_bytes}"
            )
        A, _ = make_deterministic_native_q4_k(
            _source_config(config), seed=opts.seed
        )
    else:
        A, native = make_deterministic_native_q4_k(
            _source_config(config), seed=opts.seed
        )
    start = time.perf_counter()
    prepared = prepare_systolic_q4ks_weights(native, config)
    prepare_ms = (time.perf_counter() - start) * 1000.0

    if config.transport in (
        "bfp-broadcast-preconverted-a",
        "bfp-direct-c-preconverted-a",
        "q4-direct-c-preconverted-a",
    ):
        prepared_a = prepare_systolic_bfp_activations(A, config)
        A_tensor = iron.tensor(prepared_a, dtype=np.uint8, device="npu")
    else:
        A_tensor = iron.tensor(A.reshape(-1), dtype=bfloat16, device="npu")
    B_tensor = iron.tensor(prepared, dtype=np.uint8, device="npu")
    C_tensor = iron.zeros(config.M * config.N, dtype=bfloat16, device="npu")
    throughputs = []
    times = []
    for repeat in range(opts.benchmark_repeats):
        bench: BenchmarkResult = run_iters(
            whole_array_q4ks_systolic,
            A_tensor,
            B_tensor,
            C_tensor,
            **_kwargs(opts),
            warmup=opts.warmup,
            iters=opts.iters,
        )
        if bench.npu is None:
            raise RuntimeError("runtime returned no NPU timing")
        gflops = 2.0 * config.M * config.K * config.N / (
            1000.0 * bench.npu.avg_us
        )
        throughputs.append(gflops)
        times.append(bench.npu.avg_us)
        print(
            f"round {repeat + 1}: {bench.npu.avg_us:.2f} us, "
            f"{gflops:.2f} GFLOP/s"
        )

    actual = C_tensor.numpy().reshape(config.M, config.N).astype(np.float32)
    if opts.diagnostics:
        wave_view = actual.reshape(
            config.M // config.m_wave, config.m_wave, config.N
        )
        for physical_row in range(N_AIE_ROWS):
            row_slice = wave_view[
                :, physical_row * config.m_a : (physical_row + 1) * config.m_a, :
            ]
            finite = np.isfinite(row_slice)
            finite_values = row_slice[finite]
            value_range = (
                f"[{finite_values.min():.6g}, {finite_values.max():.6g}]"
                if finite_values.size
                else "[no finite values]"
            )
            print(
                f"physical row {physical_row}: finite "
                f"{finite.sum()}/{finite.size}, range {value_range}"
            )
    metrics = {
        key: None
        for key in (
            "max_abs",
            "max_rel",
            "nrmse",
            "bfp_max_abs",
            "bfp_max_rel",
            "bfp_nrmse",
            "operand_nrmse",
            "max_bf16_ulp",
        )
    }
    verify_output = (
        opts.verify_mode != "none" and config.weight_flow != "resident"
    )
    if opts.verify_mode != "none" and config.weight_flow == "resident":
        print("resident is a single-tile compute ceiling; output verification skipped")
    if verify_output:
        full = opts.verify_mode == "full" or (
            opts.verify_mode == "auto"
            and config.M * config.K * config.N <= VERIFY_SCALAR_PRODUCT_THRESHOLD
        )
        count = config.M * config.N if full else opts.verify_samples
        if full:
            rows = np.repeat(np.arange(config.M), config.N)
            cols = np.tile(np.arange(config.N), config.M)
        else:
            rng = np.random.default_rng(opts.seed ^ 0x5134)
            rows = rng.integers(0, config.M, count)
            cols = rng.integers(0, config.N, count)
        native_expected, bfp_expected = _sample_references(
            A, native, config, rows, cols
        )
        observed = actual[rows, cols]
        native_metrics = _error_metrics(observed, native_expected)
        bfp_metrics = _error_metrics(observed, bfp_expected)
        operand_metrics = _error_metrics(bfp_expected, native_expected)
        metrics = {
            **native_metrics,
            "bfp_max_abs": bfp_metrics["max_abs"],
            "bfp_max_rel": bfp_metrics["max_rel"],
            "bfp_nrmse": bfp_metrics["nrmse"],
            "operand_nrmse": operand_metrics["nrmse"],
            "max_bf16_ulp": _max_bf16_ulp(observed, native_expected),
        }
        np.testing.assert_allclose(
            observed, native_expected, rtol=0.05, atol=0.5
        )
        print(f"Verification passed; errors: {metrics}")

    median = statistics.median(throughputs)
    result = {
        "variant": config.weight_flow,
        "transport": (
            "core-stream"
            if config.transport == "auto"
            else config.transport
        ),
        "eligible": config.weight_flow
        in ("q4-local", "q4-direct", "q4-expand-once"),
        "M": config.M,
        "K": config.K,
        "N": config.N,
        "m_a": config.m_a,
        "k_stage": config.k_stage,
        "n": config.n,
        "panel_slab": config.panel_slab,
        "c_panel_slab": config.c_panel_slab,
        "core_memory_bytes": config.worst_core_memory_bytes,
        "memtile_memory_bytes": config.memtile_bytes,
        "prepare_ms": prepare_ms,
        "average_npu_us": statistics.mean(times),
        "median_gflops": median,
        **metrics,
    }
    _write_results(opts, result)
    print(f"prepared weights in {prepare_ms:.2f} ms; median {median:.2f} GFLOP/s")
    if opts.min_gflops and median < opts.min_gflops:
        raise RuntimeError(
            f"performance gate failed: {median:.2f} < {opts.min_gflops:.2f} GFLOP/s"
        )
    print("PASS!")


def main():
    opts = _parser().parse_args()
    run_design_cli(
        whole_array_q4ks_systolic,
        opts,
        compile_kwargs=_kwargs,
        run_and_verify=_run,
        device=lambda parsed: _device_for(parsed.dev),
        validate=_validate,
    )


if __name__ == "__main__":
    main()
