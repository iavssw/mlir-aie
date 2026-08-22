# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Packing and validation for the stationary-A Q4_K systolic experiment.

The prepared representation remains quantized Q4_K.  It only changes the
model-load ordering to match the physical array: K stage (AIE column), N
panel, then the regular 8x8 nibble microtiles and BF16 scale/bias metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
from ml_dtypes import bfloat16

from packing import (
    CORE_MEMORY_BYTES,
    CORE_SAFETY_BYTES,
    CORE_STACK_BYTES,
    MEMTILE_MEMORY_BYTES,
    MEMTILE_SAFETY_BYTES,
    Q4_K_GROUP,
    Q4_K_BLOCK_BYTES,
    QK_K,
    bfp16ebs8_to_float,
    decode_q4_k,
    float_to_bfp16ebs8,
)

N_AIE_COLS = 8
N_AIE_ROWS = 4
A_CHUNK_K = 64
M_A = 32
WEIGHT_FLOWS = (
    "q4-local",
    "q4-direct",
    "q4-expand-once",
    "bfp-prepared",
    "resident",
)
TRANSPORTS = (
    "auto",
    "core-stream",
    "shim-stream",
    "tile-dma",
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
    "memtile-slab",
    "memtile-slab-direct-a",
    "memtile-fused-stream",
    "joint-slab",
    "joint-fused",
)


@dataclass(frozen=True)
class SystolicConfig:
    M: int = 4096
    K: int = 4096
    N: int = 4096
    n: int = 32
    m_a: int = M_A
    weight_flow: str = "q4-local"
    transport: str = "auto"
    panel_slab: int = 1
    c_panel_slab: int = 1

    def __post_init__(self) -> None:
        self.validate()

    @property
    def m_wave(self) -> int:
        return N_AIE_ROWS * self.m_a

    @property
    def k_stage(self) -> int:
        return self.K // N_AIE_COLS

    @property
    def n_panels(self) -> int:
        return self.N // self.n

    @property
    def weight_replay_waves(self) -> int:
        """Largest validated MemTile replay divisor, capped at four waves."""

        waves = self.M // self.m_wave
        if self.transport in (
            "memtile-fused-persistent",
            "bfp-memtile-persistent",
            "bfp-memtile-broadcast-persistent",
            "bfp-core-multicast-persistent",
            "q4-memtile-broadcast-persistent",
            "q4-expand-broadcast-persistent",
            "q4-column-dequant-cache",
            "bfp-broadcast-preconverted-a",
            "q4-direct-c-preconverted-a",
        ):
            return waves
        replay_cap = min(4, waves)
        if self.transport == "memtile-fused-cache":
            # Direct Core-slave ObjectFIFO replay is hardware-correct at two
            # waves.  Four waves returns ERT state 6 on this NPU2/FW pairing,
            # although the same count is valid for buffered FIFO consumers.
            replay_cap = min(replay_cap, 2)
        if self.transport in (
            "memtile-cache",
            "memtile-fused-cache",
        ) and self.m_a == M_A:
            # A fused transfer block owns one B descriptor plus, per replayed
            # wave, one A descriptor and one C descriptor for every grouped-C
            # slab.  All directions share the shim's 16-BD pool.
            descriptors_per_wave = 1 + self.panel_slab // self.c_panel_slab
            replay_cap = min(
                replay_cap,
                (16 - 1) // descriptors_per_wave,
            )
        return next(
            replay
            for replay in range(replay_cap, 0, -1)
            if waves % replay == 0
        )

    @property
    def weight_cache_slabs(self) -> int:
        return self.n_panels // self.panel_slab

    @property
    def groups_per_tile(self) -> int:
        return self.k_stage // Q4_K_GROUP

    @property
    def q4_weight_bytes(self) -> int:
        return self.k_stage * self.n // 2

    @property
    def q4_metadata_bytes(self) -> int:
        return self.groups_per_tile * self.n * 4

    @property
    def q4_tile_bytes(self) -> int:
        return self.q4_weight_bytes + self.q4_metadata_bytes

    @property
    def bfp_tile_bytes(self) -> int:
        return self.k_stage * self.n * 9 // 8

    @property
    def runtime_tile_bytes(self) -> int:
        return (
            self.bfp_tile_bytes
            if self.weight_flow in ("bfp-prepared", "resident")
            else self.q4_tile_bytes
        )

    @property
    def prepared_bytes(self) -> int:
        return N_AIE_COLS * self.n_panels * self.runtime_tile_bytes

    @property
    def native_bytes(self) -> int:
        return self.N * (self.K // QK_K) * Q4_K_BLOCK_BYTES

    @property
    def a_panel_values(self) -> int:
        return self.m_a * self.k_stage

    @property
    def a_panel_bfp_bytes(self) -> int:
        return self.a_panel_values * 9 // 8

    @property
    def a_chunk_k(self) -> int:
        if self.transport == "q4-column-dequant-cache" and self.m_a == 64:
            # Row-streamed A needs one input and one forwarding chunk on the
            # first three cores.  A 16-K chunk keeps the north/east core below
            # 64 KiB while it also owns the Q4 tile and C staging buffer.
            return 16
        if self.transport in (
            "q4-expand-broadcast-persistent",
            "q4-column-dequant-cache",
        ):
            return 32
        if self.m_a > M_A:
            return 16 if self.c_panel_slab > 1 else 32
        return A_CHUNK_K

    @property
    def a_chunk_values(self) -> int:
        return self.m_a * self.a_chunk_k

    def core_memory_components(
        self, *, cascade_role: str = "middle", physical_row: int = 0
    ) -> dict[str, int]:
        if cascade_role not in ("west", "middle", "east"):
            raise ValueError("cascade_role must be west, middle, or east")
        if physical_row not in range(N_AIE_ROWS):
            raise ValueError("physical_row must be in 0..3")

        preconverted_a = self.transport in (
            "bfp-broadcast-preconverted-a",
            "bfp-direct-c-preconverted-a",
            "q4-direct-c-preconverted-a",
        )
        direct_c_stream = self.transport in (
            "bfp-direct-c-preconverted-a",
            "q4-direct-c-preconverted-a",
        )
        expand_broadcast = (
            self.transport == "q4-expand-broadcast-persistent"
        )
        column_dequant_cache = self.transport == "q4-column-dequant-cache"
        if column_dequant_cache:
            # Expanded BFP is consumed directly from Core stream port zero and
            # owns no L1 ObjectFIFO buffer.  Only the north row owns the Q4
            # tile pointer consumed by the column expander.
            input_bytes = self.q4_tile_bytes if physical_row == 0 else 0
        elif expand_broadcast:
            # Row zero receives cached Q4 through a direct core stream.  It
            # dequantizes one bounded batch and multicasts BFP to rows 1-3;
            # no row materializes a complete expanded tile.
            input_bytes = self.q4_tile_bytes if physical_row == 0 else 0
        else:
            input_bytes = (
                self.bfp_tile_bytes
                if self.weight_flow == "q4-expand-once" and physical_row > 0
                else self.runtime_tile_bytes
            )
        expanded_scratch = 0
        if self.weight_flow == "q4-local":
            expanded_scratch = self.bfp_tile_bytes
        elif expand_broadcast:
            expanded_scratch = (
                self.bfp_tile_bytes if self.n == 32 else 8 * 64 * 9 // 8
            )
        elif column_dequant_cache and physical_row == 0:
            # Eight 8x8 BFP blocks occupy exactly nine 64-byte stream beats.
            # The expander emits one such batch at a time rather than keeping
            # a complete expanded tile in L1.
            expanded_scratch = 8 * 64 * 9 // 8
        elif self.weight_flow == "q4-expand-once" and physical_row == 0:
            expanded_scratch = self.bfp_tile_bytes

        # Lifetimes do not overlap, but basic-sequential currently assigns the
        # activation and weight ObjectFIFO buffers distinct addresses.
        a_fifo_copies = 2 if self.m_a > M_A and physical_row < 3 else 1
        activation_input = (
            self.a_panel_bfp_bytes
            if preconverted_a
            else a_fifo_copies * self.a_chunk_values * 2
        )
        phase_input = activation_input + input_bytes
        c_fifo_tiles = (
            self.c_panel_slab
            if self.c_panel_slab > 1
            else (1 if self.m_a > M_A else 2)
        )
        return {
            "stationary A BFP16": 0 if preconverted_a else self.a_panel_bfp_bytes,
            "A/Q4 input FIFOs": phase_input,
            "expanded B BFP16 scratch": expanded_scratch,
            # The fast m32/n32 receiver retains the complete expanded tile;
            # larger geometries use a bounded 16-block direct-stream batch.
            "streamed B BFP16 batch": (
                (
                    self.bfp_tile_bytes
                    if self.m_a == 32 and self.n == 32
                    else 16 * 64 * 9 // 8
                )
                if column_dequant_cache
                else 0
            ),
            "streamed FP32 partials": (
                self.m_a // 8 * 64 * 4
                if column_dequant_cache and self.m_a == 64
                else 0
            ),
            # The 64-row path trades C ping-pong for allocator headroom.
            "east C staging": (
                c_fifo_tiles * self.m_a * self.n * 2
                if cascade_role == "east" and not direct_c_stream
                else 0
            ),
            "stack": (
                0x800
                if self.m_a > M_A
                or self.transport == "q4-expand-broadcast-persistent"
                or column_dequant_cache
                else CORE_STACK_BYTES
            ),
            "safety margin": CORE_SAFETY_BYTES,
        }

    def core_memory_bytes(
        self, *, cascade_role: str = "middle", physical_row: int = 0
    ) -> int:
        return sum(
            self.core_memory_components(
                cascade_role=cascade_role, physical_row=physical_row
            ).values()
        )

    @property
    def worst_core_memory_bytes(self) -> int:
        return max(
            self.core_memory_bytes(cascade_role=role, physical_row=row)
            for role in ("west", "middle", "east")
            for row in range(N_AIE_ROWS)
        )

    @property
    def memtile_components(self) -> dict[str, int]:
        a_panel_bytes = (
            self.a_panel_bfp_bytes
            if self.transport
            in (
                "bfp-broadcast-preconverted-a",
                "bfp-direct-c-preconverted-a",
                "q4-direct-c-preconverted-a",
            )
            else self.a_panel_values * 2
        )
        if self.transport == "q4-column-dequant-cache":
            # Q4 travels directly from its shim to the north compute tile.
            # Only the once-expanded BFP slab is resident in the MemTile.
            weight_input = self.panel_slab * self.bfp_tile_bytes
            weight_children = 0
        elif self.transport == "shim-stream":
            weight_input = 0
            weight_children = 0
        elif self.transport in (
            "memtile-cache",
            "memtile-fused-cache",
            "memtile-fused-persistent",
            "bfp-memtile-persistent",
            "bfp-memtile-broadcast-persistent",
            "bfp-core-multicast-persistent",
            "q4-memtile-broadcast-persistent",
            "q4-expand-broadcast-persistent",
            "bfp-broadcast-preconverted-a",
            "bfp-direct-c-preconverted-a",
            "q4-direct-c-preconverted-a",
        ):
            # One producer object receives the contiguous KxN Q4 slab.  The
            # asymmetric child owns only a tile-sized L1 buffer; its replaying
            # BD reads the producer allocation in place from the MemTile.
            weight_input = self.panel_slab * self.runtime_tile_bytes
            weight_children = 0
        elif self.transport in (
            "memtile-slab",
            "memtile-slab-direct-a",
            "memtile-fused-stream",
        ):
            # Ping-pong large compressed slabs through MemTile.  The
            # asymmetric tile child lives in L1, so only the two source
            # objects consume MemTile storage.
            weight_input = 2 * self.panel_slab * self.runtime_tile_bytes
            weight_children = 0
        elif self.transport in ("joint-slab", "joint-fused"):
            # One Q4 slab is retained and replayed for two activation waves.
            weight_input = self.panel_slab * self.runtime_tile_bytes
            weight_children = 0
        else:
            weight_input = 2 * self.panel_slab * self.runtime_tile_bytes
            weight_children = (
                N_AIE_ROWS * self.runtime_tile_bytes
                if self.transport == "tile-dma"
                else self.panel_slab * self.runtime_tile_bytes
            )
        a_objects = (
            2 * N_AIE_ROWS
            if self.transport in ("joint-slab", "joint-fused")
            else (
                N_AIE_ROWS
                if self.transport
                in (
                    "bfp-broadcast-preconverted-a",
                    "bfp-direct-c-preconverted-a",
                    "q4-direct-c-preconverted-a",
                )
                else (1 if self.m_a > M_A else N_AIE_ROWS)
            )
        )
        persistent_single_c = (
            self.c_panel_slab == 1
            and self.transport
            in (
                "memtile-fused-persistent",
                "bfp-memtile-persistent",
                "bfp-memtile-broadcast-persistent",
                "bfp-core-multicast-persistent",
                "q4-memtile-broadcast-persistent",
                "q4-expand-broadcast-persistent",
                "q4-column-dequant-cache",
                "bfp-broadcast-preconverted-a",
            )
        )
        return {
            "A source staging": a_objects * a_panel_bytes,
            # The asymmetric A child is allocated in compute-tile L1, not as
            # a second MemTile object.  Confirmed against the placed MLIR
            # addresses for the 8/16/32-panel cache graphs.
            "A stream staging": 0,
            "double-buffered weight input": weight_input,
            "weight row children": weight_children,
            "double-buffered C output": (
                2 * self.c_panel_slab * self.m_a * self.n * 2
                if self.c_panel_slab > 1
                or persistent_single_c
                or self.transport == "q4-direct-c-preconverted-a"
                else 0
            ),
            "safety margin": MEMTILE_SAFETY_BYTES,
        }

    @property
    def memtile_bytes(self) -> int:
        return sum(self.memtile_components.values())

    def validate(self) -> None:
        if self.M <= 0 or self.K <= 0 or self.N <= 0:
            raise ValueError("M, K, and N must be positive")
        if self.n not in (16, 32, 64):
            raise ValueError("the systolic POC supports n=16, n=32, or n=64")
        if self.m_a not in (32, 64):
            raise ValueError("the systolic POC supports m_a=32 or m_a=64")
        m64_preconverted_ceiling = (
            self.transport
            in (
                "bfp-broadcast-preconverted-a",
                "bfp-direct-c-preconverted-a",
            )
            and self.weight_flow == "bfp-prepared"
            and self.n == 32
        ) or (
            self.transport == "q4-direct-c-preconverted-a"
            and self.weight_flow == "q4-direct"
            and self.n == 64
        )
        m64_column_cache = (
            self.transport == "q4-column-dequant-cache"
            and self.weight_flow == "q4-expand-once"
            and self.n == 32
        )
        if self.m_a == 64 and not (
            (self.weight_flow == "q4-direct" and self.n == 32)
            or m64_column_cache
            or m64_preconverted_ceiling
        ):
            raise ValueError(
                "m_a=64 requires q4-direct with n=32, the native Q4 "
                "column cache, or the preconverted BFP ceiling"
            )
        if self.weight_flow not in WEIGHT_FLOWS:
            raise ValueError(f"weight_flow must be one of {WEIGHT_FLOWS}")
        if self.transport not in TRANSPORTS:
            raise ValueError(f"transport must be one of {TRANSPORTS}")
        if self.M % self.m_wave:
            raise ValueError(
                f"M must be divisible by {self.m_wave} (four stationary A rows)"
            )
        if self.K % (N_AIE_COLS * QK_K):
            raise ValueError("K must be divisible by 2048 (one Q4_K block/stage)")
        if self.N % self.n:
            raise ValueError("N must be divisible by n")
        if self.panel_slab not in (1, 2, 4, 8, 16, 32):
            raise ValueError(
                "panel_slab must be one of 1, 2, 4, 8, 16, or 32"
            )
        if self.n_panels % self.panel_slab:
            raise ValueError("N/n must be divisible by panel_slab")
        if self.c_panel_slab not in (1, 2, 4, 8):
            raise ValueError("c_panel_slab must be one of 1, 2, 4, or 8")
        if self.n_panels % self.c_panel_slab:
            raise ValueError("N/n must be divisible by c_panel_slab")
        bfp_persistent = (
            self.transport
            in (
                "bfp-memtile-persistent",
                "bfp-memtile-broadcast-persistent",
                "bfp-core-multicast-persistent",
                "bfp-broadcast-preconverted-a",
                "bfp-direct-c-preconverted-a",
            )
            and self.weight_flow == "bfp-prepared"
        )
        expand_broadcast = (
            self.transport == "q4-expand-broadcast-persistent"
            and self.weight_flow == "q4-expand-once"
        )
        column_dequant_cache = (
            self.transport == "q4-column-dequant-cache"
            and self.weight_flow == "q4-expand-once"
        )
        if self.c_panel_slab > 1 and (
            self.weight_flow != "q4-direct"
            and not bfp_persistent
            and not expand_broadcast
            and not column_dequant_cache
        ):
            raise ValueError(
                "c_panel_slab > 1 requires q4-direct"
            )
        if self.panel_slab > 1 and not (
            bfp_persistent
            or expand_broadcast
            or column_dequant_cache
            or (
                self.weight_flow == "q4-direct"
                and self.transport
                in (
                "auto",
                "core-stream",
                "shim-stream",
                "memtile-cache",
                "memtile-fused-cache",
                "memtile-fused-persistent",
                "q4-memtile-broadcast-persistent",
                "q4-direct-c-preconverted-a",
                "memtile-slab",
                "memtile-slab-direct-a",
                "memtile-fused-stream",
                "joint-slab",
                "joint-fused",
                )
            )
        ):
            raise ValueError(
                "panel_slab > 1 requires q4-direct stream transport"
            )
        if self.transport in (
            "memtile-cache",
            "memtile-fused-cache",
            "memtile-fused-persistent",
            "bfp-memtile-persistent",
            "bfp-memtile-broadcast-persistent",
            "bfp-core-multicast-persistent",
            "q4-memtile-broadcast-persistent",
            "q4-expand-broadcast-persistent",
            "q4-column-dequant-cache",
            "bfp-direct-c-preconverted-a",
            "q4-direct-c-preconverted-a",
            "memtile-slab",
            "memtile-slab-direct-a",
            "memtile-fused-stream",
            "joint-slab",
            "joint-fused",
        ):
            if (
                self.weight_flow != "q4-direct"
                and not bfp_persistent
                and not expand_broadcast
                and not column_dequant_cache
            ) or self.panel_slab == 1:
                raise ValueError(
                    f"{self.transport} requires q4-direct and panel_slab > 1"
                )
            if self.panel_slab % self.c_panel_slab:
                raise ValueError(
                    "c_panel_slab must divide the MemTile weight panel slab"
                )
        if self.transport in (
            "memtile-slab",
            "memtile-slab-direct-a",
            "memtile-fused-stream",
        ):
            if (
                self.transport == "memtile-slab-direct-a"
                and self.m_a != M_A
            ):
                raise ValueError("memtile-slab-direct-a requires m_a=32")
            a_tasks = N_AIE_ROWS if self.m_a > M_A else 1
            # All N slabs are one contiguous host transfer per K-stage/column.
            # Its repeated 40-KiB BD feeds successive ping-pong MemTile
            # objects, so a wave needs only one B task and one C task per shim.
            if a_tasks + 2 > 16:
                raise ValueError(
                    "memtile-slab A/B/C transfer block exceeds 16 shim BDs"
                )
            if self.transport == "memtile-fused-stream" and (
                self.m_a != M_A or self.c_panel_slab == 1
            ):
                raise ValueError(
                    "memtile-fused-stream requires m_a=32 and "
                    "c_panel_slab > 1"
                )
        if self.transport == "memtile-fused-cache" and (
            self.m_a != M_A or self.c_panel_slab == 1
        ):
            raise ValueError(
                "memtile-fused-cache requires m_a=32 and "
                "c_panel_slab > 1"
            )
        if self.transport == "memtile-fused-persistent" and (
            self.m_a != M_A or self.c_panel_slab == 1
        ):
            raise ValueError(
                "memtile-fused-persistent requires m_a=32 and "
                "c_panel_slab > 1"
            )
        if self.transport in (
            "bfp-memtile-persistent",
            "bfp-memtile-broadcast-persistent",
            "bfp-core-multicast-persistent",
        ) and (
            self.weight_flow != "bfp-prepared"
            or self.m_a != M_A
            or self.c_panel_slab == 1
        ):
            raise ValueError(
                f"{self.transport} requires bfp-prepared, m_a=32, and "
                "c_panel_slab > 1"
            )
        if self.transport == "q4-memtile-broadcast-persistent" and (
            self.weight_flow != "q4-direct"
            or self.m_a != M_A
            or self.c_panel_slab == 1
        ):
            raise ValueError(
                "q4-memtile-broadcast-persistent requires q4-direct, "
                "m_a=32, and c_panel_slab > 1"
            )
        if self.transport == "q4-expand-broadcast-persistent" and (
            self.weight_flow != "q4-expand-once"
            or self.m_a != M_A
            or self.n != 32
            or self.panel_slab != 32
            or self.c_panel_slab != 4
        ):
            raise ValueError(
                "q4-expand-broadcast-persistent requires q4-expand-once, "
                "m_a=32, n=32, panel_slab=32, and c_panel_slab=4"
            )
        if self.transport == "q4-column-dequant-cache" and not (
            self.weight_flow == "q4-expand-once"
            and (
                (
                    self.n == 32
                    and self.panel_slab == 16
                    and (
                        (self.m_a == 32 and self.c_panel_slab == 4)
                        or (self.m_a == 64 and self.c_panel_slab == 1)
                    )
                )
                or (
                    self.m_a == 32
                    and self.n == 64
                    and self.panel_slab in (2, 4, 8)
                    and self.c_panel_slab == 2
                )
            )
        ):
            raise ValueError(
                "q4-column-dequant-cache requires q4-expand-once and either "
                "n=32/panel_slab=16 with (m_a=32, c_panel_slab=4) or "
                "(m_a=64, c_panel_slab=1), or m_a=32/n=64 with "
                "panel_slab=2/4/8 and c_panel_slab=2"
            )
        if self.transport == "bfp-broadcast-preconverted-a" and not (
            self.weight_flow == "bfp-prepared"
            and self.panel_slab == 16
            and (
                (self.m_a == M_A and self.c_panel_slab == 8)
                or (self.m_a == 64 and self.c_panel_slab == 1)
            )
        ):
            raise ValueError(
                "bfp-broadcast-preconverted-a requires bfp-prepared, "
                "panel_slab=16, and either (m_a=32, c_panel_slab=8) "
                "or (m_a=64, c_panel_slab=1)"
            )
        if self.transport == "bfp-direct-c-preconverted-a" and not (
            self.weight_flow == "bfp-prepared"
            and self.c_panel_slab == 1
            and (
                (self.m_a == 64 and self.n == 32 and self.panel_slab == 16)
                or (self.m_a == 32 and self.n == 64 and self.panel_slab == 8)
            )
        ):
            raise ValueError(
                "bfp-direct-c-preconverted-a requires bfp-prepared, "
                "c_panel_slab=1, and either (m_a=64, n=32, "
                "panel_slab=16) or (m_a=32, n=64, panel_slab=8)"
            )
        if self.transport == "q4-direct-c-preconverted-a" and not (
            self.weight_flow == "q4-direct"
            and self.m_a == 64
            and self.n == 64
            and self.panel_slab == 16
            and self.c_panel_slab == 1
        ):
            raise ValueError(
                "q4-direct-c-preconverted-a requires q4-direct, m_a=64, "
                "n=64, panel_slab=16, and c_panel_slab=1"
            )
        if self.transport == "joint-slab":
            if (
                self.m_a != M_A
                or self.panel_slab != 16
                or self.c_panel_slab != 8
            ):
                raise ValueError(
                    "joint-slab requires m_a=32, panel_slab=16, and "
                    "c_panel_slab=8"
                )
            if self.M % (2 * self.m_wave):
                raise ValueError("joint-slab requires M divisible by 256")
            if self.weight_cache_slabs + 2 > 16:
                raise ValueError(
                    "joint-slab A/B/C transfer block exceeds 16 shim BDs"
                )
        if self.transport == "joint-fused":
            if (
                self.m_a != M_A
                or self.panel_slab != 16
                or self.c_panel_slab != 8
            ):
                raise ValueError(
                    "joint-fused requires m_a=32, panel_slab=16, and "
                    "c_panel_slab=8"
                )
            if self.M % (2 * self.m_wave):
                raise ValueError("joint-fused requires M divisible by 256")
            if self.weight_cache_slabs % 4:
                raise ValueError(
                    "joint-fused requires the N panel count to contain "
                    "four-slab groups"
                )
        if self.k_stage > 1023:
            raise ValueError("K/8 exceeds the 10-bit DMA dimension limit")
        if self.q4_tile_bytes % 64 or self.bfp_tile_bytes % 64:
            raise ValueError("weight tiles must be an integer number of 64-byte beats")
        if self.weight_flow == "q4-expand-once" and self.transport == "tile-dma":
            raise ValueError("q4-expand-once requires core-stream transport")
        if self.worst_core_memory_bytes > CORE_MEMORY_BYTES:
            raise ValueError(
                f"configuration needs {self.worst_core_memory_bytes} bytes/core, "
                f"exceeding {CORE_MEMORY_BYTES}"
            )
        if self.memtile_bytes > MEMTILE_MEMORY_BYTES:
            raise ValueError(
                f"configuration needs {self.memtile_bytes} bytes/MemTile, "
                f"exceeding {MEMTILE_MEMORY_BYTES}"
            )


def tile_coordinates(config: SystolicConfig) -> Iterator[tuple[int, int]]:
    """Yield prepared tiles as physical K-stage column, then N panel."""

    for stage in range(N_AIE_COLS):
        for panel in range(config.n_panels):
            yield stage, panel


def _pack_q4_tile(
    tile: np.ndarray,
    q: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    k0: int,
    n0: int,
    config: SystolicConfig,
) -> None:
    cursor = 0
    for mk in range(0, config.k_stage, 8):
        for mn in range(0, config.n, 8):
            block = q[k0 + mk : k0 + mk + 8, n0 + mn : n0 + mn + 8]
            tile[cursor : cursor + 32] = (
                block[:, 0::2] | (block[:, 1::2] << 4)
            ).reshape(-1)
            cursor += 32

    scales_bf16 = scales.astype(bfloat16)
    biases_bf16 = biases.astype(bfloat16)
    first_group = k0 // Q4_K_GROUP
    for group in range(config.groups_per_tile):
        for mn in range(0, config.n, 8):
            ns = slice(n0 + mn, n0 + mn + 8)
            tile[cursor : cursor + 16] = scales_bf16[
                first_group + group, ns
            ].view(np.uint8)
            tile[cursor + 16 : cursor + 32] = biases_bf16[
                first_group + group, ns
            ].view(np.uint8)
            cursor += 32
    if cursor != config.q4_tile_bytes:
        raise AssertionError(f"Q4 tile wrote {cursor}, expected {config.q4_tile_bytes}")


def _pack_bfp_tile(
    tile: np.ndarray,
    q: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    *,
    k0: int,
    n0: int,
    config: SystolicConfig,
) -> None:
    groups = np.arange(k0, k0 + config.k_stage) // Q4_K_GROUP
    values = (
        q[k0 : k0 + config.k_stage, n0 : n0 + config.n].astype(np.float32)
        * scales[groups, n0 : n0 + config.n].astype(bfloat16).astype(np.float32)
        - biases[groups, n0 : n0 + config.n]
        .astype(bfloat16)
        .astype(np.float32)
    ).astype(bfloat16)
    cursor = 0
    for mn in range(0, config.n, 8):
        for mk in range(0, config.k_stage, 8):
            # BFP MMUL consumes B in transposed 8x8 block order.
            block = values[mk : mk + 8, mn : mn + 8]
            raw = float_to_bfp16ebs8(block.astype(np.float32).T.reshape(-1))
            tile[cursor : cursor + raw.size] = raw
            cursor += raw.size
    if cursor != config.bfp_tile_bytes:
        raise AssertionError(
            f"BFP tile wrote {cursor}, expected {config.bfp_tile_bytes}"
        )


def prepare_systolic_q4ks_weights(
    native: np.ndarray,
    config: SystolicConfig,
    storage_type: Literal["q4", "bfp16"] | None = None,
) -> np.ndarray:
    """Repack native llama.cpp Q4_K for the eight physical K stages."""

    if not isinstance(native, np.ndarray) or native.dtype != np.dtype(np.uint8):
        raise TypeError("native weights must be a uint8 numpy.ndarray")
    if native.shape != (config.native_bytes,) or not native.flags.c_contiguous:
        raise ValueError(
            f"native weights must be contiguous shape ({config.native_bytes},)"
        )
    selected = storage_type or (
        "bfp16" if config.weight_flow in ("bfp-prepared", "resident") else "q4"
    )
    if selected not in ("q4", "bfp16"):
        raise ValueError("storage_type must be q4 or bfp16")

    q, scales, biases = decode_q4_k(native, K=config.K, N=config.N)
    tile_bytes = config.q4_tile_bytes if selected == "q4" else config.bfp_tile_bytes
    prepared = np.empty(N_AIE_COLS * config.n_panels * tile_bytes, np.uint8)
    for tile_index, (stage, panel) in enumerate(tile_coordinates(config)):
        tile = prepared[tile_index * tile_bytes : (tile_index + 1) * tile_bytes]
        kwargs = dict(
            k0=stage * config.k_stage,
            n0=panel * config.n,
            config=config,
        )
        if selected == "q4":
            _pack_q4_tile(tile, q, scales, biases, **kwargs)
        else:
            _pack_bfp_tile(tile, q, scales, biases, **kwargs)
    return prepared


def prepare_systolic_bfp_activations(
    activations: np.ndarray, config: SystolicConfig
) -> np.ndarray:
    """Pack BF16 activations into the native stationary-A BFP microtile ABI.

    This helper is for the explicitly ineligible preconverted-input ceiling.
    The order is K-stage column, M wave, physical row, 8x8 row-major block.
    """

    if not isinstance(activations, np.ndarray):
        raise TypeError("activations must be a numpy.ndarray")
    if activations.shape != (config.M, config.K):
        raise ValueError(
            f"activations must have shape ({config.M}, {config.K})"
        )
    if activations.dtype != np.dtype(bfloat16):
        raise TypeError("activations must use bfloat16")
    if not activations.flags.c_contiguous:
        raise ValueError("activations must be contiguous")

    waves = config.M // config.m_wave
    result = np.empty(
        N_AIE_COLS * waves * N_AIE_ROWS * config.a_panel_bfp_bytes,
        dtype=np.uint8,
    )
    cursor = 0
    for stage in range(N_AIE_COLS):
        k0 = stage * config.k_stage
        for wave in range(waves):
            wave_m0 = wave * config.m_wave
            for physical_row in range(N_AIE_ROWS):
                m0 = wave_m0 + physical_row * config.m_a
                for rm in range(0, config.m_a, 8):
                    for rk in range(0, config.k_stage, 8):
                        block = activations[
                            m0 + rm : m0 + rm + 8,
                            k0 + rk : k0 + rk + 8,
                        ]
                        raw = float_to_bfp16ebs8(
                            block.astype(np.float32).reshape(-1)
                        )
                        result[cursor : cursor + raw.size] = raw
                        cursor += raw.size
    if cursor != result.size:
        raise AssertionError("prepared activation cursor mismatch")
    return result


def unpack_systolic_q4(
    prepared: np.ndarray, config: SystolicConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = N_AIE_COLS * config.n_panels * config.q4_tile_bytes
    if not isinstance(prepared, np.ndarray) or prepared.dtype != np.dtype(np.uint8):
        raise TypeError("prepared weights must be uint8")
    if prepared.shape != (expected,):
        raise ValueError(f"prepared Q4 weights must have shape ({expected},)")
    q = np.empty((config.K, config.N), np.uint8)
    scales = np.empty((config.K // Q4_K_GROUP, config.N), bfloat16)
    biases = np.empty_like(scales)
    for tile_index, (stage, panel) in enumerate(tile_coordinates(config)):
        tile = prepared[
            tile_index * config.q4_tile_bytes : (tile_index + 1) * config.q4_tile_bytes
        ]
        k0, n0 = stage * config.k_stage, panel * config.n
        cursor = 0
        for mk in range(0, config.k_stage, 8):
            for mn in range(0, config.n, 8):
                nibble = tile[cursor : cursor + 32].reshape(8, 4)
                block = np.empty((8, 8), np.uint8)
                block[:, 0::2] = nibble & 0x0F
                block[:, 1::2] = nibble >> 4
                q[k0 + mk : k0 + mk + 8, n0 + mn : n0 + mn + 8] = block
                cursor += 32
        first_group = k0 // Q4_K_GROUP
        for group in range(config.groups_per_tile):
            for mn in range(0, config.n, 8):
                ns = slice(n0 + mn, n0 + mn + 8)
                scales[first_group + group, ns] = (
                    tile[cursor : cursor + 16].copy().view(bfloat16)
                )
                biases[first_group + group, ns] = (
                    tile[cursor + 16 : cursor + 32].copy().view(bfloat16)
                )
                cursor += 32
        if cursor != config.q4_tile_bytes:
            raise AssertionError("prepared Q4 cursor mismatch")
    return q, scales, biases


def decode_prepared_bfp(prepared: np.ndarray, config: SystolicConfig) -> np.ndarray:
    expected = N_AIE_COLS * config.n_panels * config.bfp_tile_bytes
    if prepared.shape != (expected,) or prepared.dtype != np.dtype(np.uint8):
        raise ValueError(f"prepared BFP weights must be uint8 shape ({expected},)")
    result = np.empty((config.K, config.N), np.float32)
    for tile_index, (stage, panel) in enumerate(tile_coordinates(config)):
        tile = prepared[
            tile_index * config.bfp_tile_bytes : (tile_index + 1) * config.bfp_tile_bytes
        ]
        cursor = 0
        k0, n0 = stage * config.k_stage, panel * config.n
        for mn in range(0, config.n, 8):
            for mk in range(0, config.k_stage, 8):
                raw = tile[cursor : cursor + 72]
                result[k0 + mk : k0 + mk + 8, n0 + mn : n0 + mn + 8] = (
                    bfp16ebs8_to_float(raw).reshape(8, 8).T
                )
                cursor += 72
    return result


def logical_tap_indices(config: SystolicConfig) -> dict[str, list[int]]:
    """Model the complete runtime order for host-only coverage tests."""

    a: list[int] = []
    b: list[int] = []
    c: list[int] = []

    def append_a(wave: int) -> None:
        for stage in range(N_AIE_COLS):
            for row in range(N_AIE_ROWS):
                m0 = wave * config.m_wave + row * config.m_a
                k0 = stage * config.k_stage
                for m in range(config.m_a):
                    a.extend(
                        range(
                            (m0 + m) * config.K + k0,
                            (m0 + m) * config.K + k0 + config.k_stage,
                        )
                    )

    def append_c(wave: int, panel: int) -> None:
        for row in range(N_AIE_ROWS):
            m0 = wave * config.m_wave + row * config.m_a
            n0 = panel * config.n
            for m in range(config.m_a):
                c.extend(
                    range(
                        (m0 + m) * config.N + n0,
                        (m0 + m) * config.N + n0 + config.n,
                    )
                )

    waves = config.M // config.m_wave
    if config.transport == "memtile-cache":
        replay = config.weight_replay_waves
        for cache_base in range(0, config.n_panels, config.panel_slab):
            for wave_base in range(0, waves, replay):
                for stage in range(N_AIE_COLS):
                    first_tile = stage * config.n_panels + cache_base
                    b.extend(
                        range(
                            first_tile * config.runtime_tile_bytes,
                            (first_tile + config.panel_slab)
                            * config.runtime_tile_bytes,
                        )
                    )
                for wave in range(wave_base, wave_base + replay):
                    append_a(wave)
                    for panel in range(
                        cache_base, cache_base + config.panel_slab
                    ):
                        append_c(wave, panel)
        return {"A": a, "B": b, "C": c}

    if config.transport in ("memtile-slab", "memtile-slab-direct-a"):
        for wave in range(waves):
            append_a(wave)
            for slab_base in range(
                0, config.n_panels, config.panel_slab
            ):
                for stage in range(N_AIE_COLS):
                    first_tile = stage * config.n_panels + slab_base
                    b.extend(
                        range(
                            first_tile * config.runtime_tile_bytes,
                            (first_tile + config.panel_slab)
                            * config.runtime_tile_bytes,
                        )
                    )
                for panel in range(
                    slab_base, slab_base + config.panel_slab
                ):
                    append_c(wave, panel)
        return {"A": a, "B": b, "C": c}

    if config.transport == "joint-slab":
        for wave_base in range(0, waves, 2):
            append_a(wave_base)
            append_a(wave_base + 1)
            for slab_base in range(
                0, config.n_panels, config.panel_slab
            ):
                for stage in range(N_AIE_COLS):
                    first_tile = stage * config.n_panels + slab_base
                    b.extend(
                        range(
                            first_tile * config.runtime_tile_bytes,
                            (first_tile + config.panel_slab)
                            * config.runtime_tile_bytes,
                        )
                    )
                for wave in (wave_base, wave_base + 1):
                    for panel in range(
                        slab_base, slab_base + config.panel_slab
                    ):
                        append_c(wave, panel)
        return {"A": a, "B": b, "C": c}

    for wave in range(waves):
        append_a(wave)
        for panel in range(config.n_panels):
            for stage in range(N_AIE_COLS):
                tile = stage * config.n_panels + panel
                b.extend(
                    range(
                        tile * config.runtime_tile_bytes,
                        (tile + 1) * config.runtime_tile_bytes,
                    )
                )
            append_c(wave, panel)
    return {"A": a, "B": b, "C": c}


def _wave_panel_order(config: SystolicConfig) -> list[tuple[int, int]]:
    if config.weight_flow == "resident":
        return [(0, 0)]
    waves = config.M // config.m_wave
    if config.transport == "memtile-cache":
        return [
            (wave, panel)
            for cache_base in range(0, config.n_panels, config.panel_slab)
            for wave in range(waves)
            for panel in range(cache_base, cache_base + config.panel_slab)
        ]
    return [
        (wave, panel)
        for wave in range(waves)
        for panel in range(config.n_panels)
    ]


def weight_forward_sequence(
    config: SystolicConfig,
) -> list[tuple[int, int, int, int]]:
    """Return (wave, panel, K-stage, physical-row) consumption order."""

    return [
        (wave, panel, stage, row)
        for wave, panel in _wave_panel_order(config)
        for stage in range(N_AIE_COLS)
        for row in range(N_AIE_ROWS)
    ]


def cascade_token_sequence(
    config: SystolicConfig,
) -> list[tuple[int, int, int, int, int, int]]:
    """Return east-edge token order and its token-major local C offset.

    Tuples contain wave, panel, physical row, row block, N block, and offset.
    Every token crosses the eight horizontal cascade stages before appearing
    in this sequence.
    """

    n_blocks = config.n // 8
    result = []
    for wave, panel in _wave_panel_order(config):
        for row in range(N_AIE_ROWS):
            for row_block in range(config.m_a // 8):
                for n_block in range(n_blocks):
                    macro_token = (
                        (row_block // 2) * (n_blocks // 2) + n_block // 2
                    )
                    offset = (
                        macro_token * 256
                        + (row_block % 2) * 128
                        + (n_block % 2) * 8
                    )
                    result.append(
                        (
                            wave,
                            panel,
                            row,
                            row_block,
                            n_block,
                            offset,
                        )
                    )
    return result


def cascade_c_indices(config: SystolicConfig) -> list[int]:
    """Reconstruct logical row-major C indices from east-edge tokens."""

    result: list[int] = []
    for wave, panel, row, row_block, n_block, _ in cascade_token_sequence(
        config
    ):
        m0 = wave * config.m_wave + row * config.m_a + row_block * 8
        n0 = panel * config.n + n_block * 8
        for lane_row in range(8):
            result.extend(
                range(
                    (m0 + lane_row) * config.N + n0,
                    (m0 + lane_row) * config.N + n0 + 8,
                )
            )
    return result


def phase_lifetimes() -> dict[str, tuple[int, int]]:
    """Abstract worker phases used to validate the input-buffer overlay."""

    return {
        "activation input": (0, 1),
        "stationary activation": (0, 4),
        "packed weight input": (1, 3),
        "expanded weight": (1, 3),
        "output": (2, 4),
    }
