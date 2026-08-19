# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Native llama.cpp Q4_K decoding and AIE2P preparation helpers.

``Q4_K_S`` is a model quantization policy.  Matrix tensors selected as Q4_K
use llama.cpp's 144-byte ``block_q4_K`` ABI: two FP16 super-block factors,
twelve bytes containing eight 6-bit scales and eight 6-bit minima, and 128
nibble-packed bytes for 256 weights.

The compute-tile format is deliberately regular.  Tiles are ordered by AIE
column, round-robin N tile, then K tile.  A compressed tile contains 8x8
microtiled nibbles followed by interleaved eight-column BF16 effective-scale
and effective-bias vectors for every 32-K group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Literal

import numpy as np
from ml_dtypes import bfloat16


QK_K = 256
Q4_K_BLOCK_BYTES = 144
Q4_K_SCALE_BYTES = 12
Q4_K_GROUP = 32
N_AIE_ROWS = 4
CASCADE_CHUNK_K = 256
CORE_MEMORY_BYTES = 64 * 1024
MEMTILE_MEMORY_BYTES = 512 * 1024
CORE_STACK_BYTES = 0xD00
CORE_SAFETY_BYTES = 4 * 1024
MEMTILE_SAFETY_BYTES = 32 * 1024
COMPUTE_TYPES = ("bf16", "bfp16", "int8")
ACCUMULATION_MODES = (
    "bf16",
    "fp32",
    "cascade",
    "cascade-resident",
    "cascade-register",
    "cascade-chunked",
    "cascade-shared",
    "cascade-hybrid",
)
CACHE_MODES = (
    "stream",
    "l1-weight",
    "memtile-weight",
    "memtile-activation",
    "joint-slab",
)
ACTIVATION_INPUTS = ("bf16", "bfp16", "int8")
STORAGE_TYPES = ("q4", "bf16", "bfp16", "int8")
SUPPORTED_COLUMNS = (1, 2, 4, 8)


def ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require_array(
    name: str, value: np.ndarray, shape: tuple[int, ...], dtype: object
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    expected = np.dtype(dtype)
    if value.dtype != expected:
        raise TypeError(f"{name} must have dtype {expected}, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


@dataclass(frozen=True)
class Q4KSConfig:
    """Compile-time matrix, tile, compute, and cache configuration."""

    M: int = 1024
    K: int = 1024
    N: int = 2048
    m_c: int = 64
    m_a: int = 32
    k: int = 128
    n: int = 64
    n_aie_cols: int = 8
    compute_type: str = "bf16"
    accumulation_mode: str = "bf16"
    cache_mode: str = "stream"
    activation_input: str = "bf16"
    cache_k: int | None = None

    def __post_init__(self) -> None:
        if self.cache_k is None:
            object.__setattr__(self, "cache_k", self.K)
        self.validate()

    @property
    def a_subtiles(self) -> int:
        return self.m_c // self.m_a

    @property
    def groups_per_tile(self) -> int:
        return self.k // Q4_K_GROUP

    @property
    def weights_bytes_per_tile(self) -> int:
        return self.k * self.n // 2

    @property
    def metadata_bytes_per_tile(self) -> int:
        return self.groups_per_tile * self.n * 4

    @property
    def raw_tile_bytes(self) -> int:
        return self.weights_bytes_per_tile + self.metadata_bytes_per_tile

    @property
    def packed_rows(self) -> int:
        return ceildiv(self.raw_tile_bytes, self.k)

    @property
    def tile_bytes(self) -> int:
        return self.packed_rows * self.k

    @property
    def tile_padding_bytes(self) -> int:
        return self.tile_bytes - self.raw_tile_bytes

    @property
    def weight_storage_type(self) -> str:
        """Prepared-weight representation consumed by the selected design."""

        return (
            "bfp16"
            if self.accumulation_mode
            in (
                "cascade-resident",
                "cascade-register",
                "cascade-chunked",
            )
            else "q4"
        )

    @property
    def runtime_tile_bytes(self) -> int:
        return self.prepared_tile_bytes(self.weight_storage_type)

    @property
    def runtime_packed_rows(self) -> int:
        return ceildiv(self.runtime_tile_bytes, self.k)

    @property
    def n_k_tiles(self) -> int:
        return self.K // self.k

    @property
    def n_n_tiles(self) -> int:
        return self.N // self.n

    @property
    def n_tiles(self) -> int:
        return self.n_k_tiles * self.n_n_tiles

    @property
    def native_bytes(self) -> int:
        return self.N * (self.K // QK_K) * Q4_K_BLOCK_BYTES

    @property
    def prepared_bytes(self) -> int:
        return self.n_tiles * self.runtime_tile_bytes

    @property
    def q4_prepared_bytes(self) -> int:
        return self.n_tiles * self.tile_bytes

    @property
    def c_fifo_depth(self) -> int:
        return 1 if self.m_c >= 128 else 2

    @property
    def a_fifo_depth(self) -> int:
        # The cascade pipeline benefits from two resident A tiles and its
        # validated 64x64 tile still fits with that depth.
        if self.accumulation_mode == "cascade":
            return 2
        if self.accumulation_mode in (
            "cascade-resident",
            "cascade-register",
            "cascade-chunked",
            "cascade-shared",
            "cascade-hybrid",
        ):
            return 1
        return 1 if self.m_a >= 64 else 2


    def prepared_tile_bytes(self, storage_type: str) -> int:
        if storage_type == "q4":
            return self.tile_bytes
        if storage_type == "bf16":
            return self.k * self.n * 2
        if storage_type == "bfp16":
            return self.k * self.n * 9 // 8
        if storage_type == "int8":
            return self.k * self.n + self.metadata_bytes_per_tile
        raise ValueError(f"storage_type must be one of {STORAGE_TYPES}")

    def panel_bytes(self, storage_type: str, cache_k: int | None = None) -> int:
        panel_k = self.cache_k if cache_k is None else cache_k
        assert panel_k is not None
        return (panel_k // self.k) * self.prepared_tile_bytes(storage_type)

    @property
    def core_memory_components(self) -> dict[str, int]:
        if self.accumulation_mode == "cascade-hybrid":
            return {
                "A 32-row K-tile FIFO": self.m_a * self.k * 2,
                "compressed Q4_K tile FIFO": self.tile_bytes,
                "BFP16 weight scratch": self.k * self.n * 9 // 8,
                "BF16 local K/2 partial": self.m_c * self.n * 2,
                "stack": CORE_STACK_BYTES,
                "safety margin": CORE_SAFETY_BYTES,
            }

        if self.accumulation_mode == "cascade-shared":
            return {
                "A 16-row K-tile FIFO": self.m_a * self.k * 2,
                "compressed Q4_K tile FIFO": self.tile_bytes,
                "BFP16 weight scratch": self.k * self.n * 9 // 8,
                "neighbor-split FP32 accumulation scratch": (
                    self.m_c * self.n * 2
                ),
                "stack": CORE_STACK_BYTES,
                "safety margin": CORE_SAFETY_BYTES,
            }

        if self.accumulation_mode == "cascade-chunked":
            return {
                "A 16-row chunk FIFO": 16 * CASCADE_CHUNK_K * 2,
                "B full-N chunk FIFO": CASCADE_CHUNK_K * self.n * 9 // 8,
                "FP32 accumulation scratch": self.m_c * self.n * 4,
                "stack": CORE_STACK_BYTES,
                "safety margin": CORE_SAFETY_BYTES,
            }

        if self.accumulation_mode == "cascade-register":
            shard_k = self.K // N_AIE_ROWS
            return {
                "A 16-row K-shard FIFO": 16 * shard_k * 2,
                "B 16-column K-shard FIFO": shard_k * 16 * 9 // 8,
                "stack": CORE_STACK_BYTES,
                "safety margin": CORE_SAFETY_BYTES,
            }

        weight_scratch = {
            "bf16": self.k * self.n * 2,
            "bfp16": self.k * self.n * 9 // 8,
            "int8": self.k * self.n,
        }[self.compute_type]
        activation_scratch = 0
        if self.accumulation_mode == "cascade-resident":
            # The input FIFO already contains native AIE BFP16 blocks.
            weight_scratch = 0
        if self.compute_type == "int8":
            activation_scratch = (
                self.m_a * self.k
                + self.m_a * self.groups_per_tile * (2 + 4)
            )
        components = {
            f"A FIFO (depth {self.a_fifo_depth})": (
                self.a_fifo_depth * self.m_a * self.k * 2
            ),
            "prepared-B FIFO (depth 1)": self.runtime_tile_bytes,
            f"C FIFO (depth {self.c_fifo_depth})": (
                self.c_fifo_depth * self.m_c * self.n * 2
            ),
            f"{self.compute_type} weight scratch": weight_scratch,
            "activation conversion scratch": activation_scratch,
            "stack": CORE_STACK_BYTES,
            "safety margin": CORE_SAFETY_BYTES,
        }
        if self.accumulation_mode in ("fp32", "cascade", "cascade-resident"):
            components["FP32 accumulation scratch"] = self.m_c * self.n * 4
        if self.accumulation_mode == "cascade-resident":
            # The result ObjectFIFO is allocated in the adjacent MemTile.
            components[f"C FIFO (depth {self.c_fifo_depth})"] = 0
        return components

    @property
    def core_memory_bytes(self) -> int:
        return sum(self.core_memory_components.values())

    def activation_panel_bytes(self, storage_type: str) -> int:
        assert self.cache_k is not None
        values = self.m_c * self.cache_k
        if storage_type == "bfp16":
            return values * 9 // 8
        if storage_type == "int8":
            groups = self.cache_k // Q4_K_GROUP
            return values + self.m_c * groups * (2 + 4)
        if storage_type == "bf16":
            return values * 2
        raise ValueError("activation cache storage must be bf16, bfp16, or int8")

    def memtile_components(self, storage_type: str | None = None) -> dict[str, int]:
        if self.accumulation_mode in ("cascade-shared", "cascade-hybrid"):
            return {
                "resident compressed-Q4_K panel": (
                    self.panel_bytes("q4")
                    if self.cache_mode == "memtile-weight"
                    else 0
                ),
                "streamed compressed-Q4_K panel": (
                    self.panel_bytes("q4")
                    if self.cache_mode == "l1-weight"
                    else 0
                ),
                "double-buffered activation K tile": (
                    2 * self.m_c * self.k * 2
                ),
                "C staging": (
                    self.m_c * self.n * 2
                    if self.accumulation_mode == "cascade-hybrid"
                    else 0
                ),
                "safety margin": MEMTILE_SAFETY_BYTES,
            }

        if self.accumulation_mode == "cascade-chunked":
            return {
                "resident expanded-B panel": self.K * self.n * 9 // 8,
                "double-buffered activation chunk": (
                    2 * self.m_c * CASCADE_CHUNK_K * 2
                ),
                "C staging": 0,
                "safety margin": MEMTILE_SAFETY_BYTES,
            }

        if self.accumulation_mode == "cascade-register":
            shard_k = self.K // N_AIE_ROWS
            return {
                "resident expanded-B panel": self.K * self.n * 9 // 8,
                "resident activation K-shard": self.m_c * shard_k * 2,
                "C staging": 0,
                "safety margin": MEMTILE_SAFETY_BYTES,
            }

        storage = storage_type or (
            self.weight_storage_type
            if self.cache_mode == "memtile-weight"
            else (
                "bfp16"
                if self.compute_type == "bfp16"
                else "int8"
                if self.compute_type == "int8"
                else "q4"
            )
        )
        weight = 0
        activation = 0
        if self.cache_mode in ("memtile-weight", "joint-slab"):
            weight = self.panel_bytes(storage)
        if self.cache_mode in ("memtile-activation", "joint-slab"):
            activation_storage = (
                storage if storage in ("bfp16", "int8") else "bfp16"
            )
            activation = self.activation_panel_bytes(activation_storage)
        return {
            "resident weight panel": weight,
            "resident activation panel": activation,
            "C join staging": N_AIE_ROWS * self.m_c * self.n * 2,
            "stream staging": 2
            * (self.m_a * self.k * 2 + self.runtime_tile_bytes),
            "safety margin": MEMTILE_SAFETY_BYTES,
        }

    def validate(self) -> None:
        for name in ("M", "K", "N", "m_c", "m_a", "k", "n"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_aie_cols not in SUPPORTED_COLUMNS:
            raise ValueError(f"n_aie_cols must be one of {SUPPORTED_COLUMNS}")
        if self.compute_type not in COMPUTE_TYPES:
            raise ValueError(f"compute_type must be one of {COMPUTE_TYPES}")
        if self.accumulation_mode not in ACCUMULATION_MODES:
            raise ValueError(
                f"accumulation_mode must be one of {ACCUMULATION_MODES}"
            )
        if self.cache_mode not in CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {CACHE_MODES}")
        if self.activation_input not in ACTIVATION_INPUTS:
            raise ValueError(f"activation_input must be one of {ACTIVATION_INPUTS}")
        if self.K % QK_K:
            raise ValueError("K must be divisible by the native Q4_K block size 256")
        if self.k % Q4_K_GROUP:
            raise ValueError("k must be divisible by the Q4_K 32-value group")
        if self.K % self.k:
            raise ValueError("k must divide K")
        if self.m_c % self.m_a:
            raise ValueError("m_a must divide m_c")
        if self.accumulation_mode != "bf16" and self.compute_type != "bfp16":
            raise ValueError("FP32 and cascade accumulation require bfp16 compute")
        if self.accumulation_mode == "cascade":
            if self.m_a != self.m_c:
                raise ValueError("cascade accumulation requires m_a == m_c")
            if self.k != 64:
                raise ValueError(
                    "cascade accumulation currently requires k == 64; "
                    "larger specializations exceed AIE program memory"
                )
            if self.K % (N_AIE_ROWS * self.k):
                raise ValueError("cascade accumulation requires 4*k to divide K")
            if self.cache_mode not in ("stream", "l1-weight"):
                raise ValueError(
                    "cascade accumulation currently supports stream or l1-weight"
                )
        if self.m_a % 16:
            raise ValueError("m_a must be divisible by 16")
        if self.accumulation_mode == "cascade-resident":
            if self.m_a != self.m_c:
                raise ValueError(
                    "cascade-resident accumulation requires m_a == m_c"
                )
            if self.K % (N_AIE_ROWS * self.k):
                raise ValueError(
                    "cascade-resident accumulation requires 4*k to divide K"
                )
            if self.cache_mode != "memtile-weight":
                raise ValueError(
                    "cascade-resident accumulation requires memtile-weight"
                )
        if self.accumulation_mode == "cascade-register":
            if self.m_a != 16:
                raise ValueError(
                    "cascade-register accumulation requires m_a == 16"
                )
            if (self.K // N_AIE_ROWS) % 8:
                raise ValueError(
                    "cascade-register requires each K/4 shard to be divisible by 8"
                )
            if self.cache_mode != "memtile-weight":
                raise ValueError(
                    "cascade-register accumulation requires memtile-weight"
                )
        if self.accumulation_mode == "cascade-chunked":
            if self.m_a != 16:
                raise ValueError(
                    "cascade-chunked accumulation requires m_a == 16"
                )
            if self.K % (N_AIE_ROWS * CASCADE_CHUNK_K):
                raise ValueError(
                    "cascade-chunked requires 4*256 to divide K"
                )
            if self.cache_mode != "memtile-weight":
                raise ValueError(
                    "cascade-chunked accumulation requires memtile-weight"
                )
        if self.accumulation_mode == "cascade-shared":
            if (self.m_c, self.m_a, self.k, self.n) != (128, 16, 64, 128):
                raise ValueError(
                    "cascade-shared requires m_c=128, m_a=16, k=64, n=128"
                )
            if self.K % (N_AIE_ROWS * self.k):
                raise ValueError(
                    "cascade-shared requires 4*k to divide K"
                )
            if self.cache_mode != "memtile-weight":
                raise ValueError(
                    "cascade-shared accumulation requires memtile-weight"
                )
            if self.m_c * self.n * 2 > CORE_MEMORY_BYTES // 2:
                raise ValueError(
                    "each shared FP32 accumulator half must fit in 32 KiB"
                )
        if self.accumulation_mode == "cascade-hybrid":
            hybrid_tiles = {
                (128, 32, 64, 128),
                (128, 64, 128, 64),
                (256, 32, 64, 64),
                (256, 64, 64, 64),
                (256, 32, 128, 64),
                (512, 32, 64, 32),
                (512, 64, 64, 32),
            }
            if (self.m_c, self.m_a, self.k, self.n) not in hybrid_tiles:
                raise ValueError(
                    "cascade-hybrid requires one of: "
                    "(m_c,m_a,k,n)=(128,32,64,128), "
                    "(128,64,128,64), (256,32|64,64,64), "
                    "(256,32,128,64), or (512,32|64,64,32)"
                )
            if self.K % (2 * self.k):
                raise ValueError("cascade-hybrid requires 2*k to divide K")
            if self.cache_mode not in ("l1-weight", "memtile-weight"):
                raise ValueError(
                    "cascade-hybrid accumulation requires l1-weight or "
                    "memtile-weight"
                )
        if self.n % 16:
            raise ValueError("n must be divisible by 16")
        if self.M % (self.m_c * N_AIE_ROWS):
            raise ValueError("M must be divisible by m_c * 4 AIE rows")
        if self.N % (self.n * self.n_aie_cols):
            raise ValueError("N must be divisible by n * n_aie_cols")
        if (self.M // (self.m_c * N_AIE_ROWS)) % 2:
            raise ValueError("M / (m_c * 4) must be even for ping-pong scheduling")
        assert self.cache_k is not None
        if self.cache_k % QK_K or self.K % self.cache_k:
            raise ValueError("cache_k must be a 256-aligned divisor of K")

        dma_sizes = {
            "K/k": self.n_k_tiles,
            "N/(n*cols)": self.n_n_tiles // self.n_aie_cols,
            "packed_rows": self.packed_rows,
            "k": self.k,
            "m_a": self.m_a,
            "m_c": self.m_c,
            "n": self.n,
        }
        if self.accumulation_mode == "cascade":
            dma_sizes["4*packed_rows"] = N_AIE_ROWS * self.packed_rows
        if self.accumulation_mode == "cascade-resident":
            dma_sizes["4*runtime_packed_rows"] = (
                N_AIE_ROWS * self.runtime_packed_rows
            )
        if self.accumulation_mode == "cascade-register":
            dma_sizes.update(
                {
                    "K/32": self.K // 32,
                    "m_c/16": self.m_c // 16,
                    "n/16": self.n // 16,
                }
            )
        if self.accumulation_mode == "cascade-chunked":
            dma_sizes.update(
                {
                    "K/1024": self.K // (N_AIE_ROWS * CASCADE_CHUNK_K),
                    "256/8": CASCADE_CHUNK_K // 8,
                }
            )
        if self.accumulation_mode == "cascade-shared":
            dma_sizes.update(
                {
                    "K/(4*k)": self.K // (N_AIE_ROWS * self.k),
                    "packed tile rows": self.packed_rows,
                }
            )
        if self.accumulation_mode == "cascade-hybrid":
            dma_sizes.update(
                {
                    "K/(2*k)": self.K // (2 * self.k),
                    "packed tile rows": self.packed_rows,
                }
            )
        bad_dma = {k: v for k, v in dma_sizes.items() if v > 1023}
        if bad_dma:
            details = ", ".join(f"{k}={v}" for k, v in bad_dma.items())
            raise ValueError(f"DMA size exceeds the 10-bit limit: {details}")

        if self.core_memory_bytes > CORE_MEMORY_BYTES:
            details = ", ".join(
                f"{k}={v}" for k, v in self.core_memory_components.items()
            )
            raise ValueError(
                f"configuration needs {self.core_memory_bytes} bytes/core, "
                f"exceeding {CORE_MEMORY_BYTES}: {details}"
            )

        if self.cache_mode != "stream" and self.cache_mode != "l1-weight":
            mem = self.memtile_components()
            if sum(mem.values()) > MEMTILE_MEMORY_BYTES:
                details = ", ".join(f"{k}={v}" for k, v in mem.items())
                raise ValueError(
                    f"configuration needs {sum(mem.values())} bytes/MemTile, "
                    f"exceeding {MEMTILE_MEMORY_BYTES}: {details}"
                )


@dataclass(frozen=True)
class TapSpec:
    tensor: str
    offset: int
    sizes: tuple[int, int, int, int]
    strides: tuple[int, int, int, int]
    column: int
    row_block: int | None = None

    @property
    def volume(self) -> int:
        return int(np.prod(self.sizes))

    def indices(self) -> Iterator[int]:
        for i0 in range(self.sizes[0]):
            for i1 in range(self.sizes[1]):
                for i2 in range(self.sizes[2]):
                    for i3 in range(self.sizes[3]):
                        yield (
                            self.offset
                            + i0 * self.strides[0]
                            + i1 * self.strides[1]
                            + i2 * self.strides[2]
                            + i3 * self.strides[3]
                        )


def unpack_scale_min_k4(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode llama.cpp's eight 6-bit scale/min codes."""

    packed = _require_array("packed scales", packed, (Q4_K_SCALE_BYTES,), np.uint8)
    scales = np.empty(8, dtype=np.uint8)
    mins = np.empty(8, dtype=np.uint8)
    for j in range(8):
        if j < 4:
            scales[j] = packed[j] & 0x3F
            mins[j] = packed[j + 4] & 0x3F
        else:
            scales[j] = (packed[j + 4] & 0x0F) | ((packed[j - 4] >> 6) << 4)
            mins[j] = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4)
    return scales, mins


def pack_scale_min_k4(scales: np.ndarray, mins: np.ndarray) -> np.ndarray:
    scales = _require_array("scales", scales, (8,), np.uint8)
    mins = _require_array("mins", mins, (8,), np.uint8)
    if np.any(scales > 63) or np.any(mins > 63):
        raise ValueError("Q4_K scale and min codes must be in 0..63")
    packed = np.zeros(Q4_K_SCALE_BYTES, dtype=np.uint8)
    for j in range(4):
        packed[j] = scales[j]
        packed[j + 4] = mins[j]
    for j in range(4, 8):
        packed[j + 4] = (scales[j] & 0x0F) | ((mins[j] & 0x0F) << 4)
        packed[j - 4] |= (scales[j] >> 4) << 6
        packed[j] |= (mins[j] >> 4) << 6
    return packed


def _fp16_from_le(two_bytes: np.ndarray) -> np.float32:
    return np.frombuffer(two_bytes.tobytes(), dtype="<f2", count=1)[0].astype(np.float32)


def decode_q4_k(
    native: np.ndarray, *, K: int, N: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode native blocks into q values and effective float scale/bias."""

    if K <= 0 or N <= 0 or K % QK_K:
        raise ValueError("K and N must be positive and K must be divisible by 256")
    expected = N * (K // QK_K) * Q4_K_BLOCK_BYTES
    native = _require_array("native", native, (expected,), np.uint8)
    q = np.empty((K, N), dtype=np.uint8)
    scales = np.empty((K // Q4_K_GROUP, N), dtype=np.float32)
    biases = np.empty_like(scales)
    blocks_per_row = K // QK_K
    for col in range(N):
        for block_index in range(blocks_per_row):
            offset = (col * blocks_per_row + block_index) * Q4_K_BLOCK_BYTES
            block = native[offset : offset + Q4_K_BLOCK_BYTES]
            d = _fp16_from_le(block[0:2])
            dmin = _fp16_from_le(block[2:4])
            if not np.isfinite(d) or not np.isfinite(dmin):
                raise ValueError("Q4_K d and dmin must be finite FP16 values")
            scale_codes, min_codes = unpack_scale_min_k4(block[4:16])
            group0 = block_index * 8
            scales[group0 : group0 + 8, col] = d * scale_codes.astype(np.float32)
            biases[group0 : group0 + 8, col] = dmin * min_codes.astype(np.float32)
            qs = block[16:144]
            k0 = block_index * QK_K
            for region in range(4):
                nibble = qs[region * 32 : (region + 1) * 32]
                q[k0 + region * 64 : k0 + region * 64 + 32, col] = nibble & 0x0F
                q[k0 + region * 64 + 32 : k0 + region * 64 + 64, col] = nibble >> 4
    return q, scales, biases


def encode_q4_k(
    q: np.ndarray, scales: np.ndarray, mins: np.ndarray, d: np.ndarray, dmin: np.ndarray
) -> np.ndarray:
    """Build native blocks from explicit codes, primarily for deterministic tests."""

    if not isinstance(q, np.ndarray) or q.ndim != 2:
        raise TypeError("q must be a rank-2 numpy.ndarray")
    K, N = q.shape
    q = _require_array("q", q, (K, N), np.uint8)
    if K % QK_K:
        raise ValueError("K must be divisible by 256")
    groups = K // Q4_K_GROUP
    scales = _require_array("scales", scales, (groups, N), np.uint8)
    mins = _require_array("mins", mins, (groups, N), np.uint8)
    d = _require_array("d", d, (K // QK_K, N), np.float16)
    dmin = _require_array("dmin", dmin, (K // QK_K, N), np.float16)
    if np.any(q > 15):
        raise ValueError("Q4_K q values must be in 0..15")
    if np.any(scales > 63) or np.any(mins > 63):
        raise ValueError("Q4_K scale/min codes must be in 0..63")
    result = np.zeros(N * (K // QK_K) * Q4_K_BLOCK_BYTES, dtype=np.uint8)
    blocks_per_row = K // QK_K
    for col in range(N):
        for bi in range(blocks_per_row):
            offset = (col * blocks_per_row + bi) * Q4_K_BLOCK_BYTES
            block = result[offset : offset + Q4_K_BLOCK_BYTES]
            block[0:2] = np.asarray([d[bi, col]], dtype="<f2").view(np.uint8)
            block[2:4] = np.asarray([dmin[bi, col]], dtype="<f2").view(np.uint8)
            g0 = bi * 8
            block[4:16] = pack_scale_min_k4(
                scales[g0 : g0 + 8, col].copy(), mins[g0 : g0 + 8, col].copy()
            )
            k0 = bi * QK_K
            for region in range(4):
                low = q[k0 + region * 64 : k0 + region * 64 + 32, col]
                high = q[k0 + region * 64 + 32 : k0 + region * 64 + 64, col]
                block[16 + region * 32 : 16 + (region + 1) * 32] = low | (high << 4)
    return result


def make_deterministic_native_q4_k(
    config: Q4KSConfig, *, seed: int = 0x4B535F34
) -> tuple[np.ndarray, np.ndarray]:
    """Generate normalized LLM-like activations and native Q4_K weights."""

    rng = np.random.default_rng(seed)

    A = rng.standard_normal((config.M, config.K), dtype=np.float32)
    A -= A.mean(axis=1, keepdims=True)
    A /= np.sqrt(np.mean(A * A, axis=1, keepdims=True) + 1.0e-12)
    A = A.astype(bfloat16)

    weights = rng.standard_normal((config.K, config.N), dtype=np.float32)
    weights -= weights.mean(axis=0, keepdims=True)
    weight_rms = np.sqrt(np.mean(weights * weights, axis=0, keepdims=True))
    weights *= (1.0 / np.sqrt(np.float32(config.K))) / weight_rms

    blocks = config.K // QK_K
    grouped = weights.reshape(blocks, 8, Q4_K_GROUP, config.N)
    lower = np.minimum(grouped.min(axis=2), 0.0)
    upper = np.maximum(grouped.max(axis=2), 0.0)
    desired_scales = np.maximum((upper - lower) / 15.0, 1.0e-12)
    desired_biases = -lower

    d = (desired_scales.max(axis=1) / 63.0).astype(np.float16)
    dmin = (desired_biases.max(axis=1) / 63.0).astype(np.float16)
    d_float = d.astype(np.float32)
    dmin_float = dmin.astype(np.float32)
    scales = np.clip(
        np.rint(desired_scales / d_float[:, None, :]), 1, 63
    ).astype(np.uint8)
    mins = np.where(
        dmin_float[:, None, :] == 0,
        0,
        np.clip(np.rint(desired_biases / dmin_float[:, None, :]), 0, 63),
    ).astype(np.uint8)
    effective_scales = d_float[:, None, :] * scales.astype(np.float32)
    effective_biases = dmin_float[:, None, :] * mins.astype(np.float32)
    q = np.clip(
        np.rint(
            (grouped + effective_biases[:, :, None, :])
            / effective_scales[:, :, None, :]
        ),
        0,
        15,
    ).astype(np.uint8)
    q = q.reshape(config.K, config.N)
    scales = scales.reshape(config.K // Q4_K_GROUP, config.N)
    mins = mins.reshape(config.K // Q4_K_GROUP, config.N)
    native = encode_q4_k(q, scales, mins, d, dmin)
    return A, native


def _tile_coordinates(config: Q4KSConfig) -> Iterable[tuple[int, int, int]]:
    per_col = config.n_n_tiles // config.n_aie_cols
    for col in range(config.n_aie_cols):
        for n_round in range(per_col):
            n_tile = col + n_round * config.n_aie_cols
            for k_tile in range(config.n_k_tiles):
                yield col, k_tile, n_tile


def _compressed_tile_coordinates(
    config: Q4KSConfig,
) -> Iterable[tuple[int, int, int]]:
    """Order compressed tiles for independent cores or K-split cascades."""

    if config.accumulation_mode not in ("cascade-shared", "cascade-hybrid"):
        yield from _tile_coordinates(config)
        return
    cascade_rows = (
        2 if config.accumulation_mode == "cascade-hybrid" else N_AIE_ROWS
    )
    per_col = config.n_n_tiles // config.n_aie_cols
    chunks = config.n_k_tiles // cascade_rows
    for col in range(config.n_aie_cols):
        for n_round in range(per_col):
            n_tile = col + n_round * config.n_aie_cols
            for row in range(cascade_rows):
                for chunk in range(chunks):
                    yield col, chunk * cascade_rows + row, n_tile


def _microtile_values(values: np.ndarray, k: int, n: int) -> Iterator[np.ndarray]:
    for micro_k in range(0, k, 8):
        for micro_n in range(0, n, 8):
            yield values[micro_k : micro_k + 8, micro_n : micro_n + 8]


def _pack_compressed(
    q: np.ndarray, scales: np.ndarray, biases: np.ndarray, config: Q4KSConfig
) -> np.ndarray:
    packed = np.zeros(config.q4_prepared_bytes, dtype=np.uint8)
    scales_bf16 = scales.astype(bfloat16)
    biases_bf16 = biases.astype(bfloat16)
    for ti, (_, kt, nt) in enumerate(_compressed_tile_coordinates(config)):
        k0, n0 = kt * config.k, nt * config.n
        tile = packed[ti * config.tile_bytes : (ti + 1) * config.tile_bytes]
        cursor = 0
        q_tile = q[k0 : k0 + config.k, n0 : n0 + config.n]
        for block in _microtile_values(q_tile, config.k, config.n):
            nibble = block[:, 0::2] | (block[:, 1::2] << 4)
            tile[cursor : cursor + 32] = nibble.reshape(-1)
            cursor += 32
        g0 = k0 // Q4_K_GROUP
        for group in range(config.groups_per_tile):
            for micro_n in range(0, config.n, 8):
                ns = slice(n0 + micro_n, n0 + micro_n + 8)
                sb = scales_bf16[g0 + group, ns].view(np.uint8)
                bb = biases_bf16[g0 + group, ns].view(np.uint8)
                tile[cursor : cursor + 16] = sb
                tile[cursor + 16 : cursor + 32] = bb
                cursor += 32
        if cursor != config.raw_tile_bytes:
            raise AssertionError(f"wrote {cursor}, expected {config.raw_tile_bytes}")
    return packed


def unpack_prepared_q4(
    packed: np.ndarray, config: Q4KSConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed = _require_array("packed", packed, (config.q4_prepared_bytes,), np.uint8)
    q = np.empty((config.K, config.N), dtype=np.uint8)
    scales = np.empty((config.K // Q4_K_GROUP, config.N), dtype=bfloat16)
    biases = np.empty_like(scales)
    for ti, (_, kt, nt) in enumerate(_compressed_tile_coordinates(config)):
        k0, n0 = kt * config.k, nt * config.n
        tile = packed[ti * config.tile_bytes : (ti + 1) * config.tile_bytes]
        cursor = 0
        for mk in range(0, config.k, 8):
            for mn in range(0, config.n, 8):
                nibble = tile[cursor : cursor + 32].reshape(8, 4)
                block = np.empty((8, 8), dtype=np.uint8)
                block[:, 0::2] = nibble & 0x0F
                block[:, 1::2] = nibble >> 4
                q[k0 + mk : k0 + mk + 8, n0 + mn : n0 + mn + 8] = block
                cursor += 32
        g0 = k0 // Q4_K_GROUP
        for group in range(config.groups_per_tile):
            for mn in range(0, config.n, 8):
                ns = slice(n0 + mn, n0 + mn + 8)
                scales[g0 + group, ns] = tile[cursor : cursor + 16].copy().view(bfloat16)
                biases[g0 + group, ns] = tile[cursor + 16 : cursor + 32].copy().view(bfloat16)
                cursor += 32
        if np.any(tile[config.raw_tile_bytes :]):
            raise ValueError("packed tile has non-zero alignment padding")
    return q, scales, biases


def dequantize_prepared_q4(packed: np.ndarray, config: Q4KSConfig) -> np.ndarray:
    q, scales, biases = unpack_prepared_q4(packed, config)
    groups = np.arange(config.K) // Q4_K_GROUP
    values = (
        q.astype(np.float32) * scales[groups].astype(np.float32)
        - biases[groups].astype(np.float32)
    )
    return values.astype(bfloat16)


def float_to_bfp16ebs8(values: np.ndarray) -> np.ndarray:
    """Encode flat float values using AIE2P's 8-mantissa shared-exponent ABI."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or values.size % 8:
        raise ValueError("BFP16 input must be a flat multiple of 8 values")
    if not np.all(np.isfinite(values)):
        raise ValueError("BFP16 input values must be finite")
    out = np.empty(values.size * 9 // 8, dtype=np.uint8)
    words = values.view(np.uint32)
    cursor = 0
    for start in range(0, values.size, 8):
        w = words[start : start + 8]
        exps = ((w >> 23) & 0xFF).astype(np.int32)
        max_exp = int(exps.max())
        out[cursor] = max_exp
        for lane in range(8):
            mantissa = int(w[lane] & 0x7FFFFF)
            if exps[lane] != 0:
                mantissa |= 0x800000
            if w[lane] & 0x80000000:
                mantissa = -mantissa
            quant = mantissa >> 17
            delta = max_exp - int(exps[lane])
            quant = (-1 if quant < 0 else 0) if delta >= 32 else quant >> delta
            out[cursor + 1 + lane] = np.uint8(quant & 0xFF)
        cursor += 9
    return out


def bfp16ebs8_to_float(packed: np.ndarray) -> np.ndarray:
    if not isinstance(packed, np.ndarray) or packed.ndim != 1:
        raise TypeError("packed BFP16 must be a flat numpy.ndarray")
    packed = _require_array("packed BFP16", packed, packed.shape, np.uint8)
    if packed.size % 9:
        raise ValueError("packed BFP16 byte count must be divisible by 9")
    out = np.empty(packed.size // 9 * 8, dtype=np.float32)
    oi = 0
    for cursor in range(0, packed.size, 9):
        multiplier = np.ldexp(np.float32(1.0 / 64.0), int(packed[cursor]) - 127)
        mantissas = packed[cursor + 1 : cursor + 9].view(np.int8).astype(np.float32)
        out[oi : oi + 8] = mantissas * multiplier
        oi += 8
    return out


def _pack_expanded(
    q: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    config: Q4KSConfig,
    storage_type: str,
) -> np.ndarray:
    tile_bytes = config.prepared_tile_bytes(storage_type)
    result = np.empty(config.n_tiles * tile_bytes, dtype=np.uint8)
    scales_bf16 = scales.astype(bfloat16)
    biases_bf16 = biases.astype(bfloat16)
    groups = np.arange(config.K) // Q4_K_GROUP
    dequant = (
        q.astype(np.float32) * scales_bf16[groups].astype(np.float32)
        - biases_bf16[groups].astype(np.float32)
    ).astype(bfloat16)
    for ti, (_, kt, nt) in enumerate(_tile_coordinates(config)):
        k0, n0 = kt * config.k, nt * config.n
        tile = result[ti * tile_bytes : (ti + 1) * tile_bytes]
        q_tile = q[k0 : k0 + config.k, n0 : n0 + config.n]
        w_tile = dequant[k0 : k0 + config.k, n0 : n0 + config.n]
        cursor = 0
        if storage_type == "bf16":
            for block in _microtile_values(w_tile, config.k, config.n):
                raw = block.reshape(-1).view(np.uint8)
                tile[cursor : cursor + raw.size] = raw
                cursor += raw.size
        elif storage_type == "bfp16":
            for block in _microtile_values(w_tile, config.k, config.n):
                # mac_8x8_8x8T consumes B blocks column-major.  Shared
                # exponents therefore cover the eight K values in a column.
                raw = float_to_bfp16ebs8(
                    block.astype(np.float32).T.reshape(-1)
                )
                tile[cursor : cursor + raw.size] = raw
                cursor += raw.size
        elif storage_type == "int8":
            for block in _microtile_values(q_tile, config.k, config.n):
                tile[cursor : cursor + 64] = block.astype(np.int8).view(np.uint8).reshape(-1)
                cursor += 64
            g0 = k0 // Q4_K_GROUP
            for group in range(config.groups_per_tile):
                for mn in range(0, config.n, 8):
                    ns = slice(n0 + mn, n0 + mn + 8)
                    tile[cursor : cursor + 16] = scales_bf16[g0 + group, ns].view(np.uint8)
                    tile[cursor + 16 : cursor + 32] = biases_bf16[g0 + group, ns].view(np.uint8)
                    cursor += 32
        else:
            raise ValueError(f"expanded storage must be bf16, bfp16, or int8")
        if cursor != tile_bytes:
            raise AssertionError(f"expanded tile wrote {cursor}, expected {tile_bytes}")
    return result


def _pack_expanded_cascade_register(
    q: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    config: Q4KSConfig,
) -> np.ndarray:
    """Pack BFP16 by cascade row, 16-column panel, then the full K/4 shard.

    This model-load representation lets a MemTile replay one 16-column B
    panel while a compute tile keeps all K/4 partial sums in accfloat
    registers.  No FP32 partial-C buffer is read or written.
    """

    shard_k = config.K // N_AIE_ROWS
    result = np.empty(config.prepared_bytes, dtype=np.uint8)
    scales_bf16 = scales.astype(bfloat16)
    biases_bf16 = biases.astype(bfloat16)
    groups = np.arange(config.K) // Q4_K_GROUP
    dequant = (
        q.astype(np.float32) * scales_bf16[groups].astype(np.float32)
        - biases_bf16[groups].astype(np.float32)
    ).astype(bfloat16)

    cursor = 0
    n_rounds = config.N // (config.n * config.n_aie_cols)
    for col in range(config.n_aie_cols):
        for n_round in range(n_rounds):
            n0 = (n_round * config.n_aie_cols + col) * config.n
            for row in range(N_AIE_ROWS):
                k0 = row * shard_k
                for n16 in range(0, config.n, 16):
                    for k8 in range(0, shard_k, 8):
                        for n8 in (0, 8):
                            block = dequant[
                                k0 + k8 : k0 + k8 + 8,
                                n0 + n16 + n8 : n0 + n16 + n8 + 8,
                            ]
                            raw = float_to_bfp16ebs8(
                                block.astype(np.float32).T.reshape(-1)
                            )
                            result[cursor : cursor + raw.size] = raw
                            cursor += raw.size
    if cursor != config.prepared_bytes:
        raise AssertionError(
            f"cascade-register packing wrote {cursor}, "
            f"expected {config.prepared_bytes}"
        )
    return result


def _pack_expanded_cascade_chunked(
    q: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    config: Q4KSConfig,
) -> np.ndarray:
    """Pack BFP16 by cascade row and 256-K full-N compute chunk."""

    shard_k = config.K // N_AIE_ROWS
    result = np.empty(config.prepared_bytes, dtype=np.uint8)
    scales_bf16 = scales.astype(bfloat16)
    biases_bf16 = biases.astype(bfloat16)
    groups = np.arange(config.K) // Q4_K_GROUP
    dequant = (
        q.astype(np.float32) * scales_bf16[groups].astype(np.float32)
        - biases_bf16[groups].astype(np.float32)
    ).astype(bfloat16)

    cursor = 0
    n_rounds = config.N // (config.n * config.n_aie_cols)
    for col in range(config.n_aie_cols):
        for n_round in range(n_rounds):
            n0 = (n_round * config.n_aie_cols + col) * config.n
            for row in range(N_AIE_ROWS):
                row_k0 = row * shard_k
                for chunk in range(0, shard_k, CASCADE_CHUNK_K):
                    k0 = row_k0 + chunk
                    for k8 in range(0, CASCADE_CHUNK_K, 8):
                        for n8 in range(0, config.n, 8):
                            block = dequant[
                                k0 + k8 : k0 + k8 + 8,
                                n0 + n8 : n0 + n8 + 8,
                            ]
                            raw = float_to_bfp16ebs8(
                                block.astype(np.float32).T.reshape(-1)
                            )
                            result[cursor : cursor + raw.size] = raw
                            cursor += raw.size
    if cursor != config.prepared_bytes:
        raise AssertionError(
            f"cascade-chunked packing wrote {cursor}, "
            f"expected {config.prepared_bytes}"
        )
    return result


def prepare_q4ks_weights(
    native: np.ndarray,
    config: Q4KSConfig,
    storage_type: Literal["q4", "bf16", "bfp16", "int8"] = "q4",
) -> np.ndarray:
    if storage_type not in STORAGE_TYPES:
        raise ValueError(f"storage_type must be one of {STORAGE_TYPES}")
    q, scales, biases = decode_q4_k(native, K=config.K, N=config.N)
    if storage_type == "q4":
        return _pack_compressed(q, scales, biases, config)
    if storage_type == "bfp16" and config.accumulation_mode == "cascade-register":
        return _pack_expanded_cascade_register(q, scales, biases, config)
    if (
        storage_type == "bfp16"
        and config.accumulation_mode == "cascade-chunked"
    ):
        return _pack_expanded_cascade_chunked(q, scales, biases, config)
    return _pack_expanded(q, scales, biases, config, storage_type)


def quantize_activations_int8(
    A: np.ndarray, *, group_size: int = Q4_K_GROUP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise TypeError("A must be a rank-2 numpy.ndarray")
    if A.dtype != np.dtype(bfloat16):
        raise TypeError("A must have BF16 dtype")
    M, K = A.shape
    if K % group_size:
        raise ValueError("activation group size must divide K")
    groups = K // group_size
    reshaped = A.astype(np.float32).reshape(M, groups, group_size)
    maxima = np.max(np.abs(reshaped), axis=2)
    quant_scales = np.where(maxima == 0, 1.0, maxima / 127.0).astype(
        np.float32
    )
    scales = quant_scales.astype(bfloat16)
    q = np.where(
        reshaped >= 0,
        np.floor(reshaped / quant_scales[..., None] + 0.5),
        np.ceil(reshaped / quant_scales[..., None] - 0.5),
    )
    q = np.clip(q, -127, 127).astype(np.int8).reshape(M, K)
    sums = q.reshape(M, groups, group_size).astype(np.int32).sum(axis=2)
    return q, scales, sums


def reference_matmul(
    A: np.ndarray,
    native: np.ndarray,
    config: Q4KSConfig,
    *,
    compute_type: str | None = None,
) -> np.ndarray:
    A = _require_array("A", A, (config.M, config.K), bfloat16)
    mode = compute_type or config.compute_type
    q, native_scales, native_biases = decode_q4_k(native, K=config.K, N=config.N)
    scales = native_scales.astype(bfloat16).astype(np.float32)
    biases = native_biases.astype(bfloat16).astype(np.float32)
    group_index = np.arange(config.K) // Q4_K_GROUP
    if mode == "int8":
        A8, a_scales, a_sums = quantize_activations_int8(A)
        C = np.zeros((config.M, config.N), dtype=np.float32)
        for k0 in range(0, config.K, config.k):
            partial = np.zeros_like(C)
            for group in range(
                k0 // Q4_K_GROUP, (k0 + config.k) // Q4_K_GROUP
            ):
                ks = slice(group * Q4_K_GROUP, (group + 1) * Q4_K_GROUP)
                dot = A8[:, ks].astype(np.int32) @ q[ks].astype(np.int32)
                corrected = (
                    dot.astype(np.float32) * scales[group][None, :]
                    - a_sums[:, group, None].astype(np.float32)
                    * biases[group][None, :]
                )
                partial += (
                    a_scales[:, group, None].astype(np.float32) * corrected
                )
            C = (C + partial).astype(bfloat16).astype(np.float32)
        return C.astype(bfloat16)
    weights = (
        q.astype(np.float32) * scales[group_index] - biases[group_index]
    ).astype(bfloat16)
    if mode == "bfp16":
        a_flat = A.astype(np.float32).reshape(-1)
        A_compute = bfp16ebs8_to_float(
            float_to_bfp16ebs8(a_flat)
        ).reshape(config.M, config.K)
        rounded = np.empty_like(weights, dtype=np.float32)
        for mk in range(0, config.K, 8):
            for mn in range(0, config.N, 8):
                block = weights[mk : mk + 8, mn : mn + 8].astype(np.float32)
                encoded = float_to_bfp16ebs8(block.T.reshape(-1))
                decoded = bfp16ebs8_to_float(encoded).reshape(8, 8).T
                rounded[mk : mk + 8, mn : mn + 8] = decoded
        weights = rounded
    else:
        A_compute = A.astype(np.float32)
    if config.accumulation_mode == "cascade-hybrid":
        cascade_rows = 2
        chunks_per_row = config.K // (cascade_rows * config.k)
        row_partials: list[np.ndarray] = []
        for cascade_row in range(cascade_rows):
            partial = np.zeros((config.M, config.N), dtype=np.float32)
            for chunk in range(chunks_per_row):
                k0 = (chunk * cascade_rows + cascade_row) * config.k
                ks = slice(k0, k0 + config.k)
                partial = (
                    partial
                    + A_compute[:, ks] @ weights[ks].astype(np.float32)
                )
                if chunk + 1 != chunks_per_row:
                    partial = partial.astype(bfloat16).astype(np.float32)
            row_partials.append(partial)
        C = row_partials[-1]
        for cascade_row in range(cascade_rows - 2, -1, -1):
            C = C + row_partials[cascade_row]
        return C.astype(bfloat16)

    C = np.zeros((config.M, config.N), dtype=np.float32)
    for k0 in range(0, config.K, config.k):
        ks = slice(k0, k0 + config.k)
        C = (
            C + A_compute[:, ks] @ weights[ks].astype(np.float32)
        ).astype(bfloat16).astype(np.float32)
    return C.astype(bfloat16)


def b_partition_taps(config: Q4KSConfig) -> list[TapSpec]:
    tiles_per_col = config.n_n_tiles // config.n_aie_cols
    bytes_per_col = tiles_per_col * config.n_k_tiles * config.tile_bytes
    return [
        TapSpec(
            "B",
            col * bytes_per_col,
            (tiles_per_col, config.n_k_tiles, config.packed_rows, config.k),
            (
                config.n_k_tiles * config.tile_bytes,
                config.tile_bytes,
                config.k,
                1,
            ),
            col,
        )
        for col in range(config.n_aie_cols)
    ]


def a_transfer_taps(config: Q4KSConfig) -> list[TapSpec]:
    n_shims = min(N_AIE_ROWS, config.n_aie_cols)
    rows_per_shim = N_AIE_ROWS // config.n_aie_cols if config.n_aie_cols < 4 else 1
    repeat = config.n_n_tiles // config.n_aie_cols
    taps: list[TapSpec] = []
    for row_block in range(config.M // (config.m_c * N_AIE_ROWS)):
        for col in range(n_shims):
            taps.append(
                TapSpec(
                    "A",
                    row_block * N_AIE_ROWS * config.m_c * config.K
                    + col * rows_per_shim * config.m_c * config.K,
                    (repeat, config.n_k_tiles, config.m_c * rows_per_shim, config.k),
                    (0, config.k, config.K, 1),
                    col,
                    row_block,
                )
            )
    return taps


def c_transfer_taps(config: Q4KSConfig) -> list[TapSpec]:
    taps: list[TapSpec] = []
    n_row_blocks = config.M // (config.m_c * N_AIE_ROWS)
    for row_base in range(0, n_row_blocks, 2):
        for col in range(config.n_aie_cols):
            taps.append(
                TapSpec(
                    "C",
                    row_base * N_AIE_ROWS * config.m_c * config.N + col * config.n,
                    (
                        min(2, n_row_blocks - row_base),
                        config.n_n_tiles // config.n_aie_cols,
                        N_AIE_ROWS * config.m_c,
                        config.n,
                    ),
                    (
                        N_AIE_ROWS * config.m_c * config.N,
                        config.n * config.n_aie_cols,
                        config.N,
                        1,
                    ),
                    col,
                    row_base,
                )
            )
    return taps
