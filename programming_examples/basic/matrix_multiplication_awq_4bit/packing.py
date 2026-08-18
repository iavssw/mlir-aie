# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Host-side data contract for the NPU2 whole-array AWQ INT4 matmul.

The public packed buffer is a flat ``uint8`` array.  Tiles are ordered by AIE
column, then round-robin N tile, then K tile.  A tile contains:

* 8x8-microtiled weights, two unsigned nibbles per byte (low-N first),
* group-major BF16 scales,
* group-major zero points as duplicated 8-byte vectors, and
* zero padding to a multiple of the K tile size.

Keeping this module independent of IRON makes the ABI and validation directly
testable on machines without an NPU runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from ml_dtypes import bfloat16


N_AIE_ROWS = 4
CORE_LOCAL_MEMORY_BYTES = 64 * 1024
CORE_STACK_BYTES = 0xD00
CORE_SAFETY_BYTES = 4 * 1024
SUPPORTED_GROUP_SIZES = (32, 64, 128)
SUPPORTED_COLUMNS = (1, 2, 4, 8)
SUPPORTED_OUTPUT_DTYPES = ("bf16", "f32")


def ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _normalize_output_dtype(dtype: str | np.dtype | type) -> str:
    if dtype == "bf16":
        return "bf16"
    if dtype == "f32":
        return "f32"
    if np.dtype(dtype) == np.dtype(bfloat16):
        return "bf16"
    if np.dtype(dtype) == np.dtype(np.float32):
        return "f32"
    raise ValueError("output dtype must be 'bf16' or 'f32'")


@dataclass(frozen=True)
class AWQConfig:
    """Compile-time and host-layout parameters for one design."""

    M: int = 1024
    K: int = 1024
    N: int = 2048
    m: int = 64
    k: int = 128
    n: int = 64
    group_size: int = 128
    n_aie_cols: int = 8
    dtype_out: str = "bf16"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype_out", _normalize_output_dtype(self.dtype_out))
        self.validate()

    @property
    def groups_per_tile(self) -> int:
        return self.k // self.group_size

    @property
    def weights_bytes_per_tile(self) -> int:
        return self.k * self.n // 2

    @property
    def scales_bytes_per_tile(self) -> int:
        return self.groups_per_tile * self.n * np.dtype(bfloat16).itemsize

    @property
    def zeros_bytes_per_tile(self) -> int:
        # Each group has n logical uint4 zero points.  Every 8-column vector is
        # duplicated so the core can issue one naturally aligned 16-byte load.
        return self.groups_per_tile * self.n * 2

    @property
    def raw_tile_bytes(self) -> int:
        return (
            self.weights_bytes_per_tile
            + self.scales_bytes_per_tile
            + self.zeros_bytes_per_tile
        )

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
    def n_k_tiles(self) -> int:
        return self.K // self.k

    @property
    def n_n_tiles(self) -> int:
        return self.N // self.n

    @property
    def n_tiles(self) -> int:
        return self.n_k_tiles * self.n_n_tiles

    @property
    def packed_bytes(self) -> int:
        return self.n_tiles * self.tile_bytes

    @property
    def output_itemsize(self) -> int:
        return 2 if self.dtype_out == "bf16" else 4

    @property
    def output_fifo_depth(self) -> int:
        return 2 if self.dtype_out == "bf16" else 1

    @property
    def core_memory_components(self) -> dict[str, int]:
        return {
            "A FIFO (depth 2)": 2 * (self.m // 2) * self.k * 2,
            "packed-B FIFO (depth 2)": 2 * self.tile_bytes,
            f"C FIFO (depth {self.output_fifo_depth})": (
                self.output_fifo_depth * self.m * self.n * self.output_itemsize
            ),
            "BF16 dequant scratch": self.k * self.n * 2,
            "stack": CORE_STACK_BYTES,
            "safety margin": CORE_SAFETY_BYTES,
        }

    @property
    def core_memory_bytes(self) -> int:
        return sum(self.core_memory_components.values())

    def validate(self) -> None:
        values = {
            "M": self.M,
            "K": self.K,
            "N": self.N,
            "m": self.m,
            "k": self.k,
            "n": self.n,
        }
        for name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_aie_cols not in SUPPORTED_COLUMNS:
            raise ValueError(f"n_aie_cols must be one of {SUPPORTED_COLUMNS}")
        if self.group_size not in SUPPORTED_GROUP_SIZES:
            raise ValueError(f"group_size must be one of {SUPPORTED_GROUP_SIZES}")
        if self.m % 32:
            raise ValueError("m must be divisible by 32 for two 8x8-MMUL A halves")
        if self.k % 8:
            raise ValueError("k must be divisible by the 8-wide MMUL K dimension")
        if self.n % 16:
            raise ValueError("n must be divisible by two 8-wide MMUL N tiles")
        if self.k % self.group_size:
            raise ValueError("group_size must divide k")
        if self.K % self.group_size:
            raise ValueError("group_size must divide K")
        if self.M % (self.m * N_AIE_ROWS):
            raise ValueError("M must be divisible by m * 4 AIE rows")
        if self.K % self.k:
            raise ValueError("K must be divisible by k")
        if self.N % (self.n * self.n_aie_cols):
            raise ValueError("N must be divisible by n * n_aie_cols")
        row_blocks = self.M // (self.m * N_AIE_ROWS)
        if row_blocks % 2:
            raise ValueError("M / (m * 4) must be even for ping-pong transfer blocks")

        # Every host transfer uses a four-dimensional NPU DMA descriptor.
        dma_sizes = {
            "K/k": self.n_k_tiles,
            "N/(n*cols)": self.n_n_tiles // self.n_aie_cols,
            "packed_rows": self.packed_rows,
            "k": self.k,
            "m*rows_per_shim": self.m
            * (N_AIE_ROWS // self.n_aie_cols if self.n_aie_cols < 4 else 1),
            "4*m": N_AIE_ROWS * self.m,
            "n": self.n,
        }
        too_large = {name: value for name, value in dma_sizes.items() if value > 1023}
        if too_large:
            detail = ", ".join(f"{name}={value}" for name, value in too_large.items())
            raise ValueError(f"DMA size exceeds the 10-bit descriptor limit: {detail}")

        memory = self.core_memory_components
        if sum(memory.values()) > CORE_LOCAL_MEMORY_BYTES:
            detail = ", ".join(f"{name}={value}" for name, value in memory.items())
            raise ValueError(
                f"configuration needs {sum(memory.values())} bytes of per-core "
                f"memory, exceeding {CORE_LOCAL_MEMORY_BYTES}: {detail}"
            )


@dataclass(frozen=True)
class TapSpec:
    """Small host-only model of a four-dimensional contiguous DMA TAP."""

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


def b_partition_taps(config: AWQConfig) -> list[TapSpec]:
    """Return the exact packed-B partition transferred by each AIE column."""

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


def a_transfer_taps(config: AWQConfig) -> list[TapSpec]:
    """Return one logical A transfer per row-block and active A shim."""

    n_shims = min(N_AIE_ROWS, config.n_aie_cols)
    rows_per_shim = N_AIE_ROWS // config.n_aie_cols if config.n_aie_cols < 4 else 1
    repeat = config.n_n_tiles // config.n_aie_cols
    taps: list[TapSpec] = []
    for row_block in range(config.M // (config.m * N_AIE_ROWS)):
        for col in range(n_shims):
            taps.append(
                TapSpec(
                    "A",
                    row_block * N_AIE_ROWS * config.m * config.K
                    + col * rows_per_shim * config.m * config.K,
                    (repeat, config.n_k_tiles, config.m * rows_per_shim, config.k),
                    (0, config.k, config.K, 1),
                    col,
                    row_block,
                )
            )
    return taps


def c_transfer_taps(config: AWQConfig) -> list[TapSpec]:
    """Return C drains in the current two-row-block ping-pong order."""

    taps: list[TapSpec] = []
    n_row_blocks = config.M // (config.m * N_AIE_ROWS)
    for row_base in range(0, n_row_blocks, 2):
        current_rows = min(2, n_row_blocks - row_base)
        for col in range(config.n_aie_cols):
            taps.append(
                TapSpec(
                    "C",
                    row_base * N_AIE_ROWS * config.m * config.N + col * config.n,
                    (
                        current_rows,
                        config.n_n_tiles // config.n_aie_cols,
                        N_AIE_ROWS * config.m,
                        config.n,
                    ),
                    (
                        N_AIE_ROWS * config.m * config.N,
                        config.n * config.n_aie_cols,
                        config.N,
                        1,
                    ),
                    col,
                    row_base,
                )
            )
    return taps


def split_a_tile_to_microtiles(a_tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model the MemTile stream that emits two microtiled A-half objects."""

    if not isinstance(a_tile, np.ndarray) or a_tile.ndim != 2:
        raise TypeError("a_tile must be a rank-2 numpy.ndarray")
    if a_tile.dtype != np.dtype(bfloat16):
        raise TypeError(f"a_tile must have dtype {np.dtype(bfloat16)}")
    m, k = a_tile.shape
    if m % 32 or k % 8:
        raise ValueError("a_tile rows must be divisible by 32 and columns by 8")
    halves: list[np.ndarray] = []
    for half_index in range(2):
        half = a_tile[half_index * (m // 2) : (half_index + 1) * (m // 2)]
        microtiles = [
            half[row : row + 8, inner : inner + 8].reshape(-1)
            for row in range(0, m // 2, 8)
            for inner in range(0, k, 8)
        ]
        halves.append(np.concatenate(microtiles))
    return halves[0], halves[1]


def join_a_microtile_halves(
    first: np.ndarray, second: np.ndarray, *, m: int, k: int
) -> np.ndarray:
    """Inverse host model for :func:`split_a_tile_to_microtiles`."""

    expected_shape = ((m // 2) * k,)
    first = _require_array("first", first, expected_shape, bfloat16)
    second = _require_array("second", second, expected_shape, bfloat16)
    result = np.empty((m, k), dtype=bfloat16)
    for half_index, stream in enumerate((first, second)):
        cursor = 0
        row_base = half_index * (m // 2)
        for row in range(0, m // 2, 8):
            for inner in range(0, k, 8):
                result[row_base + row : row_base + row + 8, inner : inner + 8] = (
                    stream[cursor : cursor + 64].reshape(8, 8)
                )
                cursor += 64
    return result


def _require_array(
    name: str, value: np.ndarray, shape: tuple[int, ...], dtype: np.dtype | type
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    expected_dtype = np.dtype(dtype)
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_uint4(name: str, value: np.ndarray) -> None:
    if value.size and int(value.max()) > 15:
        raise ValueError(f"{name} contains a value outside unsigned INT4 range 0..15")


def _tile_coordinates(config: AWQConfig) -> Iterable[tuple[int, int, int]]:
    """Yield ``(column, k_tile, n_tile)`` in physical packed-buffer order."""

    n_tiles_per_col = config.n_n_tiles // config.n_aie_cols
    for col in range(config.n_aie_cols):
        for n_round in range(n_tiles_per_col):
            n_tile = col + n_round * config.n_aie_cols
            for k_tile in range(config.n_k_tiles):
                yield col, k_tile, n_tile


def pack_awq_weights(
    qweight: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    config: AWQConfig,
) -> np.ndarray:
    """Pack logical AWQ tensors according to the documented device ABI."""

    qweight = _require_array("qweight", qweight, (config.K, config.N), np.uint8)
    scales = _require_array(
        "scales", scales, (config.K // config.group_size, config.N), bfloat16
    )
    zeros = _require_array(
        "zeros", zeros, (config.K // config.group_size, config.N), np.uint8
    )
    _require_uint4("qweight", qweight)
    _require_uint4("zeros", zeros)

    packed = np.zeros(config.packed_bytes, dtype=np.uint8)
    for tile_index, (_, k_tile, n_tile) in enumerate(_tile_coordinates(config)):
        k0 = k_tile * config.k
        n0 = n_tile * config.n
        tile = packed[
            tile_index * config.tile_bytes : (tile_index + 1) * config.tile_bytes
        ]
        cursor = 0

        q_tile = qweight[k0 : k0 + config.k, n0 : n0 + config.n]
        for micro_k in range(0, config.k, 8):
            for micro_n in range(0, config.n, 8):
                block = q_tile[micro_k : micro_k + 8, micro_n : micro_n + 8]
                nibble_bytes = block[:, 0::2] | (block[:, 1::2] << 4)
                tile[cursor : cursor + 32] = nibble_bytes.reshape(-1)
                cursor += 32

        first_group = k0 // config.group_size
        group_slice = slice(first_group, first_group + config.groups_per_tile)
        n_slice = slice(n0, n0 + config.n)
        scale_bytes = scales[group_slice, n_slice].view(np.uint8).reshape(-1)
        tile[cursor : cursor + scale_bytes.size] = scale_bytes
        cursor += scale_bytes.size

        zero_tile = zeros[group_slice, n_slice]
        for group in range(config.groups_per_tile):
            for micro_n in range(0, config.n, 8):
                vector = zero_tile[group, micro_n : micro_n + 8]
                tile[cursor : cursor + 8] = vector
                tile[cursor + 8 : cursor + 16] = vector
                cursor += 16

        if cursor != config.raw_tile_bytes:
            raise AssertionError(
                f"internal packing error: wrote {cursor}, "
                f"expected {config.raw_tile_bytes}"
            )
    return packed


def unpack_awq_weights(
    packed: np.ndarray, config: AWQConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_awq_weights`, including padding validation."""

    packed = _require_array("packed", packed, (config.packed_bytes,), np.uint8)
    qweight = np.empty((config.K, config.N), dtype=np.uint8)
    scales = np.empty((config.K // config.group_size, config.N), dtype=bfloat16)
    zeros = np.empty((config.K // config.group_size, config.N), dtype=np.uint8)

    for tile_index, (_, k_tile, n_tile) in enumerate(_tile_coordinates(config)):
        k0 = k_tile * config.k
        n0 = n_tile * config.n
        tile = packed[
            tile_index * config.tile_bytes : (tile_index + 1) * config.tile_bytes
        ]
        cursor = 0
        for micro_k in range(0, config.k, 8):
            for micro_n in range(0, config.n, 8):
                nibble_bytes = tile[cursor : cursor + 32].reshape(8, 4)
                block = np.empty((8, 8), dtype=np.uint8)
                block[:, 0::2] = nibble_bytes & 0x0F
                block[:, 1::2] = nibble_bytes >> 4
                qweight[
                    k0 + micro_k : k0 + micro_k + 8,
                    n0 + micro_n : n0 + micro_n + 8,
                ] = block
                cursor += 32

        first_group = k0 // config.group_size
        group_slice = slice(first_group, first_group + config.groups_per_tile)
        n_slice = slice(n0, n0 + config.n)
        n_scale_bytes = config.scales_bytes_per_tile
        scales[group_slice, n_slice] = (
            tile[cursor : cursor + n_scale_bytes]
            .copy()
            .view(bfloat16)
            .reshape(config.groups_per_tile, config.n)
        )
        cursor += n_scale_bytes

        for group in range(config.groups_per_tile):
            for micro_n in range(0, config.n, 8):
                first = tile[cursor : cursor + 8]
                duplicate = tile[cursor + 8 : cursor + 16]
                if not np.array_equal(first, duplicate):
                    raise ValueError("packed zero-point vector is not duplicated")
                zeros[first_group + group, n0 + micro_n : n0 + micro_n + 8] = first
                cursor += 16

        if np.any(tile[config.raw_tile_bytes :]):
            raise ValueError("packed tile has non-zero alignment padding")
    return qweight, scales, zeros


def dequantize_awq(
    qweight: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int,
    *,
    dtype: str | np.dtype | type = bfloat16,
) -> np.ndarray:
    """Dequantize logical tensors as ``BF16(qweight - zero) * scale``."""

    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(f"group_size must be one of {SUPPORTED_GROUP_SIZES}")
    if not isinstance(qweight, np.ndarray) or qweight.ndim != 2:
        raise TypeError("qweight must be a rank-2 numpy.ndarray")
    K, N = qweight.shape
    if K % group_size:
        raise ValueError("group_size must divide qweight.shape[0]")
    qweight = _require_array("qweight", qweight, (K, N), np.uint8)
    scales = _require_array("scales", scales, (K // group_size, N), bfloat16)
    zeros = _require_array("zeros", zeros, (K // group_size, N), np.uint8)
    _require_uint4("qweight", qweight)
    _require_uint4("zeros", zeros)

    group_index = np.arange(K) // group_size
    result = (
        qweight.astype(np.int16) - zeros[group_index].astype(np.int16)
    ).astype(np.float32) * scales[group_index].astype(np.float32)
    return result.astype(np.dtype(dtype))


def make_deterministic_inputs(
    config: AWQConfig, *, seed: int = 1726250518
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic BF16 A, uint4 weights/zeros, and BF16 scales."""

    rng = np.random.default_rng(seed)
    A = rng.uniform(-0.25, 0.25, size=(config.M, config.K)).astype(bfloat16)
    qweight = rng.integers(0, 16, size=(config.K, config.N), dtype=np.uint8)
    zeros = rng.integers(
        0,
        16,
        size=(config.K // config.group_size, config.N),
        dtype=np.uint8,
    )
    scales = rng.uniform(
        0.005,
        0.05,
        size=(config.K // config.group_size, config.N),
    ).astype(bfloat16)
    return A, qweight, scales, zeros


def reference_matmul(
    A: np.ndarray,
    qweight: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int,
    *,
    dtype_out: str | np.dtype | type = "bf16",
    tile_k: int | None = None,
) -> np.ndarray:
    """Full CPU reference matching dequantization and K-tile store boundaries."""

    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise TypeError("A must be a rank-2 numpy.ndarray")
    if not isinstance(qweight, np.ndarray) or qweight.ndim != 2:
        raise TypeError("qweight must be a rank-2 numpy.ndarray")
    M, K = A.shape
    if K != qweight.shape[0]:
        raise ValueError("A.shape[1] must equal qweight.shape[0]")
    A = _require_array("A", A, (M, K), bfloat16)
    tile_k = _reference_tile_k(K, tile_k)
    out_name = _normalize_output_dtype(dtype_out)
    dequant = dequantize_awq(qweight, scales, zeros, group_size, dtype=bfloat16)
    result = np.zeros((M, qweight.shape[1]), dtype=np.float32)
    for k0 in range(0, K, tile_k):
        result += (
            A[:, k0 : k0 + tile_k].astype(np.float32)
            @ dequant[k0 : k0 + tile_k].astype(np.float32)
        )
        if out_name == "bf16":
            result = result.astype(bfloat16).astype(np.float32)
    return result.astype(bfloat16 if out_name == "bf16" else np.float32)


def reference_samples(
    A: np.ndarray,
    qweight: np.ndarray,
    scales: np.ndarray,
    zeros: np.ndarray,
    group_size: int,
    indices: Iterable[tuple[int, int]],
    *,
    dtype_out: str | np.dtype | type = "f32",
    tile_k: int | None = None,
) -> dict[tuple[int, int], np.float32]:
    """Evaluate selected output coordinates without materializing full C."""

    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise TypeError("A must be a rank-2 numpy.ndarray")
    if not isinstance(qweight, np.ndarray) or qweight.ndim != 2:
        raise TypeError("qweight must be a rank-2 numpy.ndarray")
    M, K = A.shape
    if K != qweight.shape[0]:
        raise ValueError("A.shape[1] must equal qweight.shape[0]")
    A = _require_array("A", A, (M, K), bfloat16)
    tile_k = _reference_tile_k(K, tile_k)
    out_name = _normalize_output_dtype(dtype_out)
    dequant = dequantize_awq(qweight, scales, zeros, group_size, dtype=bfloat16)
    result: dict[tuple[int, int], np.float32] = {}
    for row, col in indices:
        if not 0 <= row < A.shape[0] or not 0 <= col < qweight.shape[1]:
            raise IndexError(f"sample ({row}, {col}) is outside output shape")
        stored = np.float32(0)
        for k0 in range(0, K, tile_k):
            stored += np.dot(
                A[row, k0 : k0 + tile_k].astype(np.float32),
                dequant[k0 : k0 + tile_k, col].astype(np.float32),
            ).astype(np.float32)
            if out_name == "bf16":
                stored = np.float32(bfloat16(stored))
        result[(row, col)] = stored
    return result


def _reference_tile_k(K: int, tile_k: int | None) -> int:
    if tile_k is None:
        return K
    if not isinstance(tile_k, int) or isinstance(tile_k, bool) or tile_k <= 0:
        raise ValueError("tile_k must be a positive integer")
    if K % tile_k:
        raise ValueError("tile_k must divide K")
    return tile_k
