# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from packing import (  # noqa: E402
    AWQConfig,
    a_transfer_taps,
    b_partition_taps,
    c_transfer_taps,
    dequantize_awq,
    join_a_microtile_halves,
    pack_awq_weights,
    reference_matmul,
    reference_samples,
    split_a_tile_to_microtiles,
    unpack_awq_weights,
)


class PackingTests(unittest.TestCase):
    def _config(self, group: int, cols: int) -> AWQConfig:
        return AWQConfig(
            M=256,
            K=128,
            N=64 * cols,
            m=32,
            k=128,
            n=64,
            group_size=group,
            n_aie_cols=cols,
        )

    def test_round_trip_all_groups_and_columns(self):
        rng = np.random.default_rng(7)
        for group in (32, 64, 128):
            for cols in (1, 2, 4, 8):
                with self.subTest(group=group, cols=cols):
                    config = self._config(group, cols)
                    q = rng.integers(0, 16, (config.K, config.N), dtype=np.uint8)
                    s = rng.uniform(0.01, 0.1, (config.K // group, config.N)).astype(
                        bfloat16
                    )
                    z = rng.integers(
                        0, 16, (config.K // group, config.N), dtype=np.uint8
                    )
                    packed = pack_awq_weights(q, s, z, config)
                    actual_q, actual_s, actual_z = unpack_awq_weights(packed, config)
                    np.testing.assert_array_equal(actual_q, q)
                    np.testing.assert_array_equal(actual_s, s)
                    np.testing.assert_array_equal(actual_z, z)
                    self.assertEqual(packed.size, config.packed_bytes)

    def test_known_nibble_and_parameter_order(self):
        config = self._config(32, 1)
        q = np.zeros((config.K, config.N), dtype=np.uint8)
        q[:8, :8] = np.arange(64, dtype=np.uint8).reshape(8, 8) & 15
        scales = np.arange(
            (config.K // 32) * config.N, dtype=np.float32
        ).reshape(config.K // 32, config.N).astype(bfloat16)
        zeros = (
            np.arange((config.K // 32) * config.N, dtype=np.uint16)
            .reshape(config.K // 32, config.N)
            .astype(np.uint8)
            & 15
        )
        packed = pack_awq_weights(q, scales, zeros, config)

        expected_first_microtile = (
            q[:8, :8][:, 0::2] | (q[:8, :8][:, 1::2] << 4)
        ).reshape(-1)
        np.testing.assert_array_equal(packed[:32], expected_first_microtile)
        scale_start = config.weights_bytes_per_tile
        np.testing.assert_array_equal(
            packed[scale_start : scale_start + config.scales_bytes_per_tile],
            scales.view(np.uint8).reshape(-1),
        )
        zero_start = scale_start + config.scales_bytes_per_tile
        np.testing.assert_array_equal(packed[zero_start : zero_start + 8], zeros[0, :8])
        np.testing.assert_array_equal(
            packed[zero_start + 8 : zero_start + 16], zeros[0, :8]
        )
        self.assertFalse(np.any(packed[config.raw_tile_bytes : config.tile_bytes]))

    def test_known_dequantization(self):
        q = np.array([[0, 15], [7, 3], [8, 1], [2, 14]], dtype=np.uint8)
        scales = np.array([[0.5, 2.0]], dtype=bfloat16)
        zeros = np.array([[4, 1]], dtype=np.uint8)
        # Pad K to a supported group without changing the known first rows.
        q_pad = np.zeros((32, 2), dtype=np.uint8)
        q_pad[:4] = q
        actual = dequantize_awq(q_pad, scales, zeros, 32, dtype=np.float32)[:4]
        expected = (q.astype(np.int16) - zeros.astype(np.int16)) * scales.astype(
            np.float32
        )
        np.testing.assert_array_equal(actual, expected)

    def test_bf16_reference_rounds_at_k_tile_boundaries(self):
        rng = np.random.default_rng(11)
        A = rng.uniform(-1.0, 1.0, (2, 64)).astype(bfloat16)
        q = rng.integers(0, 16, (64, 3), dtype=np.uint8)
        scales = rng.uniform(0.01, 0.2, (2, 3)).astype(bfloat16)
        zeros = rng.integers(0, 16, (2, 3), dtype=np.uint8)

        full = reference_matmul(
            A, q, scales, zeros, 32, dtype_out="bf16", tile_k=32
        )
        sampled = reference_samples(
            A,
            q,
            scales,
            zeros,
            32,
            [(0, 1), (1, 2)],
            dtype_out="bf16",
            tile_k=32,
        )
        self.assertEqual(sampled[(0, 1)], np.float32(full[0, 1]))
        self.assertEqual(sampled[(1, 2)], np.float32(full[1, 2]))

    def test_validation_failures(self):
        with self.assertRaisesRegex(ValueError, "group_size must divide k"):
            AWQConfig(K=192, k=96, group_size=64)
        with self.assertRaisesRegex(ValueError, "N must be divisible"):
            AWQConfig(N=1000)
        with self.assertRaisesRegex(ValueError, "per-core memory"):
            AWQConfig(M=2048, K=2048, N=2048, m=128, k=256, n=128)
        with self.assertRaisesRegex(ValueError, "10-bit descriptor"):
            AWQConfig(M=256, K=128 * 1024, N=64, m=32, k=128, n=64,
                      n_aie_cols=1)

        config = self._config(128, 1)
        q = np.zeros((config.K, config.N), dtype=np.uint8)
        s = np.ones((1, config.N), dtype=bfloat16)
        z = np.zeros((1, config.N), dtype=np.uint8)
        q[0, 0] = 16
        with self.assertRaisesRegex(ValueError, "0..15"):
            pack_awq_weights(q, s, z, config)
        with self.assertRaisesRegex(TypeError, "dtype"):
            pack_awq_weights(q.astype(np.int8), s, z, config)

    def test_tap_coverage_and_order(self):
        for cols in (1, 2, 4, 8):
            with self.subTest(cols=cols):
                config = self._config(128, cols)
                b_taps = b_partition_taps(config)
                self.assertEqual([tap.column for tap in b_taps], list(range(cols)))
                b_indices = [index for tap in b_taps for index in tap.indices()]
                self.assertEqual(len(b_indices), config.packed_bytes)
                self.assertEqual(sorted(b_indices), list(range(config.packed_bytes)))

                a_taps = a_transfer_taps(config)
                a_unique = set(index for tap in a_taps for index in tap.indices())
                self.assertEqual(a_unique, set(range(config.M * config.K)))

                c_taps = c_transfer_taps(config)
                c_indices = [index for tap in c_taps for index in tap.indices()]
                self.assertEqual(len(c_indices), config.M * config.N)
                self.assertEqual(sorted(c_indices), list(range(config.M * config.N)))

    def test_a_half_microtile_order(self):
        a_tile = np.arange(64 * 128, dtype=np.float32).reshape(64, 128).astype(
            bfloat16
        )
        first, second = split_a_tile_to_microtiles(a_tile)
        np.testing.assert_array_equal(first[:64], a_tile[:8, :8].reshape(-1))
        np.testing.assert_array_equal(second[:64], a_tile[32:40, :8].reshape(-1))
        np.testing.assert_array_equal(
            join_a_microtile_halves(first, second, m=64, k=128), a_tile
        )

if __name__ == "__main__":
    unittest.main()
