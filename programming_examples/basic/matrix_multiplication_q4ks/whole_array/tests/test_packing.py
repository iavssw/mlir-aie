# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import unittest
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from packing import (  # noqa: E402
    Q4KSConfig,
    Q4_K_BLOCK_BYTES,
    a_transfer_taps,
    b_partition_taps,
    bfp16ebs8_to_float,
    c_transfer_taps,
    decode_q4_k,
    dequantize_prepared_q4,
    encode_q4_k,
    float_to_bfp16ebs8,
    make_deterministic_native_q4_k,
    pack_scale_min_k4,
    prepare_q4ks_weights,
    quantize_activations_int8,
    reference_matmul,
    unpack_prepared_q4,
    unpack_scale_min_k4,
)


class Q4KNativeTests(unittest.TestCase):
    def test_llamacpp_reference_fixture(self):
        # Generated with llama.cpp quantize_row_q4_K_ref from
        # x[i] = sin(i*0.17)*2 + cos(i*0.031)*0.25, then independently
        # dequantized by llama.cpp dequantize_row_q4_K.
        native = np.frombuffer(
            bytes.fromhex(
                "b91c8728f9faf9faf2f7fcffbb372d8f"
                "28394a5b6c8d9eafbfcfdfefeefefdfc"
                "ebe9d8c7b6a493826151403020100001"
                "3020101101020304151628394a5b6c8d"
                "9eaebfcfdfeefefdfcfbfae9e8d6c5b"
                "4e9e8d6c5a4938271514030201011010"
                "20304151628394a6b7c8daebfcfdfeff"
                "faebecededeededecebead9d8c6b5a49"
                "372615140302010110102030415162738"
            ),
            dtype=np.uint8,
        )
        self.assertEqual(native.size, Q4_K_BLOCK_BYTES)
        q, scales, biases = decode_q4_k(native, K=256, N=1)
        actual = (
            q[:, 0].astype(np.float32)
            * scales[np.arange(256) // 32, 0]
            - biases[np.arange(256) // 32, 0]
        )
        expected = np.array(
            [
                0.334564209,
                0.597446442,
                0.860328674,
                1.12321091,
                1.38609314,
                1.64897537,
                1.91185760,
                2.17473984,
                2.17473984,
                2.17473984,
                2.17473984,
                2.17473984,
                1.91185760,
                1.91185760,
                1.64897537,
                1.38609314,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual[:16], expected, rtol=0, atol=1e-7)

    def test_scale_min_bit_layout(self):
        scales = np.array([0, 1, 15, 16, 31, 32, 47, 63], dtype=np.uint8)
        mins = np.array([63, 47, 32, 31, 16, 15, 1, 0], dtype=np.uint8)
        packed = pack_scale_min_k4(scales, mins)
        actual_scales, actual_mins = unpack_scale_min_k4(packed)
        np.testing.assert_array_equal(actual_scales, scales)
        np.testing.assert_array_equal(actual_mins, mins)
        self.assertEqual(packed.size, 12)

    def test_known_native_nibble_order_and_formula(self):
        K, N = 256, 1
        q = (np.arange(K, dtype=np.uint16) % 16).astype(np.uint8).reshape(K, N)
        scale_codes = np.arange(1, 9, dtype=np.uint8).reshape(8, 1)
        min_codes = np.arange(8, 0, -1, dtype=np.uint8).reshape(8, 1)
        d = np.array([[0.5]], dtype=np.float16)
        dmin = np.array([[0.25]], dtype=np.float16)
        native = encode_q4_k(q, scale_codes, min_codes, d, dmin)
        self.assertEqual(native.size, Q4_K_BLOCK_BYTES)
        decoded_q, scales, biases = decode_q4_k(native, K=K, N=N)
        np.testing.assert_array_equal(decoded_q, q)
        np.testing.assert_array_equal(scales[:, 0], scale_codes[:, 0] * 0.5)
        np.testing.assert_array_equal(biases[:, 0], min_codes[:, 0] * 0.25)
        # llama.cpp stores each 64-value region as 32 low then 32 high nibbles.
        expected_first_q_byte = int(q[0, 0] | (q[32, 0] << 4))
        self.assertEqual(int(native[16]), expected_first_q_byte)

    def test_rejects_bad_native_contract(self):
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            decode_q4_k(np.zeros(1, dtype=np.uint8), K=128, N=1)
        with self.assertRaisesRegex(ValueError, "shape"):
            decode_q4_k(np.zeros(143, dtype=np.uint8), K=256, N=1)
        with self.assertRaisesRegex(TypeError, "dtype"):
            decode_q4_k(np.zeros(144, dtype=np.int8), K=256, N=1)

    def test_rejects_out_of_range_encoder_values(self):
        q = np.zeros((256, 1), dtype=np.uint8)
        scales = np.ones((8, 1), dtype=np.uint8)
        mins = np.zeros((8, 1), dtype=np.uint8)
        d = np.ones((1, 1), dtype=np.float16)
        q[17, 0] = 16
        with self.assertRaisesRegex(ValueError, "0..15"):
            encode_q4_k(q, scales, mins, d, d)
        q[17, 0] = 0
        scales[3, 0] = 64
        with self.assertRaisesRegex(ValueError, "0..63"):
            encode_q4_k(q, scales, mins, d, d)


class PreparedLayoutTests(unittest.TestCase):
    def config(self, cols=8, **kwargs):
        values = dict(M=512, K=256, N=64 * cols, n_aie_cols=cols)
        values.update(kwargs)
        return Q4KSConfig(**values)

    def test_compressed_roundtrip_all_columns(self):
        for cols in (1, 2, 4, 8):
            with self.subTest(cols=cols):
                cfg = self.config(cols)
                _, native = make_deterministic_native_q4_k(cfg)
                expected_q, expected_s, expected_b = decode_q4_k(
                    native, K=cfg.K, N=cfg.N
                )
                packed = prepare_q4ks_weights(native, cfg, "q4")
                self.assertEqual(packed.size, cfg.prepared_bytes)
                actual_q, actual_s, actual_b = unpack_prepared_q4(packed, cfg)
                np.testing.assert_array_equal(actual_q, expected_q)
                np.testing.assert_array_equal(actual_s, expected_s.astype(bfloat16))
                np.testing.assert_array_equal(actual_b, expected_b.astype(bfloat16))

    def test_default_tile_sizes_and_expanded_formats(self):
        cfg = self.config(8)
        _, native = make_deterministic_native_q4_k(cfg)
        self.assertEqual(cfg.weights_bytes_per_tile, 4096)
        self.assertEqual(cfg.metadata_bytes_per_tile, 1024)
        self.assertEqual(cfg.tile_bytes, 5120)
        expected = {
            "q4": 5120,
            "bf16": 16384,
            "bfp16": 9216,
            "int8": 9216,
        }
        for storage, tile_bytes in expected.items():
            with self.subTest(storage=storage):
                prepared = prepare_q4ks_weights(native, cfg, storage)
                self.assertEqual(prepared.size, cfg.n_tiles * tile_bytes)

    def test_generated_inputs_have_llm_normalization(self):
        cfg = self.config(1)
        A, native = make_deterministic_native_q4_k(cfg)
        activation = A.astype(np.float32)
        np.testing.assert_allclose(activation.mean(axis=1), 0, atol=0.002)
        np.testing.assert_allclose(
            np.sqrt(np.mean(activation * activation, axis=1)),
            1,
            rtol=0,
            atol=0.003,
        )

        q, scales, biases = decode_q4_k(native, K=cfg.K, N=cfg.N)
        groups = np.arange(cfg.K) // 32
        weights = (
            q.astype(np.float32) * scales[groups] - biases[groups]
        )
        column_l2 = np.sqrt(np.sum(weights * weights, axis=0))
        np.testing.assert_allclose(column_l2, 1, rtol=0.04, atol=0)
        np.testing.assert_allclose(weights.mean(axis=0), 0, atol=0.004)

    def test_bfp_encoding_roundtrip(self):
        values = np.array(
            [-2.0, -1.0, -0.5, -0.125, 0.0, 0.25, 0.5, 1.5],
            dtype=np.float32,
        )
        packed = float_to_bfp16ebs8(values)
        self.assertEqual(packed.size, 9)
        decoded = bfp16ebs8_to_float(packed)
        np.testing.assert_allclose(decoded, values, atol=1 / 32, rtol=0)

    def test_prepared_bfp_microtile_orientation(self):
        cfg = self.config(1)
        _, native = make_deterministic_native_q4_k(cfg)
        q4 = prepare_q4ks_weights(native, cfg, "q4")
        expected = dequantize_prepared_q4(q4, cfg)[:8, :8].astype(np.float32)
        bfp = prepare_q4ks_weights(native, cfg, "bfp16")
        actual = bfp16ebs8_to_float(bfp[:72]).reshape(8, 8).T
        np.testing.assert_allclose(actual, expected, rtol=0.02, atol=0.03)

    def test_tap_partition_coverage(self):
        cfg = self.config(8)
        b_taps = b_partition_taps(cfg)
        b_indices = [i for tap in b_taps for i in tap.indices()]
        self.assertEqual(sorted(b_indices), list(range(cfg.prepared_bytes)))
        self.assertEqual(len(a_transfer_taps(cfg)), 2 * 4)
        c_taps = c_transfer_taps(cfg)
        c_indices = [i for tap in c_taps for i in tap.indices()]
        self.assertEqual(sorted(c_indices), list(range(cfg.M * cfg.N)))

    def test_memory_validation_and_atb_tiles(self):
        baseline = self.config(8)
        self.assertEqual(baseline.core_memory_bytes, 61696)
        bfp_atb = self.config(
            8,
            M=1024,
            N=1024,
            m_c=128,
            m_a=32,
            k=64,
            n=128,
            compute_type="bfp16",
        )
        self.assertLessEqual(bfp_atb.core_memory_bytes, 64 * 1024)
        int8_atb = self.config(
            8,
            M=1024,
            N=1024,
            m_c=128,
            m_a=32,
            k=64,
            n=128,
            compute_type="int8",
        )
        self.assertLessEqual(int8_atb.core_memory_bytes, 64 * 1024)
        with self.assertRaisesRegex(ValueError, "bytes/core"):
            self.config(8, M=1024, N=1024, m_c=128, m_a=32, k=64, n=128, compute_type="bf16")

    def test_memtile_cache_budget(self):
        weight_cache = Q4KSConfig(
            M=1024,
            K=4096,
            N=2048,
            compute_type="bfp16",
            cache_mode="memtile-weight",
        )
        self.assertLessEqual(sum(weight_cache.memtile_components().values()), 512 * 1024)
        with self.assertRaisesRegex(ValueError, "bytes/MemTile"):
            Q4KSConfig(
                M=1024,
                K=4096,
                N=2048,
                compute_type="bfp16",
                cache_mode="joint-slab",
            )
        joint = Q4KSConfig(
            M=1024,
            K=4096,
            N=2048,
            compute_type="bfp16",
            cache_mode="joint-slab",
            cache_k=2048,
        )
        self.assertLessEqual(sum(joint.memtile_components().values()), 512 * 1024)


class ComputeReferenceTests(unittest.TestCase):
    def test_int8_activation_contract_and_reference(self):
        cfg = Q4KSConfig(M=512, K=256, N=64, n_aie_cols=1, compute_type="int8")
        A, native = make_deterministic_native_q4_k(cfg)
        A8, scales, sums = quantize_activations_int8(A)
        self.assertEqual(A8.dtype, np.int8)
        self.assertEqual(scales.shape, (cfg.M, cfg.K // 32))
        self.assertEqual(sums.shape, scales.shape)
        self.assertLessEqual(int(A8.max()), 127)
        self.assertGreaterEqual(int(A8.min()), -127)
        actual = reference_matmul(A, native, cfg, compute_type="int8")
        self.assertEqual(actual.shape, (cfg.M, cfg.N))
        self.assertEqual(actual.dtype, np.dtype(bfloat16))

    def test_prepared_dequant_matches_declared_contract(self):
        cfg = Q4KSConfig(M=512, K=256, N=64, n_aie_cols=1)
        _, native = make_deterministic_native_q4_k(cfg)
        q, scales, biases = decode_q4_k(native, K=cfg.K, N=cfg.N)
        packed = prepare_q4ks_weights(native, cfg)
        actual = dequantize_prepared_q4(packed, cfg)
        groups = np.arange(cfg.K) // 32
        expected = (
            q.astype(np.float32) * scales.astype(bfloat16)[groups].astype(np.float32)
            - biases.astype(bfloat16)[groups].astype(np.float32)
        ).astype(bfloat16)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
