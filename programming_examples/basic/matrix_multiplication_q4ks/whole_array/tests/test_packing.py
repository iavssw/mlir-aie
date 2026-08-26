# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace
import sys
import unittest
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from packing import (  # noqa: E402
    NPU_DMA_MAX_OUTER_SIZE,
    NPU_DMA_MAX_STRIDE,
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
    paired_m64_a_transfer_taps,
    paired_m64_c_transfer_taps,
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

    def test_high_perf_256_uneven_column_packing_roundtrip(self):
        cfg = Q4KSConfig(
            M=768,
            K=256,
            N=1280,
            m_c=128,
            m_a=32,
            k=64,
            n=128,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="l1-weight",
        )
        self.assertTrue(cfg.needs_high_perf_256_schedule)
        self.assertEqual(cfg.full_n_rounds, 1)
        self.assertEqual(cfg.tail_n_cols, 2)
        self.assertEqual(
            [cfg.n_panels_for_col(col) for col in range(8)],
            [2, 2, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(
            [cfg.column_panel_offset(col) for col in range(8)],
            [0, 2, 4, 5, 6, 7, 8, 9],
        )

        _, native = make_deterministic_native_q4_k(cfg)
        expected_q, expected_s, expected_b = decode_q4_k(
            native, K=cfg.K, N=cfg.N
        )
        packed = prepare_q4ks_weights(native, cfg, "q4")
        actual_q, actual_s, actual_b = unpack_prepared_q4(packed, cfg)
        np.testing.assert_array_equal(actual_q, expected_q)
        np.testing.assert_array_equal(actual_s, expected_s.astype(bfloat16))
        np.testing.assert_array_equal(actual_b, expected_b.astype(bfloat16))

        taps = b_partition_taps(cfg)
        self.assertEqual([tap.sizes[0] for tap in taps], [2, 2, 1, 1, 1, 1, 1, 1])
        indices = [index for tap in taps for index in tap.indices()]
        self.assertEqual(sorted(indices), list(range(cfg.prepared_bytes)))

    def test_high_perf_fixed_tile_accepts_all_256_fringe_shapes(self):
        common = dict(
            K=256,
            m_c=128,
            m_a=32,
            k=64,
            n=128,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="l1-weight",
        )
        for M in (256, 512, 768, 1024, 1280, 8192):
            for N in (256, 512, 768, 1024, 1280, 32768):
                with self.subTest(M=M, N=N):
                    cfg = Q4KSConfig(M=M, N=N, **common)
                    self.assertTrue(cfg.supports_high_perf_256_contract)
        self.assertFalse(Q4KSConfig(M=512, N=1024, **common).needs_high_perf_256_schedule)
        self.assertTrue(Q4KSConfig(M=768, N=1024, **common).needs_high_perf_256_schedule)
        self.assertTrue(Q4KSConfig(M=1024, N=1280, **common).needs_high_perf_256_schedule)
        for K, expected_slab in (
            (4096, 1),
            (8192, 2),
            (14336, 4),
            (32768, 8),
        ):
            with self.subTest(K=K):
                cfg = Q4KSConfig(M=768, K=K, N=1024, **{
                    key: value for key, value in common.items() if key != "K"
                })
                self.assertEqual(cfg.dma_k_slab_tiles, expected_slab)
                self.assertLessEqual(
                    cfg.n_k_tiles // cfg.dma_k_slab_tiles,
                    NPU_DMA_MAX_OUTER_SIZE,
                )
                self.assertLessEqual(
                    cfg.dma_k_slab_tiles * cfg.packed_rows,
                    1023,
                )
        with self.assertRaisesRegex(ValueError, "divisible by m_c"):
            Q4KSConfig(M=384, N=1024, **common)
        with self.assertRaisesRegex(ValueError, "divisible by n"):
            Q4KSConfig(M=512, N=1152, **common)

    def test_paired_m64_tail_tap_coverage_and_order(self):
        cfg = Q4KSConfig(
            M=768,
            K=256,
            N=1024,
            m_c=64,
            m_a=32,
            k=64,
            n=128,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="l1-weight",
        )
        a_taps = paired_m64_a_transfer_taps(cfg)
        c_taps = paired_m64_c_transfer_taps(cfg)
        self.assertEqual(len(a_taps), 8)
        self.assertEqual(
            [tap.row_block for tap in a_taps],
            [0] * 4 + [2] * 4,
        )
        self.assertTrue(all(tap.sizes == (1, 4, 128, 64) for tap in a_taps[:4]))
        self.assertTrue(all(tap.sizes == (4, 128, 64) for tap in a_taps[4:]))
        a_indices = [i for tap in a_taps for i in tap.indices()]
        a_counts = np.bincount(a_indices, minlength=cfg.M * cfg.K).reshape(
            cfg.M, cfg.K
        )
        expected_a = np.ones((cfg.M, cfg.K), dtype=np.int64)
        expected_a[512:768, :] = 2
        np.testing.assert_array_equal(a_counts, expected_a)
        self.assertEqual(len(c_taps), 72)
        self.assertEqual(
            [tap.row_block for tap in c_taps],
            [0] * 8 + [2] * 64,
        )
        c_indices = [i for tap in c_taps for i in tap.indices()]
        c_counts = np.bincount(c_indices, minlength=cfg.M * cfg.N).reshape(
            cfg.M, cfg.N
        )
        expected_c = np.ones((cfg.M, cfg.N), dtype=np.int64)
        expected_c[512:768, :] = 2
        np.testing.assert_array_equal(c_counts, expected_c)

    def test_paired_m64_large_dimensions_use_compact_paired_taps(self):
        common = dict(
            M=512,
            m_c=64,
            m_a=32,
            k=64,
            n=128,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="l1-weight",
        )

        large_n = Q4KSConfig(K=4096, N=14336, **common)
        large_n_a = paired_m64_a_transfer_taps(large_n)
        large_n_c = paired_m64_c_transfer_taps(large_n)
        self.assertEqual((len(large_n_a), len(large_n_c)), (4, 8))
        self.assertTrue(all(tap.sizes[0] == 14 for tap in large_n_a))
        self.assertTrue(all(tap.sizes[1] == 64 for tap in large_n_a))
        self.assertTrue(all(tap.sizes[0] == 14 for tap in large_n_c))
        self.assertTrue(all(tap.sizes[1] == 8 for tap in large_n_c))
        self.assertTrue(
            all(tap.sizes[0] <= NPU_DMA_MAX_OUTER_SIZE for tap in large_n_a)
        )
        self.assertTrue(
            all(tap.strides[1] <= NPU_DMA_MAX_STRIDE for tap in large_n_c)
        )

        large_k = Q4KSConfig(K=14336, N=4096, **common)
        large_k_a = paired_m64_a_transfer_taps(large_k)
        large_k_c = paired_m64_c_transfer_taps(large_k)
        self.assertEqual((len(large_k_a), len(large_k_c)), (4, 8))
        self.assertTrue(
            all(tap.sizes[0] <= NPU_DMA_MAX_OUTER_SIZE for tap in large_k_a)
        )
        self.assertTrue(all(tap.sizes[0] == 4 for tap in large_k_a))
        self.assertTrue(
            all(
                tap.sizes[1]
                == large_k.n_k_tiles // large_k.dma_k_slab_tiles
                for tap in large_k_a
            )
        )
        self.assertTrue(
            all(
                tap.sizes[3] == large_k.dma_k_slab_tiles * large_k.k
                for tap in large_k_a
            )
        )
        self.assertTrue(
            all(tap.strides[1] <= NPU_DMA_MAX_STRIDE for tap in large_k_a)
        )

        # The odd 256-row tail duplicates a contiguous 128-row source range
        # across a row pair and must not encode the former 2*m_c*K-byte jump.
        large_k_tail = Q4KSConfig(K=14336, N=4096, M=768, **{
            key: value for key, value in common.items() if key != "M"
        })
        tail_a = paired_m64_a_transfer_taps(large_k_tail)[4:]
        self.assertEqual(len(tail_a), 4 * 4)
        self.assertTrue(
            all(
                tap.sizes
                == (
                    large_k_tail.n_k_tiles
                    // large_k_tail.dma_k_slab_tiles,
                    128,
                    large_k_tail.dma_k_slab_tiles * large_k_tail.k,
                )
                for tap in tail_a
            )
        )
        self.assertTrue(
            all(max(stride * 2 for stride in tap.strides) < NPU_DMA_MAX_STRIDE
                for tap in tail_a)
        )

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
        with self.assertRaisesRegex(ValueError, "ObjectFIFO deadlocks"):
            self.config(
                8,
                M=1024,
                K=256,
                N=1024,
                m_c=128,
                m_a=64,
                k=64,
                n=128,
                compute_type="bfp16",
            )
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

        large_k = self.config(
            8,
            M=1024,
            K=4096,
            N=1024,
            m_c=64,
            m_a=16,
            k=128,
            n=128,
            compute_type="bfp16",
        )
        self.assertEqual(large_k.a_fifo_depth, 2)
        self.assertEqual(large_k.c_fifo_depth, 1)
        self.assertEqual(
            large_k.core_memory_components["bfp16 weight scratch"], 18432
        )
        self.assertEqual(large_k.core_memory_bytes, 60672)

        paired = self.config(
            8,
            M=768,
            K=4096,
            N=4096,
            m_c=64,
            m_a=32,
            k=64,
            n=128,
            compute_type="bfp16",
            cache_mode="l1-weight",
        )
        self.assertTrue(paired.uses_paired_m64_schedule)
        self.assertEqual(paired.core_memory_bytes, 62720)
        self.assertEqual(
            paired.core_memory_components["C FIFO (depth 1)"], 32768
        )

        single = self.config(
            8,
            M=256,
            K=4096,
            N=4096,
            m_c=64,
            m_a=32,
            k=64,
            n=128,
            compute_type="bfp16",
            cache_mode="l1-weight",
        )
        self.assertTrue(single.uses_exact_single_m64_schedule)
        self.assertFalse(single.uses_paired_m64_schedule)
        self.assertEqual(
            single.core_memory_components["C FIFO (depth 1)"], 16384
        )

        single_m128 = self.config(
            8,
            M=512,
            K=4096,
            N=4096,
            m_c=128,
            m_a=32,
            k=64,
            n=128,
            compute_type="bfp16",
            cache_mode="l1-weight",
        )
        self.assertTrue(single_m128.uses_exact_single_row_wave)
        self.assertFalse(single_m128.uses_paired_m64_schedule)

        single_m64_a64 = self.config(
            8,
            M=256,
            K=4096,
            N=4096,
            m_c=64,
            m_a=64,
            k=64,
            n=128,
            compute_type="bfp16",
            cache_mode="l1-weight",
        )
        self.assertEqual(single_m64_a64.a_fifo_depth, 2)
        self.assertEqual(single_m64_a64.core_memory_bytes, 54528)

        low_overhead_m80 = self.config(
            8,
            M=1280,
            K=4096,
            N=4096,
            m_c=80,
            m_a=80,
            k=64,
            n=128,
            compute_type="bfp16",
            cache_mode="l1-weight",
        )
        self.assertEqual(low_overhead_m80.a_fifo_depth, 2)
        self.assertLessEqual(low_overhead_m80.core_memory_bytes, 64 * 1024)

        # k=512 is the largest compiled power-of-two configuration in the
        # performance sweep.  k=1024 already exceeds the AIE DMA's 10-bit
        # transfer dimension (and its useful shapes also exceed L1 budgets).
        with self.assertRaisesRegex(ValueError, "DMA size exceeds"):
            self.config(
                8,
                M=1024,
                K=4096,
                N=1024,
                m_c=128,
                m_a=16,
                k=1024,
                n=16,
                compute_type="bfp16",
            )

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

    def test_accumulation_memory_and_cascade_program_limit(self):
        local_fp32 = Q4KSConfig(
            M=512,
            K=256,
            N=512,
            m_c=64,
            m_a=32,
            k=64,
            n=32,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="fp32",
        )
        self.assertIn(
            "FP32 accumulation scratch", local_fp32.core_memory_components
        )
        cascade = Q4KSConfig(
            M=512,
            K=256,
            N=512,
            m_c=64,
            m_a=64,
            k=64,
            n=64,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="cascade",
        )
        self.assertEqual(cascade.a_fifo_depth, 2)
        with self.assertRaisesRegex(ValueError, "program memory"):
            Q4KSConfig(
                M=512,
                K=512,
                N=512,
                m_c=64,
                m_a=64,
                k=128,
                n=64,
                n_aie_cols=8,
                compute_type="bfp16",
                accumulation_mode="cascade",
            )

    def test_hybrid_cascade_memory_and_validation(self):
        hybrid = Q4KSConfig(
            M=2048,
            K=4096,
            N=512,
            m_c=256,
            m_a=32,
            k=128,
            n=64,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="cascade-hybrid",
            cache_mode="memtile-weight",
            cache_k=4096,
        )
        self.assertEqual(hybrid.a_fifo_depth, 1)
        self.assertLessEqual(hybrid.core_memory_bytes, 64 * 1024)
        self.assertLessEqual(
            sum(hybrid.memtile_components().values()), 512 * 1024
        )
        self.assertIn(
            "BF16 local K/2 partial", hybrid.core_memory_components
        )
        l1_hybrid = replace(hybrid, cache_mode="l1-weight")
        self.assertEqual(
            l1_hybrid.memtile_components()[
                "resident compressed-Q4_K panel"
            ],
            0,
        )
        self.assertEqual(
            l1_hybrid.memtile_components()[
                "streamed compressed-Q4_K panel"
            ],
            l1_hybrid.panel_bytes("q4"),
        )
        self.assertLessEqual(
            sum(l1_hybrid.memtile_components().values()), 512 * 1024
        )
        with self.assertRaisesRegex(
            ValueError, "requires l1-weight or memtile-weight"
        ):
            Q4KSConfig(
                M=2048,
                K=4096,
                N=512,
                m_c=256,
                m_a=32,
                k=128,
                n=64,
                n_aie_cols=8,
                compute_type="bfp16",
                accumulation_mode="cascade-hybrid",
                cache_mode="stream",
                cache_k=4096,
            )


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
