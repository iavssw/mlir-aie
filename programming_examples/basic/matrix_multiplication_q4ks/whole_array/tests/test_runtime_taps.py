# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import unittest
from pathlib import Path

import numpy as np

WHOLE_ARRAY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WHOLE_ARRAY))

from whole_array import NPU_DMA_MAX_STRIDE, generate_taps  # noqa: E402


class RuntimeTapTests(unittest.TestCase):
    def taps(self, cache_mode):
        return generate_taps(
            M=512,
            K=256,
            N=512,
            m_c=64,
            m_a=32,
            k=128,
            n=64,
            n_aie_cols=8,
            compute_type="bf16",
            cache_mode=cache_mode,
            activation_input="bf16",
            cache_k=256,
        )

    def test_stream_coverage_and_weight_refills(self):
        a, b, c = self.taps("stream")
        np.testing.assert_array_equal(a.access_count(), np.ones((512, 256)))
        np.testing.assert_array_equal(b.access_count().ravel(), np.full((81920,), 2))
        np.testing.assert_array_equal(c.access_count(), np.ones((512, 512)))

    def test_paired_m64_full_pair_and_tail_coverage(self):
        for M, expected_b_replays in ((256, 1), (512, 1), (768, 2)):
            with self.subTest(M=M):
                a, b, c = generate_taps(
                    M=M,
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
                    activation_input="bf16",
                    cache_k=256,
                )
                expected_a = np.ones((M, 256), dtype=np.int64)
                if M > 256 and M % 512:
                    # The exact tail duplicates its own A half to preserve
                    # FIFO rates, but performs MMUL only once.
                    expected_a[-256:, :] = 2
                np.testing.assert_array_equal(a.access_count(), expected_a)
                np.testing.assert_array_equal(
                    b.access_count().ravel(),
                    np.full((163840,), expected_b_replays),
                )
                expected_c = np.ones((M, 1024), dtype=np.int64)
                if M > 256 and M % 512:
                    expected_c[-256:, :] = 2
                np.testing.assert_array_equal(c.access_count(), expected_c)

    def test_paired_m64_long_k_uses_one_inner_k_task_per_wave(self):
        # K=8192 exceeds the paired-wave shim stride and exercises the same
        # schedule as K=14336 without constructing the full production shape.
        a, b, c = generate_taps(
            M=512,
            K=8192,
            N=1024,
            m_c=64,
            m_a=32,
            k=64,
            n=128,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="l1-weight",
            activation_input="bf16",
            cache_k=8192,
        )
        self.assertEqual((len(a), len(b), len(c)), (4, 8, 8))
        for tap in a:
            self.assertEqual(tap.sizes, [1, 64, 128, 128])
            self.assertEqual(tap.strides[0:2], [0, 128])
        for tap in b:
            self.assertEqual(tap.sizes, [1, 64, 160, 64])
            self.assertEqual(tap.strides[1], 10240)

    def test_high_perf_256_m_and_n_fringe_coverage(self):
        a, b, c = generate_taps(
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
            activation_input="bf16",
            cache_k=256,
        )
        # Both N waves replay A once.  The internal 64-row fringe stream is
        # exact-size, so the last 256 rows are no longer duplicated.
        expected_a = np.full((768, 256), 2, dtype=np.int64)
        np.testing.assert_array_equal(a.access_count(), expected_a)
        np.testing.assert_array_equal(
            b.access_count().ravel(), np.full((204800,), 2)
        )
        np.testing.assert_array_equal(
            c.access_count(), np.ones((768, 1280), dtype=np.int64)
        )

        # Ten N panels are packed unevenly as 2/2/1/1/1/1/1/1 across columns.
        self.assertEqual(
            [tap.offset for tap in b[:8]],
            [0, 40960, 81920, 102400, 122880, 143360, 163840, 184320],
        )

    def test_high_perf_odd_full_wave_and_n_only_fringe(self):
        for M, N, a_replays, b_replays in (
            (1536, 1024, 1, 3),
            (1024, 1280, 2, 2),
        ):
            with self.subTest(M=M, N=N):
                a, b, c = generate_taps(
                    M=M,
                    K=256,
                    N=N,
                    m_c=128,
                    m_a=32,
                    k=64,
                    n=128,
                    n_aie_cols=8,
                    compute_type="bfp16",
                    accumulation_mode="bf16",
                    cache_mode="l1-weight",
                    activation_input="bf16",
                    cache_k=256,
                )
                np.testing.assert_array_equal(
                    a.access_count(), np.full((M, 256), a_replays)
                )
                np.testing.assert_array_equal(
                    b.access_count().ravel(),
                    np.full((N // 128 * 20480,), b_replays),
                )
                np.testing.assert_array_equal(c.access_count(), np.ones((M, N)))

    def test_high_perf_long_k_fringe_uses_bounded_exact_transfers(self):
        a, b, c = generate_taps(
            M=768,
            K=14336,
            N=1024,
            m_c=128,
            m_a=32,
            k=64,
            n=128,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="l1-weight",
            activation_input="bf16",
            cache_k=14336,
        )
        # A complete wave maps one contiguous 128-row region to each core,
        # avoiding the unencodable inter-half stride. The 256-row tail shares
        # each 128-row source across a core pair; each worker selects 64 rows.
        self.assertEqual((len(a), len(b), len(c)), (8, 16, 24))
        a_taps = list(a)
        b_taps = list(b)
        self.assertTrue(all(len(tap.sizes) == 3 for tap in a_taps))
        self.assertTrue(all(tap.sizes == [56, 128, 256] for tap in a_taps))
        self.assertTrue(
            all(tap.strides == [256, 14336, 1] for tap in a_taps)
        )
        self.assertEqual(
            [tap.offset for tap in a_taps],
            [
                0,
                128 * 14336,
                256 * 14336,
                384 * 14336,
                512 * 14336,
                512 * 14336,
                640 * 14336,
                640 * 14336,
            ],
        )
        self.assertTrue(all(tap.sizes == [1, 56, 320, 64] for tap in b_taps))
        self.assertTrue(
            all(max(tap.sizes) <= 1023 for tap in (*a_taps, *b_taps))
        )
        self.assertTrue(
            all(tap.sizes == [1, 4, 64, 128] for tap in list(c)[:16])
        )

        # If N is also wide, the inverse C-row stride would exceed 20 bits.
        # The consumer is then split into exact 64-row chunks whose only host
        # row stride is N itself.
        wide_a, wide_b, wide_c = generate_taps(
            M=768,
            K=14336,
            N=14336,
            m_c=128,
            m_a=32,
            k=64,
            n=128,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="l1-weight",
            activation_input="bf16",
            cache_k=14336,
        )
        self.assertEqual((len(wide_a), len(wide_b), len(wide_c)), (112, 224, 1344))
        self.assertTrue(all(tap.sizes == [1, 1, 64, 128] for tap in wide_c))
        self.assertLessEqual(
            max(max(tap.strides) for tap in wide_c), NPU_DMA_MAX_STRIDE
        )

    def test_memtile_weight_panel_order_and_single_load(self):
        a, b, c = self.taps("memtile-weight")
        np.testing.assert_array_equal(a.access_count(), np.ones((512, 256)))
        np.testing.assert_array_equal(b.access_count().ravel(), np.ones((81920,)))
        np.testing.assert_array_equal(c.access_count(), np.ones((512, 512)))
        self.assertEqual(len(b), 8)
        self.assertEqual([tap.offset for tap in b], [i * 10240 for i in range(8)])

    def test_large_memtile_weight_uses_bounded_replay_slabs(self):
        a, b, c = generate_taps(
            M=4096,
            K=256,
            N=1024,
            m_c=128,
            m_a=32,
            k=64,
            n=128,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="memtile-weight",
            activation_input="bf16",
            cache_k=256,
        )
        np.testing.assert_array_equal(a.access_count(), np.ones((4096, 256)))
        np.testing.assert_array_equal(
            b.access_count().ravel(), np.full((163840,), 2)
        )
        np.testing.assert_array_equal(c.access_count(), np.ones((4096, 1024)))
        self.assertEqual(len(b), 16)

    def test_large_c_join_is_split_below_dma_limit(self):
        _, _, c = generate_taps(
            M=4096,
            K=256,
            N=512,
            m_c=256,
            m_a=32,
            k=64,
            n=64,
            n_aie_cols=8,
            compute_type="bfp16",
            accumulation_mode="bf16",
            cache_mode="l1-weight",
            activation_input="bf16",
            cache_k=256,
        )
        np.testing.assert_array_equal(c.access_count(), np.ones((4096, 512)))
        for tap in c:
            self.assertLessEqual(max(tap.sizes), 1023)

    def test_large_hybrid_cascade_coverage_and_bounded_replay(self):
        a, b, c = generate_taps(
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
            activation_input="bf16",
            cache_k=4096,
        )
        a_count = a.access_count()
        b_count = b.access_count().ravel()
        c_count = c.access_count()
        self.assertTrue(np.all(a_count == 1))
        self.assertTrue(np.all(b_count == 1))
        self.assertTrue(np.all(c_count == 1))
        self.assertEqual(len(a), 4)
        self.assertEqual(len(b), 8)
        self.assertEqual(len(c), 16)
        self.assertEqual(
            [tap.offset for tap in b[:8]],
            [column * 163840 for column in range(8)],
        )
        self.assertEqual(
            [tap.offset for tap in a[:4]], [0, 128, 1048576, 1048704]
        )

    def test_large_hybrid_cascade_l1_weight_coverage(self):
        a, b, c = generate_taps(
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
            cache_mode="l1-weight",
            activation_input="bf16",
            cache_k=4096,
        )
        np.testing.assert_array_equal(
            a.access_count(), np.ones((2048, 4096))
        )
        np.testing.assert_array_equal(
            b.access_count().ravel(), np.full((1310720,), 4)
        )
        np.testing.assert_array_equal(
            c.access_count(), np.ones((2048, 512))
        )
        self.assertEqual((len(a), len(b), len(c)), (16, 32, 16))
        self.assertEqual(
            [tap.offset for tap in a[:4]], [0, 128, 1048576, 1048704]
        )
        self.assertEqual(
            [tap.offset for tap in b[:4]], [0, 163840, 327680, 491520]
        )

    def test_hybrid_middle_worker_balances_fifo_acquire_release(self):
        source = (WHOLE_ARRAY / "whole_array.py").read_text()
        middle = source.split("def hybrid_middle_fn(", 1)[1].split(
            "def hybrid_top_fn(", 1
        )[0]
        self.assertEqual(middle.count("elem_a = in_a.acquire(1)"), 2)
        self.assertEqual(middle.count("in_a.release(1)"), 2)

if __name__ == "__main__":
    unittest.main()
