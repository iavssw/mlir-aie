# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import unittest
from pathlib import Path

import numpy as np

WHOLE_ARRAY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WHOLE_ARRAY))

from whole_array import generate_taps  # noqa: E402


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
