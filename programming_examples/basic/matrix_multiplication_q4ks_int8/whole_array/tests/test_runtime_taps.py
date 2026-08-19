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


if __name__ == "__main__":
    unittest.main()
