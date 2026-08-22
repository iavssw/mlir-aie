# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parents[1]
PARENT = HERE.parent
for directory in (str(HERE), str(PARENT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

from packing import Q4KSConfig, decode_q4_k, make_deterministic_native_q4_k
from whole_array import generate_taps
from systolic_packing import (
    SystolicConfig,
    cascade_c_indices,
    cascade_token_sequence,
    decode_prepared_bfp,
    logical_tap_indices,
    phase_lifetimes,
    prepare_systolic_q4ks_weights,
    tile_coordinates,
    unpack_systolic_q4,
    weight_forward_sequence,
)


def _inputs(M: int = 256, K: int = 2048, N: int = 256):
    source = Q4KSConfig(
        M=M,
        K=K,
        N=N,
        m_c=32,
        m_a=16,
        k=256,
        n=16,
        n_aie_cols=8,
        compute_type="bfp16",
        accumulation_mode="bf16",
        cache_mode="l1-weight",
    )
    return make_deterministic_native_q4_k(source, seed=0x53595354)


class SystolicPackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A, cls.native = _inputs()

    def test_q4_round_trip_for_all_n_tiles(self):
        native_q, native_s, native_b = decode_q4_k(
            self.native, K=2048, N=256
        )
        for n in (16, 32, 64):
            with self.subTest(n=n):
                flow = "q4-direct" if n == 64 else "q4-local"
                cfg = SystolicConfig(
                    M=256, K=2048, N=256, n=n, weight_flow=flow
                )
                prepared = prepare_systolic_q4ks_weights(self.native, cfg)
                q, scales, biases = unpack_systolic_q4(prepared, cfg)
                np.testing.assert_array_equal(q, native_q)
                np.testing.assert_array_equal(
                    scales, native_s.astype(bfloat16)
                )
                np.testing.assert_array_equal(
                    biases, native_b.astype(bfloat16)
                )
                self.assertEqual(prepared.size, cfg.prepared_bytes)

    def test_stage_then_panel_order(self):
        cfg = SystolicConfig(M=256, K=2048, N=256, n=32)
        coordinates = list(tile_coordinates(cfg))
        self.assertEqual(coordinates[: cfg.n_panels], [(0, p) for p in range(8)])
        self.assertEqual(coordinates[cfg.n_panels], (1, 0))
        self.assertEqual(coordinates[-1], (7, 7))

    def test_bfp_ceiling_matches_q4_bf16_then_bfp(self):
        cfg = SystolicConfig(
            M=256,
            K=2048,
            N=256,
            n=32,
            weight_flow="bfp-prepared",
        )
        prepared = prepare_systolic_q4ks_weights(self.native, cfg)
        decoded = decode_prepared_bfp(prepared, cfg)
        self.assertEqual(decoded.shape, (cfg.K, cfg.N))
        self.assertTrue(np.all(np.isfinite(decoded)))
        q, scales, biases = decode_q4_k(self.native, K=cfg.K, N=cfg.N)
        group = np.arange(cfg.K) // 32
        bf16_weights = (
            q.astype(np.float32)
            * scales[group].astype(bfloat16).astype(np.float32)
            - biases[group].astype(bfloat16).astype(np.float32)
        ).astype(bfloat16).astype(np.float32)
        # BFP16 has a seven-bit signed mantissa.  The normalized Q4_K weights
        # are small, so an absolute bound is more useful than relative error
        # around zero.
        self.assertLess(float(np.max(np.abs(decoded - bf16_weights))), 0.01)

    def test_tap_coverage_and_weight_replay(self):
        configs = (
            SystolicConfig(M=256, K=2048, N=256, n=32),
            SystolicConfig(
                M=256,
                K=2048,
                N=256,
                m_a=64,
                n=32,
                weight_flow="q4-direct",
            ),
        )
        for cfg in configs:
            with self.subTest(m_a=cfg.m_a):
                indices = logical_tap_indices(cfg)
                self.assertEqual(len(indices["A"]), cfg.M * cfg.K)
                self.assertEqual(sorted(indices["A"]), list(range(cfg.M * cfg.K)))
                self.assertEqual(len(indices["C"]), cfg.M * cfg.N)
                self.assertEqual(sorted(indices["C"]), list(range(cfg.M * cfg.N)))
                replay = cfg.M // cfg.m_wave
                counts = np.bincount(indices["B"], minlength=cfg.prepared_bytes)
                np.testing.assert_array_equal(
                    counts, np.full(cfg.prepared_bytes, replay)
                )
    def test_panel_slab_taps_are_large_and_cover_prepared_weights(self):
        cfg = SystolicConfig(
            M=256,
            K=2048,
            N=256,
            m_a=64,
            n=32,
            weight_flow="q4-direct",
            transport="core-stream",
            panel_slab=8,
        )
        _, b_taps, _ = generate_taps(
            M=cfg.M,
            K=cfg.K,
            N=cfg.N,
            m_a=cfg.m_a,
            n=cfg.n,
            weight_flow=cfg.weight_flow,
            transport=cfg.transport,
            panel_slab=cfg.panel_slab,
        )
        self.assertEqual(len(b_taps), 8)
        for tap in b_taps:
            self.assertEqual(tap.sizes, [1, 8, 80, 64])
            self.assertEqual(tap.strides, [0, 5120, 64, 1])
        np.testing.assert_array_equal(
            b_taps.access_count(), np.ones((1, cfg.prepared_bytes), np.int32)
        )

    def test_memtile_cache_replays_large_weight_slabs(self):
        cfg = SystolicConfig(
            M=1024,
            K=2048,
            N=512,
            m_a=64,
            n=32,
            weight_flow="q4-direct",
            transport="memtile-cache",
            panel_slab=8,
            c_panel_slab=2,
        )
        self.assertEqual(cfg.weight_replay_waves, 4)
        self.assertEqual(cfg.weight_cache_slabs, 2)

        a_taps, b_taps, c_taps = generate_taps(
            M=cfg.M,
            K=cfg.K,
            N=cfg.N,
            m_a=cfg.m_a,
            n=cfg.n,
            weight_flow=cfg.weight_flow,
            transport=cfg.transport,
            panel_slab=cfg.panel_slab,
            c_panel_slab=cfg.c_panel_slab,
        )
        self.assertEqual(len(a_taps), 256)
        self.assertEqual(len(b_taps), 16)
        self.assertEqual(len(c_taps), 128)
        for tap in b_taps:
            self.assertEqual(tap.sizes, [2, 4, 80, 64])
            self.assertEqual(tap.strides, [20480, 5120, 64, 1])
        np.testing.assert_array_equal(
            a_taps.access_count(),
            np.full((cfg.M, cfg.K), cfg.weight_cache_slabs, np.int32),
        )
        np.testing.assert_array_equal(
            b_taps.access_count(),
            np.ones((1, cfg.prepared_bytes), np.int32),
        )
        np.testing.assert_array_equal(
            c_taps.access_count(), np.ones((cfg.M, cfg.N), np.int32)
        )

        modeled = logical_tap_indices(cfg)
        self.assertEqual(len(modeled["A"]), cfg.M * cfg.K * 2)
        self.assertEqual(len(modeled["B"]), cfg.prepared_bytes)
        self.assertEqual(len(modeled["C"]), cfg.M * cfg.N)
        order = weight_forward_sequence(cfg)
        panels_per_wave = cfg.panel_slab * 8 * 4
        self.assertEqual(order[0][:2], (0, 0))
        self.assertEqual(order[panels_per_wave][:2], (1, 0))
        self.assertEqual(order[4 * panels_per_wave][:2], (0, 8))

    def test_selected_column_cache_budget_and_tap_coverage(self):
        cfg = SystolicConfig(
            M=128,
            K=4096,
            N=512,
            m_a=32,
            n=32,
            weight_flow="q4-expand-once",
            transport="q4-column-dequant-cache",
            panel_slab=16,
            c_panel_slab=4,
        )
        self.assertEqual(cfg.worst_core_memory_bytes, 64064)
        self.assertEqual(cfg.memtile_bytes, 475136)
        self.assertEqual(
            cfg.core_memory_components()["streamed FP32 partials"], 0
        )

        a_taps, b_taps, c_taps = generate_taps(
            M=cfg.M,
            K=cfg.K,
            N=cfg.N,
            m_a=cfg.m_a,
            n=cfg.n,
            weight_flow=cfg.weight_flow,
            transport=cfg.transport,
            panel_slab=cfg.panel_slab,
            c_panel_slab=cfg.c_panel_slab,
        )
        self.assertEqual((len(a_taps), len(b_taps), len(c_taps)), (8, 8, 4))
        self.assertEqual(b_taps[0].sizes, [4, 4, 160, 64])
        self.assertEqual(b_taps[0].strides, [40960, 10240, 64, 1])
        np.testing.assert_array_equal(
            a_taps.access_count(), np.ones((cfg.M, cfg.K), np.int32)
        )
        np.testing.assert_array_equal(
            b_taps.access_count(), np.ones((1, cfg.prepared_bytes), np.int32)
        )
        np.testing.assert_array_equal(
            c_taps.access_count(), np.ones((cfg.M, cfg.N), np.int32)
        )

    def test_column_cache_n64_no_spill_geometry(self):
        cfg = SystolicConfig(
            M=128,
            K=2048,
            N=1024,
            m_a=32,
            n=64,
            weight_flow="q4-expand-once",
            transport="q4-column-dequant-cache",
            panel_slab=2,
            c_panel_slab=2,
        )
        self.assertEqual(cfg.worst_core_memory_bytes, 37568)
        self.assertEqual(cfg.memtile_bytes, 151552)
        self.assertEqual(
            cfg.core_memory_components()["streamed FP32 partials"], 0
        )

        a_taps, b_taps, c_taps = generate_taps(
            M=cfg.M,
            K=cfg.K,
            N=cfg.N,
            m_a=cfg.m_a,
            n=cfg.n,
            weight_flow=cfg.weight_flow,
            transport=cfg.transport,
            panel_slab=cfg.panel_slab,
            c_panel_slab=cfg.c_panel_slab,
        )
        self.assertEqual((len(a_taps), len(b_taps), len(c_taps)), (64, 64, 32))
        self.assertEqual(b_taps[0].sizes, [1, 2, 160, 64])
        self.assertEqual(b_taps[0].strides, [0, 10240, 64, 1])
        np.testing.assert_array_equal(
            a_taps.access_count(), np.full((cfg.M, cfg.K), 8, np.int32)
        )
        np.testing.assert_array_equal(
            b_taps.access_count(), np.ones((1, cfg.prepared_bytes), np.int32)
        )
        np.testing.assert_array_equal(
            c_taps.access_count(), np.ones((cfg.M, cfg.N), np.int32)
        )

    def test_memtile_slab_sweeps_n_with_one_stationary_a_wave(self):
        cfg = SystolicConfig(
            M=128,
            K=2048,
            N=512,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            transport="memtile-slab",
            panel_slab=16,
            c_panel_slab=8,
        )
        a_taps, b_taps, c_taps = generate_taps(
            M=cfg.M,
            K=cfg.K,
            N=cfg.N,
            m_a=cfg.m_a,
            n=cfg.n,
            weight_flow=cfg.weight_flow,
            transport=cfg.transport,
            panel_slab=cfg.panel_slab,
            c_panel_slab=cfg.c_panel_slab,
        )

        # Each column receives A once and one contiguous 16-panel Q4 slab.
        # Each physical row uses one repeated C task to drain both C8 objects.
        self.assertEqual(len(a_taps), 8)
        self.assertEqual(len(b_taps), 8)
        self.assertEqual(len(c_taps), 4)
        for tap in b_taps:
            self.assertEqual(tap.sizes, [4, 4, 80, 64])
            self.assertEqual(tap.strides, [20480, 5120, 64, 1])
        for tap in c_taps:
            self.assertEqual(tap.sizes, [2, 1, 32, 256])
            self.assertEqual(tap.strides, [256, 0, 512, 1])
        np.testing.assert_array_equal(
            a_taps.access_count(), np.ones((cfg.M, cfg.K), np.int32)
        )
        np.testing.assert_array_equal(
            b_taps.access_count(), np.ones((1, cfg.prepared_bytes), np.int32)
        )
        np.testing.assert_array_equal(
            c_taps.access_count(), np.ones((cfg.M, cfg.N), np.int32)
        )

        modeled = logical_tap_indices(cfg)
        self.assertEqual(len(modeled["A"]), cfg.M * cfg.K)
        self.assertEqual(len(modeled["B"]), cfg.prepared_bytes)
        self.assertEqual(len(modeled["C"]), cfg.M * cfg.N)

        multi_a, multi_b, multi_c = generate_taps(
            M=128,
            K=2048,
            N=1024,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            transport="memtile-slab",
            panel_slab=16,
            c_panel_slab=8,
        )
        self.assertEqual(len(multi_a), 8)
        self.assertEqual(len(multi_b), 8)
        self.assertEqual(len(multi_c), 4)
        for tap in multi_b:
            self.assertEqual(tap.sizes, [8, 4, 80, 64])
            self.assertEqual(tap.strides, [20480, 5120, 64, 1])

        full = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            transport="memtile-slab",
            panel_slab=16,
            c_panel_slab=8,
        )
        self.assertEqual(full.memtile_bytes, 512 * 1024)

        direct_a = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            transport="memtile-slab-direct-a",
            panel_slab=16,
            c_panel_slab=8,
        )
        self.assertEqual(direct_a.memtile_bytes, full.memtile_bytes)

    def test_joint_slab_reuses_two_activation_waves(self):
        cfg = SystolicConfig(
            M=256,
            K=2048,
            N=1024,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            transport="joint-slab",
            panel_slab=16,
            c_panel_slab=8,
        )
        a_taps, b_taps, c_taps = generate_taps(
            M=cfg.M,
            K=cfg.K,
            N=cfg.N,
            m_a=cfg.m_a,
            n=cfg.n,
            weight_flow=cfg.weight_flow,
            transport=cfg.transport,
            panel_slab=cfg.panel_slab,
            c_panel_slab=cfg.c_panel_slab,
        )
        self.assertEqual(len(a_taps), 8)
        self.assertEqual(len(b_taps), 8)
        self.assertEqual(len(c_taps), 8)
        for tap in a_taps:
            self.assertEqual(tap.sizes, [8, 32, 128, 2])
            self.assertEqual(tap.strides, [65536, 2048, 2, 1])
        for tap in b_taps:
            self.assertEqual(tap.sizes, [8, 4, 80, 64])
            self.assertEqual(tap.strides, [20480, 5120, 64, 1])
        for tap in c_taps:
            self.assertEqual(tap.sizes, [2, 2, 32, 256])
            self.assertEqual(tap.strides, [131072, 256, 1024, 1])
        np.testing.assert_array_equal(
            a_taps.access_count(), np.ones((cfg.M, cfg.K), np.int32)
        )
        np.testing.assert_array_equal(
            b_taps.access_count(), np.ones((1, cfg.prepared_bytes), np.int32)
        )
        np.testing.assert_array_equal(
            c_taps.access_count(), np.ones((cfg.M, cfg.N), np.int32)
        )
        modeled = logical_tap_indices(cfg)
        self.assertEqual(len(modeled["A"]), cfg.M * cfg.K)
        self.assertEqual(len(modeled["B"]), cfg.prepared_bytes)
        self.assertEqual(len(modeled["C"]), cfg.M * cfg.N)

        full = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            transport="joint-slab",
            panel_slab=16,
            c_panel_slab=8,
        )
        self.assertEqual(full.memtile_bytes, 480 * 1024)

    def test_c_panel_slab_taps_are_large_and_cover_output(self):
        cfg = SystolicConfig(
            M=256,
            K=2048,
            N=256,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            transport="core-stream",
            c_panel_slab=8,
        )
        _, _, c_taps = generate_taps(
            M=cfg.M,
            K=cfg.K,
            N=cfg.N,
            m_a=cfg.m_a,
            n=cfg.n,
            weight_flow=cfg.weight_flow,
            transport=cfg.transport,
            c_panel_slab=cfg.c_panel_slab,
        )
        self.assertEqual(len(c_taps), 8)
        for tap in c_taps:
            self.assertEqual(tap.sizes, [2, 16, 16, 16])
            self.assertEqual(tap.strides, [4096, 16, 256, 1])
        np.testing.assert_array_equal(
            c_taps.access_count(), np.ones((cfg.M, cfg.N), np.int32)
        )

    def test_memory_budget_is_role_specific_and_fits(self):
        cfg = SystolicConfig(M=256, K=2048, N=256, n=32)
        self.assertLessEqual(cfg.worst_core_memory_bytes, 64 * 1024)
        self.assertLess(
            cfg.core_memory_bytes(cascade_role="middle"),
            cfg.core_memory_bytes(cascade_role="east"),
        )
        self.assertLessEqual(cfg.memtile_bytes, 512 * 1024)
        direct = SystolicConfig(
            M=4096, K=4096, N=4096, n=64, weight_flow="q4-direct"
        )
        self.assertLessEqual(direct.worst_core_memory_bytes, 64 * 1024)
        self.assertEqual(
            direct.core_memory_components()["expanded B BFP16 scratch"], 0
        )
        wide = SystolicConfig(
            M=4096, K=4096, N=4096, m_a=64, n=32, weight_flow="q4-direct"
        )
        self.assertEqual(wide.worst_core_memory_bytes, 65536)
        self.assertLessEqual(wide.memtile_bytes, 512 * 1024)
        slab = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=64,
            n=32,
            weight_flow="q4-direct",
            panel_slab=8,
        )
        self.assertEqual(slab.memtile_bytes, 344064)
        self.assertEqual(
            slab.memtile_components["double-buffered weight input"], 163840
        )
        self.assertEqual(
            slab.memtile_components["weight row children"], 81920
        )
        c_slab = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            c_panel_slab=8,
        )
        self.assertEqual(c_slab.worst_core_memory_bytes, 56576)
        self.assertEqual(c_slab.memtile_bytes, 227328)
        self.assertEqual(
            c_slab.memtile_components["double-buffered C output"], 32768
        )
        shim_stream = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=32,
            n=32,
            weight_flow="q4-direct",
            transport="shim-stream",
            panel_slab=2,
            c_panel_slab=8,
        )
        self.assertEqual(shim_stream.memtile_bytes, 196608)
        self.assertEqual(
            shim_stream.memtile_components["double-buffered weight input"], 0
        )
        self.assertEqual(
            shim_stream.memtile_components["weight row children"], 0
        )
        wide_c2 = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=64,
            n=32,
            weight_flow="q4-direct",
            c_panel_slab=2,
        )
        self.assertEqual(wide_c2.a_chunk_k, 16)
        self.assertEqual(wide_c2.worst_core_memory_bytes, 65536)
        self.assertEqual(wide_c2.memtile_bytes, 145408)
        cached = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=64,
            n=32,
            weight_flow="q4-direct",
            transport="memtile-cache",
            panel_slab=16,
        )
        self.assertEqual(cached.weight_replay_waves, 4)
        self.assertEqual(cached.worst_core_memory_bytes, 65536)
        self.assertEqual(cached.memtile_bytes, 262144)
        cached_c2 = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=64,
            n=32,
            weight_flow="q4-direct",
            transport="memtile-cache",
            panel_slab=16,
            c_panel_slab=2,
        )
        self.assertEqual(cached_c2.memtile_bytes, 278528)
        cached_32 = SystolicConfig(
            M=4096,
            K=4096,
            N=4096,
            m_a=64,
            n=32,
            weight_flow="q4-direct",
            transport="memtile-cache",
            panel_slab=32,
            c_panel_slab=2,
        )
        self.assertEqual(cached_32.weight_cache_slabs, 4)
        self.assertEqual(cached_32.memtile_bytes, 442368)
        with self.assertRaisesRegex(ValueError, "MemTile"):
            SystolicConfig(
                M=4096,
                K=4096,
                N=4096,
                m_a=64,
                n=32,
                weight_flow="q4-direct",
                panel_slab=16,
            )

    def test_validation_failures(self):
        invalid = [
            dict(M=129),
            dict(K=4096 + 256),
            dict(N=257),
            dict(n=48),
            dict(m_a=48),
            dict(m_a=64),
            dict(m_a=64, n=64, weight_flow="q4-direct"),
            dict(weight_flow="unknown"),
            dict(transport="unknown"),
            dict(panel_slab=3, weight_flow="q4-direct"),
            dict(panel_slab=16, weight_flow="q4-direct"),
            dict(panel_slab=2),
            dict(panel_slab=2, weight_flow="q4-direct", transport="tile-dma"),
            dict(
                weight_flow="q4-direct",
                transport="memtile-cache",
                panel_slab=1,
            ),
            dict(
                weight_flow="q4-direct",
                transport="memtile-cache",
                panel_slab=2,
                c_panel_slab=4,
            ),
            dict(c_panel_slab=3, weight_flow="q4-direct"),
            dict(c_panel_slab=2),
            dict(weight_flow="q4-expand-once", transport="tile-dma"),
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                values = dict(M=256, K=2048, N=256)
                values.update(kwargs)
                SystolicConfig(**values)

    def test_weight_forward_and_four_row_coverage(self):
        cfg = SystolicConfig(M=256, K=2048, N=256, n=32)
        sequence = weight_forward_sequence(cfg)
        self.assertEqual(
            sequence[:8],
            [
                (0, 0, 0, 0),
                (0, 0, 0, 1),
                (0, 0, 0, 2),
                (0, 0, 0, 3),
                (0, 0, 1, 0),
                (0, 0, 1, 1),
                (0, 0, 1, 2),
                (0, 0, 1, 3),
            ],
        )
        expected = (cfg.M // cfg.m_wave) * cfg.n_panels
        counts = {
            (stage, row): sum(
                entry[2:] == (stage, row) for entry in sequence
            )
            for stage in range(8)
            for row in range(4)
        }
        self.assertEqual(set(counts.values()), {expected})

    def test_cascade_token_offsets_and_c_reconstruction(self):
        cases = (
            (32, 16, "q4-local"),
            (32, 32, "q4-local"),
            (32, 64, "q4-direct"),
            (64, 32, "q4-direct"),
        )
        for m_a, n, flow in cases:
            with self.subTest(m_a=m_a, n=n):
                cfg = SystolicConfig(
                    M=256, K=2048, N=256, m_a=m_a, n=n, weight_flow=flow
                )
                tokens = [
                    token
                    for token in cascade_token_sequence(cfg)
                    if token[:3] == (0, 0, 0)
                ]
                occupied = []
                for _, _, _, _, _, offset in tokens:
                    for row in range(8):
                        occupied.extend(
                            range(offset + row * 16, offset + row * 16 + 8)
                        )
                self.assertEqual(
                    sorted(occupied), list(range(cfg.m_a * cfg.n))
                )
                indices = cascade_c_indices(cfg)
                self.assertEqual(len(indices), cfg.M * cfg.N)
                self.assertEqual(
                    sorted(indices), list(range(cfg.M * cfg.N))
                )

    def test_phase_overlay_and_transport_memory(self):
        lifetimes = phase_lifetimes()
        self.assertLessEqual(
            lifetimes["activation input"][1],
            lifetimes["packed weight input"][0],
        )
        self.assertLess(
            lifetimes["stationary activation"][0],
            lifetimes["expanded weight"][1],
        )
        streamed = SystolicConfig(
            M=256, K=2048, N=256, n=32, transport="core-stream"
        )
        dma = SystolicConfig(
            M=256, K=2048, N=256, n=32, transport="tile-dma"
        )
        self.assertEqual(
            dma.memtile_bytes - streamed.memtile_bytes,
            3 * streamed.runtime_tile_bytes,
        )

    def test_real_core_memory_overflow_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "bytes/core"):
            SystolicConfig(M=256, K=6144, N=256, n=32)
        with self.assertRaisesRegex(ValueError, "bytes/core"):
            SystolicConfig(
                M=4096,
                K=4096,
                N=4096,
                m_a=64,
                n=32,
                weight_flow="q4-direct",
                c_panel_slab=4,
            )


if __name__ == "__main__":
    unittest.main()
