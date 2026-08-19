# Whole-array Q4_K implementation

See the [example README](../README.md) for the native llama.cpp ABI, prepared
tile layout, compute paths, cache status, Python/C++ workflows, verification,
tracing, and performance-selection protocol.

The implementation is split into:

- whole_array.py: JIT callable, whole-array schedules, direct runner, and
  verification/benchmark controls.
- q4ks_bf16.cc: the smaller BF16-specialized Chess translation unit.
- q4ks.cc: native BFP16, local-FP32, full-FP32 cascade, split-symbol hybrid
  cascade, and grouped INT8-affine kernels.
- benchmark_sweep.py: compute/cache/accumulation capacity and hardware
  candidate selection.
- tests/: host ABI/packing/TAP tests and NPU2 compile/run lit coverage.

The selected ceiling path is native BFP with BF16 K-tile writeback and the
`128/32 x 64 x 128` asymmetric tile. MemTile weight replay is executable but
bounded to at most four row-block repeats per slab; see the parent README for
the measured sustained results and `perf-*` targets.

The performance-cascade candidate is `cascade-hybrid` with the
`256/32 x 128 x 64` tile and `l1-weight` streamed-panel scheduling. Two
independent two-row chains keep local prefix partials in BF16, perform each
final K tile and the physical-cascade reduction in `accfloat`, and store BF16
once at each top row. Use `make perf-cascade-hybrid` for the direct
16,000-GFLOP/s research gate and `make perf-cascade-hybrid-3x` for the
three-round median gate. The current Default-mode result is 12.623 TOPS, so
the greater-than-16-TOPS cascade goal is not yet qualified.
