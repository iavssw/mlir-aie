# Whole-array Q4_K implementation

See the [example README](../README.md) for the native llama.cpp ABI, prepared
tile layout, compute paths, cache status, Python/C++ workflows, verification,
tracing, and performance-selection protocol.

The implementation is split into:

- whole_array.py: JIT callable, whole-array schedules, direct runner, and
  verification/benchmark controls.
- q4ks_bf16.cc: the smaller BF16-specialized Chess translation unit.
- q4ks.cc: native BFP16 and grouped INT8 affine kernels.
- benchmark_sweep.py: capacity and hardware candidate selection.
- tests/: host ABI/packing/TAP tests and NPU2 compile/run lit coverage.
