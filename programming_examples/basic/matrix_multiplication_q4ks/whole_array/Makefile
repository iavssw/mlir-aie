##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=whole_array
targetname=whole_array_q4ks
aie_py_src=whole_array.py

# Fixed high-performance native-Q4_K/BFP16 tile.  Aligned work uses the
# original 512-row by 1024-column whole-array wave.  An internal exact fringe
# schedule handles logical M and N in 256-element increments without changing
# this compile-time tile or padding the host tensors.  Command-line assignments
# still override every value because these are conditional defaults.
M?=4096
K?=4096
N?=4096

m_c?=128
m_a?=32
k?=64
n?=128
n_aie_cols?=8
compute_type?=bfp16

# Accumulation mode selects the compiled AIE topology; it is not a runtime
# switch within one xclbin.  The mode is included in the artifact filename,
# and makefile-common passes the same value to the compiler and C++ host.
#
#   bf16           Physical cascade OFF (default). Accumulate each K tile in
#                  accfloat, then store/reload BF16 between K tiles.
#   fp32           Physical cascade OFF. Keep the complete C tile in local
#                  FP32 scratch and convert to BF16 once at the end.
#   cascade        Physical cascade ON. Split K over all four AIE rows and
#                  forward accfloat values through the hardware cascade.
#   cascade-hybrid Physical cascade ON. Use two independent two-row chains;
#                  prefix partials use BF16 and the final reduction is accfloat.
#
# Use `make perf-cascade` or `make perf-cascade-hybrid` for their validated
# tile configurations.  Setting only accumulation_mode may be rejected when
# the other tile/cache constraints do not match the selected topology.
accumulation_mode?=bf16
cache_mode?=l1-weight
activation_input?=bf16

include ${srcdir}/../makefile-common

# Used by gen_pdi_single.sh so artifact discovery follows the Makefile's
# selected tile and modes.  This avoids duplicating an internal BUILD_DT label
# in the generator; generated PDI names expose the bf16_q4k_bf16 user contract.
.PHONY: print-target-suffix
print-target-suffix:
	@printf '%s\n' '${target_suffix}'

# Run the documented aligned BFP16 configuration with the primary
# warmup/iteration counts, sampled Q4_K verification, and the 16-TOPS gate.
.PHONY: perf-bfp16 perf-bfp16-raw perf-aligned-4096 perf-fringe-768-wide
perf-bfp16: warmup=16
perf-bfp16: iters=20
perf-bfp16: verify=true
perf-bfp16: verify_samples=1000
perf-bfp16: min_gflops=16000
perf-bfp16: run

# Optional throughput-only diagnostic. This uses the same workload but skips
# verification; it must not be reported as a correctness-qualified result.
perf-bfp16-raw: warmup=16
perf-bfp16-raw: iters=20
perf-bfp16-raw: verify=false
perf-bfp16-raw: min_gflops=16000
perf-bfp16-raw: run

# Explicit names for the aligned fast path and a representative M+N fringe.
# The fringe target has a lower gate because one third of its M work is the
# unavoidable 256-row half-wave; it still uses the same fixed compile-time tile.
perf-aligned-4096: perf-bfp16

perf-fringe-768-wide: M=768
perf-fringe-768-wide: K=4096
perf-fringe-768-wide: N=16384
perf-fringe-768-wide: warmup=16
perf-fringe-768-wide: iters=20
perf-fringe-768-wide: verify=true
perf-fringe-768-wide: verify_samples=1000
perf-fringe-768-wide: min_gflops=13000
perf-fringe-768-wide: run

# Safe mC/mA/n screening sweep for the k=64 native-BFP16 kernel.  These use
# short correctness-qualified runs; use perf-bfp16 for the formal winning
# configuration.  The 128/64x64x128 shape is deliberately omitted because its
# depth-one A ObjectFIFO was observed to wedge amdxdna on NPU2.
.PHONY: bench-bfp16-tile-sweep
bench-bfp16-tile-sweep:
	$(MAKE) run M=4096 K=4096 N=4096 m_c=128 m_a=32 k=64 n=128 warmup=2 iters=5 verify=true min_gflops=0
	$(MAKE) run M=4096 K=4096 N=4096 m_c=128 m_a=16 k=64 n=128 warmup=2 iters=5 verify=true min_gflops=0
	$(MAKE) run M=4096 K=4096 N=4096 m_c=64 m_a=32 k=64 n=128 warmup=2 iters=5 verify=true min_gflops=0
	$(MAKE) run M=4096 K=4096 N=4096 m_c=256 m_a=32 k=64 n=64 warmup=2 iters=5 verify=true min_gflops=0
	$(MAKE) run M=4096 K=4096 N=4096 m_c=32 m_a=32 k=64 n=256 warmup=2 iters=5 verify=true min_gflops=0

# BF16-writeback K-tile accuracy/performance sweep.  Larger K keeps each
# accfloat partial alive longer before BF16 writeback, but the 64-KiB core
# budget forces progressively narrower M/N tiles.  The k=64 target remains
# the throughput default; k=128 is the fastest measured larger-K option and
# k=512 is the maximum configuration that passes placement and Chess.
.PHONY: perf-bfp16-k128 perf-bfp16-k256 perf-bfp16-k512 perf-bfp16-k-sweep
perf-bfp16-k128: M=4096
perf-bfp16-k128: K=4096
perf-bfp16-k128: N=4096
perf-bfp16-k128: m_c=64
perf-bfp16-k128: m_a=16
perf-bfp16-k128: k=128
perf-bfp16-k128: n=128
perf-bfp16-k128: compute_type=bfp16
perf-bfp16-k128: accumulation_mode=bf16
perf-bfp16-k128: cache_mode=l1-weight
perf-bfp16-k128: warmup=16
perf-bfp16-k128: iters=20
perf-bfp16-k128: verify=true
perf-bfp16-k128: verify_samples=1000
perf-bfp16-k128: min_gflops=12000
perf-bfp16-k128: run

perf-bfp16-k256: M=4096
perf-bfp16-k256: K=4096
perf-bfp16-k256: N=4096
perf-bfp16-k256: m_c=128
perf-bfp16-k256: m_a=64
perf-bfp16-k256: k=256
perf-bfp16-k256: n=32
perf-bfp16-k256: compute_type=bfp16
perf-bfp16-k256: accumulation_mode=bf16
perf-bfp16-k256: cache_mode=l1-weight
perf-bfp16-k256: warmup=16
perf-bfp16-k256: iters=20
perf-bfp16-k256: verify=true
perf-bfp16-k256: verify_samples=1000
perf-bfp16-k256: min_gflops=8000
perf-bfp16-k256: run

perf-bfp16-k512: M=4096
perf-bfp16-k512: K=4096
perf-bfp16-k512: N=4096
perf-bfp16-k512: m_c=128
perf-bfp16-k512: m_a=16
perf-bfp16-k512: k=512
perf-bfp16-k512: n=16
perf-bfp16-k512: compute_type=bfp16
perf-bfp16-k512: accumulation_mode=bf16
perf-bfp16-k512: cache_mode=l1-weight
perf-bfp16-k512: warmup=16
perf-bfp16-k512: iters=20
perf-bfp16-k512: verify=true
perf-bfp16-k512: verify_samples=1000
perf-bfp16-k512: min_gflops=5000
perf-bfp16-k512: run

perf-bfp16-k-sweep:
	$(MAKE) perf-bfp16
	$(MAKE) perf-bfp16-k128
	$(MAKE) perf-bfp16-k256
	$(MAKE) perf-bfp16-k512

# Correctness-qualified latency sweeps for the first five 256-row increments.
# These deliberately use min_gflops=0: each M is judged by its own fastest
# attainable latency rather than failing short workloads against a 16-TOPS
# full-array gate. The non-default tiles are the measured small-M winners.
.PHONY: perf-small-long-k perf-small-wide-n
perf-small-long-k:
	$(MAKE) run M=256  K=14336 N=4096 m_c=64 m_a=16 k=128 n=128 warmup=4 iters=10 verify=true min_gflops=0
	$(MAKE) run M=512  K=14336 N=4096 m_c=64 m_a=32 k=64  n=128 warmup=4 iters=10 verify=true min_gflops=0
	$(MAKE) run M=768  K=14336 N=4096 m_c=96 m_a=48 k=64  n=128 warmup=4 iters=10 verify=true min_gflops=0
	$(MAKE) run M=1024 K=14336 N=4096 m_c=64 m_a=32 k=64  n=128 warmup=4 iters=10 verify=true min_gflops=0
	$(MAKE) run M=1280 K=14336 N=4096 m_c=64 m_a=32 k=64  n=128 warmup=4 iters=10 verify=true min_gflops=0

perf-small-wide-n:
	$(MAKE) run M=256  K=4096 N=14336 m_c=64 m_a=64 k=64 n=128 warmup=4 iters=10 verify=true min_gflops=0
	$(MAKE) run M=512  K=4096 N=14336 m_c=64 m_a=32 k=64 n=128 warmup=4 iters=10 verify=true min_gflops=0
	$(MAKE) run M=768  K=4096 N=14336 m_c=96 m_a=48 k=64 n=128 warmup=4 iters=10 verify=true min_gflops=0
	$(MAKE) run M=1024 K=4096 N=14336 m_c=64 m_a=32 k=64 n=128 warmup=4 iters=10 verify=true min_gflops=0
	$(MAKE) run M=1280 K=4096 N=14336 m_c=80 m_a=80 k=64 n=128 warmup=4 iters=10 verify=true min_gflops=0

# Stable name for the selected end-to-end ceiling candidate.  Put the device
# in Turbo mode separately when measuring the absolute clock ceiling.
.PHONY: perf-ceiling perf-ceiling-3x
perf-ceiling: perf-bfp16

perf-ceiling-3x:
	python3 ${srcdir}/benchmark_sweep.py --hardware -M 4096 -K 4096 -N 4096 --compute-types bfp16 --accumulation-modes bf16 --cache-modes l1-weight --warmup 16 --iters 20 --rounds 3 --min-gflops 12000

.PHONY: perf-fp32 perf-cascade perf-cascade-hybrid perf-cascade-hybrid-3x perf-memtile
perf-fp32: M=4096
perf-fp32: K=4096
perf-fp32: N=4096
perf-fp32: m_c=128
perf-fp32: m_a=32
perf-fp32: k=64
perf-fp32: n=32
perf-fp32: compute_type=bfp16
perf-fp32: accumulation_mode=fp32
perf-fp32: cache_mode=l1-weight
perf-fp32: warmup=16
perf-fp32: iters=20
perf-fp32: verify=true
perf-fp32: min_gflops=5000
perf-fp32: run

perf-cascade: M=4096
perf-cascade: K=4096
perf-cascade: N=4096
perf-cascade: m_c=64
perf-cascade: m_a=64
perf-cascade: k=64
perf-cascade: n=64
perf-cascade: compute_type=bfp16
perf-cascade: accumulation_mode=cascade
perf-cascade: cache_mode=l1-weight
perf-cascade: warmup=16
perf-cascade: iters=20
perf-cascade: verify=true
perf-cascade: min_gflops=7000
perf-cascade: run

# Performance-oriented physical-cascade design. Two independent two-row chains
# keep prefix K tiles in the fast BF16-writeback kernel, then reduce the final
# tile in accfloat before one BF16 write at the top of each chain.
perf-cascade-hybrid: M=4096
perf-cascade-hybrid: K=4096
perf-cascade-hybrid: N=4096
perf-cascade-hybrid: m_c=256
perf-cascade-hybrid: m_a=32
perf-cascade-hybrid: k=128
perf-cascade-hybrid: n=64
perf-cascade-hybrid: n_aie_cols=8
perf-cascade-hybrid: compute_type=bfp16
perf-cascade-hybrid: accumulation_mode=cascade-hybrid
perf-cascade-hybrid: cache_mode=l1-weight
perf-cascade-hybrid: cache_k=4096
perf-cascade-hybrid: warmup=16
perf-cascade-hybrid: iters=20
perf-cascade-hybrid: verify=true
perf-cascade-hybrid: verify_samples=1000
perf-cascade-hybrid: min_gflops=16000
perf-cascade-hybrid: run

perf-cascade-hybrid-3x:
	python3 ${srcdir}/benchmark_sweep.py --hardware -M 4096 -K 4096 -N 4096 --compute-types bfp16 --accumulation-modes cascade-hybrid --cache-modes l1-weight --warmup 16 --iters 20 --rounds 3 --min-gflops 16000

perf-memtile: M=4096
perf-memtile: K=4096
perf-memtile: N=4096
perf-memtile: m_c=128
perf-memtile: m_a=32
perf-memtile: k=64
perf-memtile: n=128
perf-memtile: compute_type=bfp16
perf-memtile: accumulation_mode=bf16
perf-memtile: cache_mode=memtile-weight
perf-memtile: warmup=16
perf-memtile: iters=20
perf-memtile: verify=true
perf-memtile: min_gflops=12000
perf-memtile: run
