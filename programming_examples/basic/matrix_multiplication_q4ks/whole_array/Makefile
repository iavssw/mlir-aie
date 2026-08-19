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

# Correctness-qualified native-BFP16 default. Command-line assignments still
# override every value because these are conditional defaults.
M?=4096
K?=4096
N?=4096

m_c?=128
m_a?=32
k?=64
n?=128
n_aie_cols?=8
compute_type?=bfp16
accumulation_mode?=bf16
cache_mode?=l1-weight
activation_input?=bf16

include ${srcdir}/../makefile-common

# Run the documented BFP16 configuration with the primary warmup/iteration
# counts, sampled Q4_K verification, and the 12-TFLOP/s performance gate.
.PHONY: perf-bfp16 perf-bfp16-raw
perf-bfp16: warmup=16
perf-bfp16: iters=20
perf-bfp16: verify=true
perf-bfp16: verify_samples=1000
perf-bfp16: min_gflops=12000
perf-bfp16: run

# Optional throughput-only diagnostic. This uses the same workload but skips
# verification; it must not be reported as a correctness-qualified result.
perf-bfp16-raw: warmup=16
perf-bfp16-raw: iters=20
perf-bfp16-raw: verify=false
perf-bfp16-raw: min_gflops=12000
perf-bfp16-raw: run

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
