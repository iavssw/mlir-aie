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
