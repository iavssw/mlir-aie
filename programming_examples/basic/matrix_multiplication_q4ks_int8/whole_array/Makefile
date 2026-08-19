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

# Eligible BF16-input Q4_K INT8 experiment. Command-line assignments still
# override every value because these are conditional defaults.
M?=4096
K?=4096
N?=4096

m_c?=128
m_a?=32
k?=64
n?=128
n_aie_cols?=8
compute_type?=int8
cache_mode?=l1-weight
activation_input?=bf16

include ${srcdir}/../makefile-common

# Run the eligible INT8 configuration with activation quantization, Q4_K
# affine correction, and BF16 output included in NPU time.
.PHONY: perf-int8 perf-bfp16-reference
perf-int8: warmup=16
perf-int8: iters=20
perf-int8: verify=true
perf-int8: verify_samples=1000
perf-int8: min_gflops=90
perf-int8: run

# Rebuild and run the native-BFP16 comparison in the same isolated tree.
perf-bfp16-reference: compute_type=bfp16
perf-bfp16-reference: warmup=16
perf-bfp16-reference: iters=20
perf-bfp16-reference: verify=true
perf-bfp16-reference: verify_samples=1000
perf-bfp16-reference: min_gflops=12000
perf-bfp16-reference: run
