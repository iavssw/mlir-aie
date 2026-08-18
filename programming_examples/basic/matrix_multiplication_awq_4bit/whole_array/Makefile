##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=whole_array
targetname=whole_array_awq_4bit_${dtype_out}
aie_py_src=whole_array.py

M?=1024
K?=1024
N?=2048
m?=64
k?=128
n?=64
group_size?=128
n_aie_cols?=8

aieargs+=-m $m -k $k -n $n --group-size ${group_size} --n-aie-cols ${n_aie_cols}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_g${group_size}_${n_aie_cols}c_${dtype_out}

include ${srcdir}/../makefile-common
