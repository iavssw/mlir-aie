##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
subdir=whole_array
targetname=whole_array

M?=4096
K?=4096
N?=4096

m?=64
k?=64
n?=64

n_aie_cols?=8
b_col_maj?=1
c_col_maj?=0
use_scalar?=0

aieargs+=-m $m -k $k -n $n --n-aie-cols ${n_aie_cols} --b-col-maj ${b_col_maj} --c-col-maj ${c_col_maj}
ifeq (${use_scalar}, 1)
	aieargs+= --scalar 1
endif
runargs+=--b_col_maj ${b_col_maj} --c_col_maj ${c_col_maj}
target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c

include ${srcdir}/../makefile-common
