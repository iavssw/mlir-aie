##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Q4_K-derived, prequantized INT8 x INT8 -> INT32 whole-array ceiling.
# This deliberately excludes activation quantization, Q4_K affine correction,
# and BF16 output conversion from the timed region.
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include ${current_dir}../../../makefile-common

subdir=upper_bound
targetname=int8_upper_bound
aie_py_src=whole_array.py

M?=4096
K?=4096
N?=4096
m?=64
k?=64
n?=64
n_aie_cols?=8

warmup?=1
iters?=1
verify?=true
verify_samples?=1000
min_gops?=0
verbosity?=0
kernel_repeats?=1

target_suffix=${M}x${K}x${N}_${m}x${k}x${n}_${n_aie_cols}c_r${kernel_repeats}
xclbin_target=build/int8_ceiling_${target_suffix}.xclbin
insts_target=build/int8_ceiling_insts_${target_suffix}.bin

aieargs=--dev npu2 -M ${M} -K ${K} -N ${N} -m ${m} -k ${k} -n ${n} \
	--n-aie-cols ${n_aie_cols} --dtype_in i8 --dtype_out i32 \
	--b-col-maj 1 --c-col-maj 0 --use-chess 1 \
	--kernel-repeats ${kernel_repeats}
runargs=--warmup ${warmup} --iters ${iters} --verify=${verify} \
	--verify-samples ${verify_samples} --min-gops ${min_gops} \
	--verbosity ${verbosity} --kernel-repeats ${kernel_repeats}

.PHONY: all
all: ${xclbin_target} ${targetname}.exe

${xclbin_target} ${insts_target} &: ${srcdir}/${aie_py_src}
	mkdir -p $(@D)
	python3 $< ${aieargs} --xclbin-path=${xclbin_target} --insts-path=${insts_target}

${targetname}.exe: ${srcdir}/test.cpp ${srcdir}/../common.h ${srcdir}/../CMakeLists.txt
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake -E env CXXFLAGS="-std=c++23 -ggdb" cmake `${getwslpath} ${srcdir}/..` -D CMAKE_C_COMPILER=${CC} -D CMAKE_CXX_COMPILER=${CXX} -DTARGET_NAME=${targetname} -Dsubdir=${subdir}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

.PHONY: run
run: ${targetname}.exe ${xclbin_target} ${insts_target}
	${powershell} ./$< -x ${xclbin_target} -i ${insts_target} --kernel MLIR_AIE \
		-M ${M} -K ${K} -N ${N} ${runargs}

.PHONY: perf-ceiling
perf-ceiling: warmup=16
perf-ceiling: iters=20
perf-ceiling: verify=true
perf-ceiling: verify_samples=1000
perf-ceiling: run

.PHONY: perf-resident
perf-resident: M=512
perf-resident: K=256
perf-resident: N=512
perf-resident: m=64
perf-resident: k=64
perf-resident: n=64
perf-resident: kernel_repeats=1024
perf-resident: warmup=16
perf-resident: iters=20
perf-resident: verify=true
perf-resident: run

.PHONY: clean
clean:
	rm -rf build _build ${targetname}.exe trace.txt
