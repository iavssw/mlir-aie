##===- Makefile -----------------------------------------------------------===##
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
current_dir := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
include ${current_dir}../../../makefile-common

subdir=whole_array_systolic
targetname=matrix_multiplication_q4ks_systolic
aie_py_src=whole_array.py
devicename=npu2

M?=4096
K?=4096
N?=4096
n?=32
m_a?=32
n_aie_cols?=8
weight_flow?=q4-expand-once
transport?=q4-column-dequant-cache
panel_slab?=16
c_panel_slab?=4
prebuilt_dir?=

warmup?=1
iters?=1
verify?=true
verify_mode?=sampled
verify_samples?=1000
min_gflops?=0
trace_size?=65536
verbosity?=0
power_mode?=unknown

k_stage=$(shell expr ${K} / 8)
target_suffix=${M}x${K}x${N}_m${m_a}k${k_stage}n${n}_p${panel_slab}_cp${c_panel_slab}_${weight_flow}_${transport}
xclbin_target?=build/final_${target_suffix}.xclbin
insts_target?=build/insts_${target_suffix}.bin
kernel_prj=$(patsubst %.xclbin,%.prj,${xclbin_target})
host_exe?=${targetname}.exe

aieargs=--dev npu2 -M ${M} -K ${K} -N ${N} --m-tile ${m_a} --n-tile ${n} --weight-flow ${weight_flow} --transport ${transport} --panel-slab ${panel_slab} --c-panel-slab ${c_panel_slab}
runargs=--warmup ${warmup} --iters ${iters} --verify=${verify} --verify-samples ${verify_samples} --min-gflops ${min_gflops} --verbosity ${verbosity} --tile-m-c ${m_a} --tile-m-a ${m_a} --tile-k ${k_stage} --tile-n ${n} --n-aie-cols 8 --compute-type bfp16 --accumulation-mode cascade --cache-mode ${weight_flow} --cache-k ${K}

trace_M?=256
trace_K?=2048
trace_N?=512
trace_n?=32
trace_m_a?=${m_a}
trace_k_stage=$(shell expr ${trace_K} / 8)
trace_suffix=${trace_M}x${trace_K}x${trace_N}_m${trace_m_a}k${trace_k_stage}n${trace_n}_p${panel_slab}_cp${c_panel_slab}_${weight_flow}_${transport}
trace_xclbin_target?=build/trace_${trace_suffix}.xclbin
trace_insts_target?=build/trace_insts_${trace_suffix}.bin
trace_aieargs=--dev npu2 -M ${trace_M} -K ${trace_K} -N ${trace_N} --m-tile ${trace_m_a} --n-tile ${trace_n} --weight-flow ${weight_flow} --transport ${transport} --panel-slab ${panel_slab} --c-panel-slab ${c_panel_slab} --trace_size ${trace_size}
trace_runargs=--warmup ${warmup} --iters ${iters} --verify=${verify} --verify-samples ${verify_samples} --min-gflops ${min_gflops} --verbosity ${verbosity} --tile-m-c ${trace_m_a} --tile-m-a ${trace_m_a} --tile-k ${trace_k_stage} --tile-n ${trace_n} --n-aie-cols 8 --compute-type bfp16 --accumulation-mode cascade --cache-mode ${weight_flow} --cache-k ${trace_K} --trace_sz ${trace_size}

.PHONY: all
all: ${xclbin_target} ${host_exe}

${xclbin_target} ${insts_target} &: ${srcdir}/${aie_py_src} ${srcdir}/q4ks_systolic.cc ${srcdir}/systolic_packing.py ${srcdir}/../packing.py
	mkdir -p $(@D)
ifneq ($(strip ${prebuilt_dir}),)
	mkdir -p ${kernel_prj}
	cp ${prebuilt_dir}/*.o ${kernel_prj}/
endif
	Q4KS_SYSTOLIC_PREBUILT_DIR="${prebuilt_dir}" python3 ${srcdir}/${aie_py_src} ${aieargs} --xclbin-path=${xclbin_target} --insts-path=${insts_target}

${trace_xclbin_target} ${trace_insts_target} &: ${srcdir}/${aie_py_src} ${srcdir}/q4ks_systolic.cc ${srcdir}/systolic_packing.py ${srcdir}/../packing.py
	mkdir -p $(@D)
	python3 ${srcdir}/${aie_py_src} ${trace_aieargs} --xclbin-path=${trace_xclbin_target} --insts-path=${trace_insts_target}

${host_exe}: ${srcdir}/test.cpp ${srcdir}/systolic_common.h ${srcdir}/../test.cpp ${srcdir}/../common.h ${srcdir}/../CMakeLists.txt
	cmake -E env CXXFLAGS="-std=c++23 -ggdb" cmake -S ${srcdir}/.. -B _build -D CMAKE_C_COMPILER=/usr/bin/clang -D CMAKE_CXX_COMPILER=/usr/bin/clang++ -D TARGET_NAME=${targetname} -Dsubdir=${subdir}
	cmake --build _build --config Release -j2
	cp _build/${targetname} $@

.PHONY: run
run: ${host_exe} ${xclbin_target} ${insts_target}
	./${host_exe} -x ${xclbin_target} -i ${insts_target} --kernel MLIR_AIE -M ${M} -K ${K} -N ${N} ${runargs}

.PHONY: run-python
run-python:
	python3 ${srcdir}/${aie_py_src} ${aieargs} --verify-mode ${verify_mode} --verify-samples ${verify_samples} --warmup ${warmup} --iters ${iters} --min-gflops ${min_gflops}

.PHONY: trace
trace: ${host_exe} ${trace_xclbin_target} ${trace_insts_target}
	./${host_exe} -x ${trace_xclbin_target} -i ${trace_insts_target} --kernel MLIR_AIE -M ${trace_M} -K ${trace_K} -N ${trace_N} ${trace_runargs}
	${srcdir}/../../../../python/utils/trace/parse.py --input trace.txt --mlir $(trace_xclbin_target:.xclbin=.prj)/input_with_addresses.mlir --output trace_q4ks_systolic.json

.PHONY: host-tests cpp-byte-test
host-tests:
	PYTHONPATH=${srcdir}/.. python3 -m unittest discover -s ${srcdir}/tests -p 'test_*.py' -v

cpp-byte-test: ${host_exe}
	PYTHONPATH=${srcdir}/.. python3 ${srcdir}/test_cpp_packing.py --host ./${host_exe}

.PHONY: bench-primitives bench-sweep
bench-primitives:
	python3 ${srcdir}/benchmark_sweep.py --hardware --preset primitives --warmup ${warmup} --iters ${iters} --rounds 1 --power-mode ${power_mode}

bench-sweep:
	python3 ${srcdir}/benchmark_sweep.py --hardware --preset sweep --include-baseline -M ${M} -K ${K} -N ${N} --warmup ${warmup} --iters ${iters} --rounds 1 --power-mode ${power_mode}

.PHONY: perf-reference perf-native perf-native-report perf-large perf-ceiling
perf-reference:
	python3 ${srcdir}/benchmark_sweep.py --hardware --preset correctness -M 256 -K 2048 -N 512 --warmup 1 --iters 1 --rounds 1 --full-verify --power-mode ${power_mode}

perf-native:
	python3 ${srcdir}/benchmark_sweep.py --hardware --preset native -M 4096 -K 4096 -N 4096 --warmup 16 --iters 20 --rounds 3 --min-gflops 16000 --power-mode ${power_mode}

perf-native-report:
	python3 ${srcdir}/benchmark_sweep.py --hardware --preset native -M 4096 -K 4096 -N 4096 --warmup 16 --iters 20 --rounds 3 --power-mode ${power_mode}

perf-large: perf-native

perf-ceiling:
	python3 ${srcdir}/benchmark_sweep.py --hardware --preset ceiling -M 4096 -K 4096 -N 4096 --warmup 16 --iters 20 --rounds 3 --min-gflops 16000 --power-mode ${power_mode}

.PHONY: clean
clean:
	rm -rf build _build ${host_exe} results trace.txt trace_q4ks_systolic.json
