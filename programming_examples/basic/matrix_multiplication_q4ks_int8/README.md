# Q4_K INT8 compute-ceiling experiment

This directory is an isolated AIE2P/Chess experiment derived from
`matrix_multiplication_q4ks`. It measures how much of the NPU's integer matrix
engine can be exposed by Q4_K inference and compares that result with the
native-BFP16 Q4_K path. The original `matrix_multiplication_q4ks` directory is
not modified.

The input contract for the eligible path is unchanged: A is BF16, B is a flat
native llama.cpp `block_q4_K` payload (144 bytes for every 256 weights), and C
is BF16. Weight preparation remains model-load work; activation quantization,
Q4_K dequantization/affine correction, matrix multiplication, and BF16 output
conversion are included in NPU time.

## Measured bounds

All throughput uses `2*M*K*N / NPU_time` and eight NPU2 columns.

| Experiment | Shape / timing | Throughput | What is included |
| --- | --- | ---: | --- |
| Resident raw INT8 ceiling | 512x256x512, 1024 repeats, 16 warmups + 20 timed, 7.96201 ms average | **17.2618 TOPS** | Native 8x8x8 INT8 MMUL, INT32 accumulation, and output zeroing; A/B tiles remain in L1 across repeats |
| Prequantized full-array INT8 ceiling | 4096x4096x4096, 16 warmups + 20 timed, 16.3132 ms average | **8.42500 TOPS** | Host-prequantized INT8 A, raw Q4 codes expanded before timing, full array DMA/MMUL/INT32 output |
| Eligible Q4_K INT8 | 4096x4096x4096, 16 warmups + 20 timed, 1.34155 s average | **0.102448 TOPS** | Native Q4_K weights, BF16 A input, per-row/per-32-K activation quantization, INT8 MMUL, group affine correction, BF16 output |
| Native-BFP16 Q4_K reference | 4096x4096x4096, 16 warmups + 20 timed, 8.30570 ms average | **16.5476 TFLOP/s** | Native Q4_K weights, BF16 A input, on-core BFP16 conversion/dequantization, native BFP16 MMUL, BF16 output |

The resident result is a raw compute ceiling, not an inference result. It
deliberately amortizes host and MemTile transfers and excludes activation
quantization, Q4_K scale/min correction, and BF16 output conversion. The
prequantized full-array result is a more conservative optimistic bound but is
also ineligible as an LLM path because its conversion work is performed on the
host.

The eligible INT8 result is slow because Q4_K has an independent scale and
minimum for every output row and 32 K values. An INT32 dot product cannot be
accumulated across those groups before applying the floating-point scale and
bias correction. The resulting conversion and groupwise outer-product work
dominates the native INT8 MMUL.

For this workload, **native BFP16 is the desired compute datatype**. The raw
resident INT8 engine is only 1.04x faster than the complete native-BFP16 path;
the prequantized full-array INT8 ceiling is 0.51x as fast, and the eligible
INT8 implementation is roughly 162x slower. AIE2P's native BFP16 MMUL applies
shared exponents in the compute datapath, which matches Q4_K's need for scaled
groups much better than explicit INT32 correction.

## Implemented INT8 changes

The eligible kernel in `whole_array/q4ks.cc` now uses native AIE2P
`aie::mmul<8, 8, 8, int8, int8, acc32>`, vectorized BF16 activation
quantization, vectorized group correction, and BF16 output conversion. At
1024x256x1024 this is about 20.5x faster than the earlier scalar 4x8x8 POC
(98.1 versus 4.79 GFLOP/s), while retaining the native Q4_K data contract.

`upper_bound/` contains the two deliberately optimistic INT8 ceilings. It
generates native Q4_K weights, decodes their raw Q codes for the prequantized
kernel, verifies exact INT32 dot products, and provides both full-array and
L1-resident benchmark targets.

## Reproduce

Every build should source the project environment from `/home/greg/Desktop`.
Its obsolete final `cd` can print a harmless error; explicitly enter the target
directory afterward.

```sh
cd /home/greg/Desktop
source /home/greg/Desktop/env-mlir-aie.sh

cd /home/greg/Desktop/mlir-aie-fork-experimental/programming_examples/basic/matrix_multiplication_q4ks_int8/upper_bound
make perf-resident
make perf-ceiling

cd ../whole_array
make perf-int8
make perf-bfp16-reference
```

The `perf-int8` and `perf-bfp16-reference` targets use the primary 4096-cubed,
16-warmup, 20-timed-iteration protocol. The upper-bound targets use the same
protocol. Override dimensions or repetitions through normal Make variables,
for example `make perf-resident kernel_repeats=2048`.

## Correctness

The resident and prequantized ceilings require exact sampled INT32 dot-product
agreement. The eligible path is checked both against its INT8 affine model and
against a direct dequantized native-Q4_K BF16 reference. The measured large
eligible run reported maximum absolute error 0.0546875 and normalized RMSE
0.0116346 against the direct Q4_K reference. The same-tree BFP16 comparison
reported maximum absolute error 0.046875 and normalized RMSE 0.0146555.
