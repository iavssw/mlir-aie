# llama.cpp Q4_K compute-type matmul

This NPU2/Chess experiment computes C = A @ dequant(B) directly from a flat
llama.cpp block_q4_K tensor. It is independent of the existing
matrix_multiplication and matrix_multiplication_awq_4bit examples.

Q4_K_S is a model quantization preset, not a distinct block ABI. This POC
accepts tensors whose selected GGML type is Q4_K. Q5_K tensors, GGUF parsing,
and graph/backend integration are intentionally out of scope.

## Implemented paths

The same whole-array graph exposes three Chess kernels:

| compute type | weight path | activation path | accumulation |
|---|---|---|---|
| bf16 | Q4_K affine dequant to BF16 L1 scratch | BF16 | emulated BF16/BFP16 MMUL |
| bfp16 | affine dequant directly to bfp16ebs8 L1 scratch | BF16 converted to BFP16 | native BFP16 accfloat |
| int8 | nibbles expanded to signed INT8 lanes | BF16 quantized per row/32-K group | INT8 MMUL to INT32, then affine float correction |

All paths produce BF16. The kernel receives an explicit A-subtile index,
dequantizes B only for subtile zero, and has no mutable call counter.

Three cache modes are executable:

- stream: compressed Q4 tiles use the baseline whole-array schedule.
- l1-weight: labels the asymmetric-tile experiment; one dequantized B tile
  is retained while all mC/mA A subtiles are processed.
- memtile-weight: uses an N-panel-outer schedule, loads one compressed
  full-K panel per column, and replays its K tiles for every M row block with
  ObjectFifo.repeat_count.

memtile-activation and joint-slab are represented by exact capacity models
and the sweep table, but are intentionally rejected by device compilation:
the compute-to-MemTile converter/replay graph is not yet implemented.
Preconverted BFP16/INT8 activation inputs are likewise modelled as ceilings
and cannot be selected as the primary result.

## Environment

Every build starts from the Desktop, sources the requested environment, and
then explicitly enters this directory:

~~~bash
cd /home/greg/Desktop
source /home/greg/Desktop/env-mlir-aie.sh
cd /home/greg/Desktop/mlir-aie-fork-experimental/programming_examples/basic/matrix_multiplication_q4ks/whole_array
~~~

## Native and prepared data contracts

The input is a flat numpy.uint8 payload with exactly
N * (K / 256) * 144 bytes. Blocks are N-row-major and K-block-minor. Each
144-byte block is the llama.cpp ABI:

1. little-endian FP16 d;
2. little-endian FP16 dmin;
3. twelve bytes encoding eight 6-bit scales and eight 6-bit minima;
4. 128 bytes holding 256 Q4 nibbles.

For group g, dequantization is
weight = q * BF16(d * scale_code[g]) - BF16(dmin * min_code[g]).
The host repacker rounds effective scales and biases to BF16 to match the
kernel.

When no external Q4_K payload is supplied, the Python and C++ hosts generate
the same deterministic LLM-like input profile. Each BF16 activation row is
centered and normalized to unit RMS, modelling a post-LayerNorm/RMSNorm token.
Each floating-point weight column is centered and normalized to unit L2 norm,
then quantized into the native 144-byte Q4_K block hierarchy. Thus the device
still receives real llama.cpp-format Q4_K weights rather than independently
random nibbles and metadata.

prepare_q4ks_weights(native_B, config, storage_type) supports q4, bf16,
bfp16, and int8 prepared forms. Device execution currently consumes q4.
Compressed tiles are ordered by AIE column, round-robin N tile, then K tile.
Each tile holds 8x8 microtiled nibbles, followed by interleaved eight-column
BF16 scale/bias vectors and zero alignment padding. Unlike the earlier AWQ
format, Q4_K has a floating bias, so there is no duplicated uint8 zero-point
vector.

For the symmetric 64/32 x 128 x 64 tile a compressed tile is 5,120 bytes:
4,096 bytes of nibbles and 1,024 bytes of scale/bias metadata. The validator
checks the 64-KiB compute-tile and 512-KiB MemTile budgets, DMA limits,
matrix/tile divisibility, Q4_K block alignment, and ping/pong row scheduling.

## Python and XRT workflows

The Make workflow defaults to the measured native-BFP16 throughput candidate:
4096 cubed, the asymmetric 128/32 x 64 x 128 tile, eight columns, BF16 host
activations, and L1 weight reuse.  The Python CLI retains its smaller BF16
defaults unless these options are supplied explicitly.

The ordinary Make run keeps sampled verification enabled:

~~~bash
make run
~~~

To run the correctness-qualified default with 16 warmups, 20 timed iterations,
sampled Q4_K verification, and a 12,000-GFLOP/s gate, use:

~~~bash
make perf-bfp16
~~~

On the local NPU2, the full 16-warmup/20-iteration target measured 16.19
TFLOP/s average (15.64--16.65 TFLOP/s over the timed iterations). Sampled
verification passed with maximum absolute error 0.078125 and normalized RMSE
0.02077. `make perf-bfp16-raw` remains available only as an explicitly
unverified diagnostic.

A small full-verification run is:

~~~bash
python3 whole_array.py -M 512 -K 256 -N 512 --m-c 64 --m-a 32 -k 128 -n 64 --n-aie-cols 8 --compute-type bf16 --cache-mode stream --verify-mode full
~~~

The native-BFP asymmetric candidate is:

~~~bash
python3 whole_array.py -M 1024 -K 256 -N 1024 --m-c 128 --m-a 32 -k 64 -n 128 --n-aie-cols 8 --compute-type bfp16 --cache-mode l1-weight --verify-mode full
~~~

whole_array.py exports prepare_q4ks_weights, whole_array_q4ks, and
generate_taps.

Build and run the equivalent C++ XRT host with:

~~~bash
make
make run
~~~

Select another candidate with Make variables:

~~~bash
make run M=1024 K=256 N=1024 m_c=128 m_a=32 k=64 n=128 compute_type=bfp16 cache_mode=l1-weight
~~~

Trace builds use a separate four-column diagnostic configuration because a
spare shim column is required:

~~~bash
make trace trace_size=65536
~~~

## Tests and performance selection

Host ABI, packing, references, memory models, and TAPs:

~~~bash
make host-tests
~~~

A capacity-only sweep is safe without an attached NPU:

~~~bash
python3 benchmark_sweep.py -M 4096 -K 4096 -N 4096
~~~

Hardware sweeps write ignored CSV/JSON tables:

~~~bash
make bench-sweep
make perf-reference
make perf-large
~~~

perf-large applies the primary protocol: BF16 activation input, eight
columns, 4096 cubed, 16 warmups, 20 timed iterations, three rounds, and a
12,000-GFLOP/s gate. Weight preparation is outside execution; compressed
MemTile loads, activation conversion, in-execution dequantization, and output
conversion are included. The sweep writes the fastest correct qualifying row
to build/selected_default.json.

The Make default is the verified native-BFP16/L1-weight candidate. The Python
CLI keeps the conservative BF16 streaming defaults for small direct tests.
Do not interpret compile-only success or capacity-model rows as a 12-TOPS
measurement.
