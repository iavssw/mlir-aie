# llama.cpp Q4_K compute-type matmul

This NPU2/Chess experiment computes C = A @ dequant(B) directly from a flat
llama.cpp block_q4_K tensor. It is independent of the existing
matrix_multiplication and matrix_multiplication_awq_4bit examples.

Q4_K_S is a model quantization preset, not a distinct block ABI. This POC
accepts tensors whose selected GGML type is Q4_K. Q5_K tensors, GGUF parsing,
and graph/backend integration are intentionally out of scope.

## Implemented paths

The same whole-array graph exposes three Chess kernels:

| compute type | weight path | activation path | matrix operation |
|---|---|---|---|
| bf16 | Q4_K affine dequant to BF16 L1 scratch | BF16 | emulated BF16/BFP16 MMUL |
| bfp16 | affine dequant directly to bfp16ebs8 L1 scratch | BF16 converted to BFP16 | native BFP16 accfloat |
| int8 | nibbles expanded to signed INT8 lanes | BF16 quantized per row/32-K group | INT8 MMUL to INT32, then affine float correction |

All paths produce BF16. The kernel receives an explicit A-subtile index,
dequantizes B only for subtile zero, and has no mutable call counter.

Native BFP16 has four user-facing accumulation modes:

| mode | behavior |
|---|---|
| bf16 | accumulate a 64-K tile in `accfloat`, then write/read BF16 between K tiles |
| fp32 | retain the whole output tile in local FP32 scratch and convert to BF16 only once |
| cascade | split K over four AIE rows, pass `accfloat` values over cascade streams, retain the completed sum in FP32, then write BF16 |
| cascade-hybrid | run two independent two-row K splits, keep fast BF16 local prefix partials, then reduce each final `accfloat` contribution over physical cascade and write BF16 once |

The full-FP32 `cascade` specialization is intentionally limited to `k=64`.
A `k=128` experiment fit data memory but overflowed AIE program memory for
its three fully scheduled cascade roles, so validation rejects it before
compilation.

`cascade-hybrid` is the performance-oriented compromise. Each column runs
two independent two-row chains; the rows in a chain own even and odd K tiles
for the same output tile. Prefix K tiles update a BF16 local partial, so those
boundaries still round to BF16. The final K tile loads that partial into
`accfloat`; the final matrix work and the bottom-to-top two-row reduction stay
in `accfloat`, without a memory spill, before the top row performs one BF16
store. It is therefore not full-K FP32.

For the 4096-cubed candidate (`mC=256`, `mA=32`, `k=128`, `n=64`), every
role object contains separate branch-free normal and final kernel symbols.
This removes the costly runtime final-tile branch while retaining physical
cascade edges. Dynamic worker loops keep the linked program-memory image at
12,826 bytes, including the worker controller and fixed libraries.

There are not separate BFP and BF16 matrix engines to run concurrently. Chess
lowers both the native-BFP path and the BF16-emulation path to the same
`VMAC.f` matrix slot. In the selected kernel, BF16 A loads/conversion, BFP
block loads, and `VMAC.f` already co-issue in one VLIW packet. Preconverting A
to BFP moved both operands onto the block-load path and reduced throughput.

Three cache modes are executable:

- stream: compressed Q4 tiles use the baseline whole-array schedule.
- l1-weight: retains one dequantized B tile while all mC/mA A subtiles are
  processed. In `cascade-hybrid`, it streams one compressed full-K panel per
  output pair through a single shim channel and demultiplexes the two parity
  shards in the MemTile without hardware replay.
- memtile-weight: uses an N-panel-outer schedule, loads one compressed
  full-K panel per column, and replays its K tiles with ObjectFifo.repeat_count.
  Hardware replay is bounded to the largest row-slab divisor no greater than
  four. Eight replays compile but stall on this NPU2; at 4096 cubed, two
  four-row slabs retain 4x compressed-weight reuse without that deadlock.

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

To run the correctness-qualified aligned default with 16 warmups, 20 timed
iterations, sampled Q4_K verification, and a 16,000-GFLOP/s gate, use:

~~~bash
make perf-ceiling
~~~

`make perf-bfp16` is an alias for the same selected path. The explicitly
unverified `make perf-bfp16-raw` target is only a throughput diagnostic.

### Measured ceiling and accuracy tradeoff

The following measurements use the local eight-column Krackan NPU in XRT
`Default` power mode. Throughput is end to end on the NPU and includes
compressed-panel loads, in-execution Q4_K dequantization, activation
conversion, matrix work, accumulation traffic, and BF16 output conversion.
Host Q4_K preparation is model-load work and is excluded.

| path | 4096-cubed throughput | BFP/FP32-reference NRMSE | direct-Q4_K-reference NRMSE |
|---|---:|---:|---:|
| native BFP, BF16 K-tile writeback, L1 reuse | **16.472 TOPS median round-average** | 0.01919 | 0.01381 |
| native BFP, bounded MemTile weight replay | 12.588 TOPS average | 0.01919 | 0.01381 |
| native BFP, local FP32 accumulation | 5.693 TOPS average | 0.01637 | 0.00991 |
| native BFP, four-row cascade FP32 | 7.681 TOPS average | 0.01637 | 0.00991 |
| native BFP, 256/32 x 64 x 64 reuse tile | 7.060 TOPS average | 0.01919 | 0.01381 |
| native BFP, two-row cascade-hybrid, MemTile replay | 12.281 TOPS average | 0.01746 | 0.01140 |
| native BFP, two-row cascade-hybrid, streamed panel | **12.623 TOPS average** | 0.01746 | 0.01140 |

The streamed-panel hybrid result uses 16 warmups and 20 timed iterations after
a reboot, passes sampled native-Q4_K verification, and keeps the last tile and
two-row reduction in `accfloat`. Its maximum absolute error is 0.0585938 and
its full run spans 12.014--12.948 TOPS. It improves on bounded MemTile replay,
but it does not meet the 16-TOPS cascade research gate.

The selected path was run for three rounds of 16 warmups plus 20 timed
iterations. Round averages were 16.504, 16.472, and 16.190 TOPS; the complete
timed range was 15.283--16.930 TOPS. Sampled checks passed with maximum
absolute error 0.078125. The C++ verifier also reports BF16 K-tile writeback
drift directly against the same BFP computation accumulated in FP32; the
measured drift was max-absolute 0.0390625 and NRMSE 0.0086545.

A post-reboot A/B check on 2026-08-25 found the device back in `Default`
power mode. The current-source aligned xclbin averaged 15.824 TOPS and the
archived pre-fringe xclbin averaged 15.893 TOPS under identical inputs and
timing; the 0.43% difference rules out a meaningful aligned-path regression.
The current xclbin's best iteration was 16.107 TOPS. The checked-in
16,000-GFLOP/s target is therefore a strict Turbo/performance-session gate;
use `min_gflops=0` when doing correctness work at Default clocks.

The result is matrix-issue bound, not compressed-weight bandwidth bound:
MemTile caching reduces weight ingress but adds sustained replay/synchronizing
bubbles. Short one-iteration MemTile results reached 16.03 TOPS, but the
16/20 sustained average is the 12.588-TOPS number above.

### Output and activation tile sweep

The k=64 native-BFP16 path was also swept across legal mC, mA, and n values
after a reboot, with the NPU in XRT `Default` power mode.  The losing shapes
use two warmups and five timed iterations; the winner uses the formal 16
warmups and 20 timed iterations.  Every completed shape passed sampled native
Q4_K verification.

| `(mC,mA,k,n)` | Throughput | Result |
|---|---:|---|
| `(128,32,64,128)` | **16.298 TOPS** | selected |
| `(128,16,64,128)` | 15.196 TOPS | extra subtile calls |
| `(64,32,64,128)` | 12.043 TOPS | insufficient B reuse |
| `(256,32,64,64)` | 7.176 TOPS | narrow N and single-row C drains |
| `(32,32,64,256)` | 7.974 TOPS | excessive Q4/dequant traffic |
| `(128,64,64,128)` | ineligible | depth-one A ObjectFIFO wedged amdxdna |

The selected tile is the only safe measured shape above 16 TOPS.  Increasing
mC to 256 makes the joined C height 1024, beyond the DMA's 10-bit dimension,
so the runtime must drain one row block at a time.  Increasing n to 256 forces
mC down to 32 under the 64-KiB L1 budget and quadruples compressed-weight and
dequantization work.  Reducing mA to 16 doubles kernel-entry/ObjectFIFO
overhead.  The unsafe mA=64 shape is rejected by both Python and C++ validators
until its FIFO schedule is redesigned.

The selected `(128,32,64,128)` tile is also the Make/PDI default. Aligned
512-row by 1024-column waves take the unchanged 16-TOPS fast path. If logical
M or N has a 256-element fringe, a specialized runtime keeps the same tile:
each core receives exact 64-row activation halves, keeps one 64x128 result
half locally, serializes both halves through one C FIFO, and activates only
the columns that own real N panels. Native Q4_K B is still dequantized once
per complete 128-row core tile. Thus `M=256,512,768,...` and
`N=256,512,768,...` are supported without host padding or a globally slower
`n=32` kernel.

At K > 4096, bounded 2/4/8-tile K slabs keep the shim outer loop within 64.
Each core then reads a contiguous 128-row A region to avoid the large
inter-half stride. A final 256-row wave shares that region across a core pair,
but each worker computes a distinct 64-row half; no MMUL or C row is
duplicated. If N is also wide, C is drained through ordered 64-row consumer
chunks so no stride exceeds 20 bits. This extends the fixed-tile contract
through K,N=32768 without changing the L3 BF16/Q4_K/BF16 ABI.

The exhaustive combined-fringe test at `768x256x1280` passes every output with
direct-Q4_K NRMSE 0.0159203. At `768x4096x16384`, the integrated fixed-tile
path averages 13.7065 TOPS with 16 warmups and 20 timed iterations. The lower
average is expected because one third of its M work is the 256-row half-wave;
the aligned interior remains the approximately 16-TOPS kernel.

The bounded long-K path also passes sampled hardware verification at
`768x14336x4096`, averaging 13.797 TOPS in Default mode with direct-Q4_K
NRMSE 0.02599. Simultaneous long-K/wide-N corners pass Iron resolution, all
host TAP checks, and Chess/xclbin compilation through
`768x32768x32768` (slab factor 8). Its device run remains pending because the
subsequent read-only XRT probe wedged; no driver reset or additional kernel
load was attempted.

Reproduce the safe screening sweep with:

~~~bash
make bench-bfp16-tile-sweep
~~~

### Larger K-tile sweep

Increasing the compile-time K tile reduces BF16 partial-writeback boundaries,
but it also grows the A FIFO and BFP16 weight scratch.  The following candidates
are the fastest legal shapes measured at each larger K in the same Default-power
session:

| K tile | `(mC,mA,k,n)` | Throughput | BF16-writeback drift NRMSE | Direct-Q4 reference NRMSE |
|---:|---|---:|---:|---:|
| 64 | `(128,32,64,128)` | **16.481 TOPS** | 0.0086545 | 0.0138094 |
| 128 | `(64,16,128,128)` | **12.550 TOPS** | 0.0069166 | 0.0122607 |
| 256 | `(128,64,256,32)` | 8.620 TOPS | 0.0051407 | 0.0109443 |
| 512 | `(128,16,512,16)` | 5.364 TOPS | 0.0039258 | 0.0104556 |

The k=64 and k=128 rows use 16 warmups and 20 timed iterations; the k=256 and
k=512 rows are three-iteration screening measurements.  All rows passed the
same sampled native-Q4_K verification.  The k=128 result lowers writeback-drift
NRMSE by 20.1% and direct-Q4 NRMSE by 11.2%, but is 23.8% slower.  No larger-K
candidate exceeds the k=64 BF16-writeback default, so the default is unchanged.

The limiting tradeoff is the 64-KiB compute-tile memory.  Keeping n=128 at
k=128 requires mC=64, which halves weight reuse and doubles Q4 transfer and
dequantization.  At k=256 and k=512, n must fall to 32 and 16, multiplying A
traffic and narrowing the matrix-issue schedule.  k=1024 exceeds the AIE DMA's
10-bit transfer-dimension limit; its useful shapes also cannot retain their
buffers, stack, and 4-KiB safety margin within 64 KiB.

Reproduce the sweep with:

~~~bash
make perf-bfp16-k-sweep
~~~

For the absolute clock ceiling, first select Turbo mode in another terminal;
this needs administrator permission and is deliberately not changed by Make:

~~~bash
xrt-smi examine --report platform
sudo xrt-smi configure --pmode turbo
make perf-ceiling-3x
sudo xrt-smi configure --pmode default
~~~

Always record the reported power mode with a ceiling number. A reboot resets
this device to Default. The table above intentionally contains only
Default-mode measurements; Turbo requires an interactive
administrator-authorized device-state change.

A small full-verification run is:

~~~bash
python3 whole_array.py -M 512 -K 256 -N 512 --m-c 64 --m-a 32 -k 128 -n 64 --n-aie-cols 8 --compute-type bf16 --cache-mode stream --verify-mode full
~~~

The native-BFP fixed-tile 256-row/column fringe check is:

~~~bash
python3 whole_array.py -M 256 -K 256 -N 1280 --m-c 128 --m-a 32 -k 64 -n 128 --n-aie-cols 8 --compute-type bfp16 --cache-mode l1-weight --verify-mode full
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
make run M=768 K=256 N=1280 m_c=128 m_a=32 k=64 n=128 compute_type=bfp16 cache_mode=l1-weight
~~~

The stable comparison targets use the exact configurations from the table:

~~~bash
make perf-fp32
make perf-cascade
make perf-memtile
make perf-cascade-hybrid
make perf-cascade-hybrid-3x
~~~

All retain BF16 host activations and BF16 output. `perf-fp32` keeps one local
FP32 C tile across K; `perf-cascade` passes intermediate `accfloat` values
between rows; `perf-memtile` exercises bounded compressed-panel replay. The
hybrid target uses the 256/32 x 128 x 64 two-row, streamed-panel design and
fails if its average is below 16,000 GFLOP/s.
`perf-cascade-hybrid-3x` applies three rounds of 16 warmups plus 20 timed iterations and gates the median round-average.

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
make perf-cascade-hybrid-3x
~~~

perf-large applies the primary protocol: BF16 activation input, eight
columns, 4096 cubed, 16 warmups, 20 timed iterations, three rounds, and a
12,000-GFLOP/s gate. Weight preparation is outside execution; compressed
MemTile loads, activation conversion, in-execution dequantization, and output
conversion are included. The sweep writes the fastest correct qualifying row
to build/selected_default.json.

The Make default is the verified native-BFP16/L1-weight candidate. The Python
CLI keeps the conservative BF16 streaming defaults for small direct tests.
Do not interpret compile-only success or capacity-model rows as a
greater-than-16-TOPS measurement. The measured hybrid result is explicitly
reported below that goal until a future design passes the gate.
