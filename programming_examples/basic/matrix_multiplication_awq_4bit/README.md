# Configurable AWQ INT4 whole-array matmul

This NPU2-only example computes `C = A @ dequant(B)` with BF16 activations,
unsigned INT4 weights and zero points, BF16 groupwise scales, and BF16 or F32
output.  It uses the current `@iron.jit` whole-array Worker/Runtime/TaskGroup
pipeline and a local Chess kernel; it does not use the legacy dialect API or
mutable kernel call-order state.

The default is `M=1024, K=1024, N=2048`, a `64x128x64` core tile, group size
128, eight NPU2 columns, BF16 output, and Chess.

## Environment

Start every build or run from the Desktop, source the requested environment,
and then change to this example explicitly:

```bash
cd /home/greg/Desktop
source /home/greg/Desktop/env-mlir-aie.sh
cd /home/greg/Desktop/mlir-aie-fork-experimental/programming_examples/basic/matrix_multiplication_awq_4bit/whole_array
```

The environment script currently prints a harmless failed `cd` for an
obsolete example directory.  The explicit final `cd` above is intentional.

## Python run and callable

Run the default design and automatically choose full or sampled reference
verification:

```bash
python3 whole_array.py
```

A small full-verification run is:

```bash
python3 whole_array.py -M 512 -K 128 -N 512 \
  -m 64 -k 128 -n 64 --group-size 128 --n-aie-cols 8 \
  --dtype_out bf16 --verify-mode full
```

`whole_array.py` exports the `@iron.jit` callable
`whole_array_awq_4bit(A, packed_B, C, **compile_time_config)` (also aliased as
`whole_array`).  `A`, `packed_B`, and `C` are flat device tensors.  The helper
module [`packing.py`](packing.py) provides `AWQConfig`, deterministic input
generation, packing/unpacking, dequantization, full reference matmul, sampled
reference evaluation, and host-only TAP models.

## Packed B contract

Logical tensors have these exact types and shapes:

| tensor | dtype | shape |
|---|---|---|
| `qweight` | `numpy.uint8`, values 0..15 | `(K, N)` |
| `scales` | `ml_dtypes.bfloat16` | `(K / group_size, N)` |
| `zeros` | `numpy.uint8`, values 0..15 | `(K / group_size, N)` |

The output is one flat `numpy.uint8` buffer.  Physical tile order is AIE
column, round-robin N tile owned by that column, then K tile.  Within each
tile:

1. Weights are visited by K 8-block, then N 8-block.  Each 8x8 block is
   row-major and adjacent N values occupy one byte (`even` in the low nibble,
   `odd` in the high nibble).
2. BF16 scales are group-major, then N-major.
3. Zero points are group-major and 8 columns at a time.  Each 8-byte vector is
   duplicated to form the kernel's native 16-byte load.
4. Zero padding extends the tile to a multiple of `k` bytes.

For the default tile/group this is exactly 4096 weight bytes + 128 scale bytes
+ 128 zero bytes = 4352 bytes, represented as `34x128` uint8.  There is no
oversized host allocation.

Dequantization is `BF16(qweight - zero) * scale`, stored to aligned per-core
BF16 scratch.  The kernel dequantizes on explicit A-half index zero, reuses the
scratch for half one, and offsets C by the explicit half index.

## Configuration rules

Groups 32, 64, and 128 and NPU2 column counts 1, 2, 4, and 8 are supported.
The validator checks matrix/tile divisibility, 8x8x8 MMUL geometry, 10-bit DMA
sizes, two-row-block ping-pong scheduling, packed buffer size, and the 64-KiB
compute-tile budget.  That budget includes depth-two A/B FIFOs, depth-two BF16
or depth-one F32 C, BF16 dequant scratch, the `0xD00` stack, and a 4-KiB safety
margin.

The default group-128/BF16 tile uses 65,280 bytes and is deliberately close to
the limit.  A group-64/F32 `64x128x64` tile needs 65,792 bytes and is rejected;
use `64x128x32` for that correctness configuration.

## Make, CMake, XRT, and trace

Build the current configuration and run the C++ XRT host:

```bash
make
make run
```

The required small group-64/F32 workflow is:

```bash
make run M=512 K=128 N=256 m=64 k=128 n=32 \
  group_size=64 n_aie_cols=8 dtype_out=f32
```

Tracing builds a trace-enabled xclbin, runs it, and emits
`trace_awq_mm.json`:

```bash
make trace trace_size=65536
```

The trace target deliberately defaults to a `512x128x64`, one-column
diagnostic configuration, while ordinary build/run and performance targets
retain the eight-column defaults above.  One representative compute tile is
traced through the first otherwise-unused shim column; this avoids perturbing
the fully occupied eight-column performance routing.  Override the diagnostic
shape with `trace_M`, `trace_K`, `trace_N`, `trace_m`, `trace_k`, `trace_n`,
`trace_group_size`, and `trace_n_aie_cols`.

The equivalent direct-Python trace compile is:

```bash
python3 whole_array.py -M 512 -K 128 -N 64 \
  -m 64 -k 128 -n 64 --group-size 128 --n-aie-cols 1 \
  --trace_size=65536 --xclbin-path=build/trace.xclbin \
  --insts-path=build/trace_insts.bin
```

Keep a spare shim column for tracing.  The fully occupied eight-column layout
is reserved for normal/performance runs because its stream network has no
legal additional compute-trace route.

Host-only contract tests run with `make host-tests`.  Generated xclbins,
instructions, CMake trees, executables, traces, and benchmark logs are ignored.

## Performance

The reference gate uses Chess, eight columns, a `64x128x64` tile, group 128,
and BF16 output.  Each of three rounds has 16 warmups and 20 timed iterations.
Throughput is `2*M*K*N / NPU_time`; the median round average must be at least
6840 GFLOP/s (90% of the recorded 7600 GFLOP/s reference):

```bash
make perf-reference
```

The large run uses the same protocol at `4096^3` and reports speedup over the
measured current BF16 baseline of 3250.3 GFLOP/s without gating:

```bash
make perf-large
```

The equivalent direct controls are `--warmup`, `--iters`,
`--benchmark-repeats`, and `--min-gflops`.

The implementation validation on this NPU2 system produced these effective
throughputs (GFLOP/s), using the exact three-round protocol above:

| problem | round averages | median | result |
|---|---:|---:|---|
| `1024x1024x2048` | 8194.33, 8460.83, 8473.09 | 8460.83 | 23.7% above the 6840 gate |
| `4096x4096x4096` | 12008.12, 12011.87, 12036.66 | 12011.87 | 3.696x the 3250.3 baseline |
