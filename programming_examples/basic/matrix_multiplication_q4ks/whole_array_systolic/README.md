# Q4_K stationary-activation systolic array

This directory is an isolated AIE2P/Chess experiment. It does not modify the
parent `whole_array` implementation.

The selected eligible path starts with native llama.cpp `block_q4_K` bytes and
BF16 activations. Eight physical columns own disjoint K/8 partitions, four
physical rows own independent 32-row activation tiles, and every horizontal
row is an eight-stage hardware-cascade chain. Activations are converted to
BFP16 once per resident wave. The north core in each column dequantizes a
16-panel Q4 slab once, writes expanded BFP16 into that column's MemTile, and an
explicit repeated MemTile DMA multicasts it to all four rows. Full `accfloat`
tokens move west-to-east; only the eastern core rounds completed output to
BF16.

```text
Q4 L3 -> north dequantizer -> expanded-BFP MemTile cache (per K column)
                                      || repeated multicast to four rows
BF16 A -> stationary BFP A: [K0] -> [K1] -> ... -> [K7] -> BF16 C
                              accfloat hardware cascade ------>
```

There is no partial-C MemTile writeback and no BF16 rounding between K
partitions.

## Data contract

- `A` is row-major BF16 with shape `M x K`.
- The native weight input contains `N * (K / 256)` consecutive 144-byte
  llama.cpp `block_q4_K` records. It represents a logical `K x N` matrix.
- `prepare_systolic_q4ks_weights` reorders model-load data by physical K-stage
  column, then N panel. Q4 tiles contain 8x8 nibble microtiles followed by
  group-major BF16 effective scale/bias vectors. No zero-point duplication or
  tile padding is used.
- `C` is row-major BF16 with shape `M x N`.
- The POC requires eight columns, `M % (4*m_a) == 0`, `K % 2048 == 0`,
  `K/8 <= 1023`, `N % n == 0`, and `n` equal to 16, 32, or 64. The selected
  path is `m_a=32,n=32,panel_slab=16,c_panel_slab=4`.
- The C++ normalized-data adapter additionally requires `N % 128 == 0` because
  it deliberately reuses the parent harness's validated LLM generator.

The public Python interfaces are:

```python
prepared = prepare_systolic_q4ks_weights(native_B, config)
whole_array_q4ks_systolic(
    A,
    prepared,
    C,
    M=4096,
    K=4096,
    N=4096,
    m_a=32,
    n=32,
    weight_flow="q4-expand-once",
    transport="q4-column-dequant-cache",
    panel_slab=16,
    c_panel_slab=4,
)
```

`whole_array_systolic` is an alias of the callable.

## Weight-flow variants

| Variant | Timed contents | Eligibility |
|---|---|---|
| `q4-expand-once` + `q4-column-dequant-cache` | Dequantize each Q4 slab once per column, cache expanded BFP in MemTile, replay to four rows | Eligible and selected |
| `q4-expand-once` + `q4-expand-broadcast-persistent` | Cache Q4; north row dequantizes per A wave and directly multicasts BFP to rows 1-3 | Eligible, measured and rejected |
| `q4-local` | Q4 transfer/forward and four local dequantizations | Eligible legacy baseline |
| `bfp-prepared` | Pre-expanded BFP transfer and compute | Ineligible model-load ceiling |
| `resident` | One A/B tile is replayed entirely in-core for the nominal operation count | Ineligible compute-issue ceiling |

The selected cache uses explicit MemTile S2MM/MM2S descriptors and native DMA
repeat count, because a synthetic stream ObjectFIFO was charged as a third
core input by the placer. A separate same-offset ObjectFIFO split prototype
remains an explicitly unsafe experiment and is not rerun unless
`--include-unsafe-tile-dma` is supplied.

## Build and run

Every build/run should start from `/home/greg/Desktop`, source the environment,
and then explicitly enter this directory. The environment script's obsolete
final `cd` error is harmless.

```bash
cd /home/greg/Desktop
source /home/greg/Desktop/env-mlir-aie.sh || true
cd /home/greg/Desktop/mlir-aie-fork-experimental/programming_examples/basic/matrix_multiplication_q4ks/whole_array_systolic

make host-tests
make cpp-byte-test

# Direct JIT run with full small-case verification.
make run-python M=256 K=2048 N=512 verify_mode=full

# Build xclbin/instructions and run the full-featured XRT C++ harness.
make run M=256 K=2048 N=512 warmup=1 iters=1 verify=true

# Trace the small functional design.
make trace

# Candidate tables and headline measurements.
make bench-primitives power_mode=turbo
make bench-sweep power_mode=turbo
make perf-native-report power_mode=default
make perf-native power_mode=default     # intentionally enforces 16,000 GFLOP/s
make perf-ceiling power_mode=turbo      # resident ceiling gate
```

`perf-native` intentionally fails while the native path is below the requested
16-TOPS gate. `perf-native-report` records the same three-round protocol without
gating. Weight preparation is outside NPU time; all NPU transfers, BF16-to-BFP
conversion, Q4 dequantization, weight forwarding, cascade accumulation, and
BF16 output conversion are included.

Generated xclbins, instructions, traces, executables, prepared payloads, and
CSV/JSON tables are ignored.

## Correctness

The host suite checks:

- native-to-systolic Q4 round trips at n=16, n=32, and n=64;
- native 6-bit metadata and nibble ordering through the reused parent decoder;
- stage/panel partition coverage and B replay count;
- BFP encoding and Python/C++ byte identity;
- four-row weight forwarding and eight-stage cascade token order;
- token-major-to-row-major C reconstruction;
- phase lifetime overlays and role-specific 64-KiB/512-KiB budgets;
- real DMA/core-memory overflow rejection.

The selected cache passes full verification at `256 x 2048 x 512` with
`rtol=0.05, atol=0.5`. The vectorized verifier reports both the native
Q4_K/BF16 reference and a BFP16-operands/FP32-accumulation reference:

- native-reference maximum absolute error: `0.046875`;
- native-reference NRMSE: `0.00969857`;
- BFP16/FP32-reference maximum absolute error: `0.0742188`;
- BFP16/FP32-reference NRMSE: `0.0156524`;
- BF16 intermediate-writeback drift: exactly zero (no partial writeback).

Maximum relative and BF16-ULP errors can look large for values that cross zero;
absolute error and NRMSE are the useful aggregate metrics here.

## Measured performance

Throughput is `2*M*K*N / NPU_time`. These are end-to-end NPU timings; they do
not include host model-load packing.

| 4096-cubed candidate | NPU time | Throughput | Result |
|---|---:|---:|---|
| resident n=32, rigorous 3-round median | 6,605.23 us | **20.802 TOPS** | Ineligible compute ceiling |
| expanded-BFP column cache, m32/n32, rigorous 3-round median | 20,373.5 us | **6.746 TOPS** | Selected eligible path |
| expanded-BFP column cache, m32/n32, best short run | 20,035.9 us | 6.860 TOPS | Correct, non-gating run |
| expanded-BFP column cache, m32/n64, bounded receiver | 28,612.6 us | 4.803 TOPS | Correct, issue-limited |
| expanded-BFP column cache, m64/n32, FP32 partial spill | 36,687.9 us | 3.746 TOPS | Correct, spill-limited |
| q4-local n=32, historical baseline | 34,518.72 us | 3.982 TOPS | Eligible legacy path |
| bfp-prepared n=32 | 40,998.75 us | 3.352 TOPS | Ineligible |
| compressed-Q4 cache, north dequantize/direct multicast | 52,967.3 us | 2.595 TOPS | Correct, rejected |
| legacy q4-expand-once n=32 | 43,108.50 us | 3.188 TOPS | Eligible legacy path |
| q4-local n=16, post-reboot | 61,200.44 us | 2.246 TOPS | Eligible, rejected |

The 20.802-TOPS result proves that the native BFP16 MMUL plus eight-stage
`accfloat` cascade can exceed 16 TOPS when A and B are resident. Native Q4_K
does not yet meet the 16-TOPS end-to-end gate. The selected design is 1.69x
faster than the historical q4-local path, but remains 2.37x below the gate.

Directly dequantizing the cached Q4 slab on the north compute row and
multicasting BFP16 to the other rows is functionally valid, but it repeats the
conversion for every A wave and reaches only 2.595 TOPS. Therefore dequantizing
once per K-column and retaining the expanded representation is essential.
MemTile performs only storage and repeated DMA multicast; the north compute
worker performs the one-time Q4-to-BFP16 conversion.

The remaining limit is expanded-BFP replay plus A/C movement and scheduling.
The resident result proves that neither native BFP16 issue rate nor the
eight-stage FP32 cascade is the fundamental limit. Larger m=64 and n=64 tiles
reduce transaction count but lower the native issue rate; m=64 also requires
expensive FP32 partial-result spill/reload between bounded weight batches.

The selected native result used 16 warmups, 20 timed iterations, and three
rounds. Round averages were 6,745.98, 6,751.12, and 6,729.96 GFLOP/s, giving a
median of 6,745.98 GFLOP/s. It was measured after the latest reboot while the
system reported Default power mode; the experiment did not change the mode.
Pass the actual setting through `power_mode=...` when recording a new result.

## Driver safety

A normal Chess compile can be quiet for several minutes. A clean design
failure returns an MLIR verifier error or `ERT_CMD_STATE_ABORT`. The known
wedged-driver signature is a Python/XRT process in uninterruptible `D` state
with wait channel `rpm_resume` and no active compiler child. Stop testing and
reboot if that occurs; this experiment never resets or unbinds the driver. An
early invalid repeat-semaphore prototype caused one TDR, after which XRT
recovered and a known-good design passed. All selected-path correctness and
three-round performance runs completed without a TDR.
