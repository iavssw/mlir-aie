# Whole-array Q4_K implementation

See the [example README](../README.md) for the native llama.cpp ABI, prepared
tile layout, compute paths, cache status, Python/C++ workflows, verification,
tracing, and performance-selection protocol.

The implementation is split into:

- whole_array.py: JIT callable, whole-array schedules, direct runner, and
  verification/benchmark controls.
- q4ks_bf16.cc: the smaller BF16-specialized Chess translation unit.
- q4ks.cc: native BFP16, local-FP32, full-FP32 cascade, split-symbol hybrid
  cascade, and grouped INT8-affine kernels.
- benchmark_sweep.py: compute/cache/accumulation capacity and hardware
  candidate selection.
- gen_pdi_single.sh: compile-only generation of matching PDI/instruction pairs
  for multiple legal M sizes.
- tests/: host ABI/packing/TAP tests and NPU2 compile/run lit coverage.

The Make/PDI default is the measured high-performance
`128/32 x 64 x 128`, eight-column native-BFP16 schedule. Its aligned wave is
512 M rows by 1024 N columns and retains one dequantized Q4_K tile while four
32-row activation subtiles use it. Logical M and N may nevertheless advance
in 256-element steps; the caller does not pad A, native Q4_K B, or C.

## Exact 256-element fringes

Aligned matrices continue to use the original fast whole-array builder. A
specialized builder is selected only when M is not a multiple of 512 or N is
not a multiple of 1024:

- Complete 512x1024 waves retain the same Q4-to-BFP16 conversion reuse.
- At K <= 4096, a 256-row tail sends exactly 64 rows to each physical AIE row.
  No activation or result row is duplicated.
- A 256/512/768-column tail activates exactly two/four/six columns. Inactive
  columns consume only the required A multicast and cannot hold a B/C lock.
- During a complete 512-row wave, each core keeps the first 64x128 C half in a
  16-KiB local buffer and the second in the output FIFO. The halves are emitted
  serially through one legal compute-to-MemTile channel, so the design retains
  the 32-KiB C footprint and dequantizes each B tile only once.
- Runtime task groups are ping-ponged across N panels to overlap descriptor
  setup with active compute.

For K > 4096, the shim's six-bit outer loop and 20-bit inter-half stride need
a bounded fallback. The smallest legal divisor folds 2/4/8 adjacent K tiles
into one MemTile slab (through K=32768), and each core reads one contiguous
128-row region. A 256-row tail shares each 128-row range across a core pair;
the two workers select disjoint 64-row halves, so only activation DMA is
duplicated—MMUL and C are not. If N is simultaneously wide, the inverse C
mapping is exposed as ordered 64-row consumer chunks, keeping every host row
stride at N and every task group at or below 32 transfers.

The combined-fringe hardware test `768x256x1280` checks every one of its
983,040 outputs and passes with direct native-Q4_K NRMSE 0.0159203. The
representative LLM-shaped `768x4096x16384` run passes sampled verification and
averages 13.7065 TOPS (16 warmups, 20 timed iterations). This is close to the
13.998-TOPS two-kernel upper-bound estimate: one third of the M work is a
256-row half-wave, so a 16-TOPS average is not physically expected for that
shape. Fully aligned 512x1024 interiors retain the approximately 16-TOPS path.

The bounded long-K path passes sampled hardware verification at
`768x14336x4096`, averaging 13.797 TOPS in Default mode with direct-Q4_K
NRMSE 0.02599. Simultaneous long-K/wide-N builds pass host TAP tests and Chess
compilation through `768x32768x32768` with slab factor 8; hardware execution
is deferred because the next read-only XRT probe wedged after the long-K run.

After the same reboot, a controlled aligned A/B test measured 15.824 TOPS for
current source and 15.893 TOPS for the archived pre-fringe xclbin in Default
mode (0.43% apart), with a 16.107-TOPS best iteration. `perf-aligned-4096`
keeps a strict 16,000-GFLOP/s gate and should be run in a recorded Turbo
session; use `min_gflops=0` for Default-clock correctness checks.

Use the named checks directly:

~~~bash
make perf-aligned-4096
make perf-fringe-768-wide
~~~

The performance-cascade candidate is `cascade-hybrid` with the
`256/32 x 128 x 64` tile and `l1-weight` streamed-panel scheduling. Two
independent two-row chains keep local prefix partials in BF16, perform each
final K tile and the physical-cascade reduction in `accfloat`, and store BF16
once at each top row. Use `make perf-cascade-hybrid` for the direct
16,000-GFLOP/s research gate and `make perf-cascade-hybrid-3x` for the
three-round median gate. The current Default-mode result is 12.623 TOPS, so
the greater-than-16-TOPS cascade goal is not yet qualified.

## Multi-M PDI generation

`gen_pdi_single.sh` keeps model shapes in a small editable `gen_sizes` array
and tries M values in 256-row increments. It overrides only `M`, `K`, and `N`;
tile size, columns, compute type, accumulation mode, and cache mode come from
the local Makefile. It queries Make for the exact internal artifact suffix, so
there is no duplicate `BUILD_DT` setting to keep synchronized. Copied files
use the external `bf16_q4k_bf16` tensor contract. Generate the listed
PDI/instruction pairs without programming the NPU:

~~~bash
cd /home/greg/Desktop
source env-mlir-aie.sh || true
cd mlir-aie-fork-experimental/programming_examples/basic/matrix_multiplication_q4ks/whole_array
./gen_pdi_single.sh
~~~

The checked-in `mC=128,mA=32,k=64,n=128` schedule accepts every M multiple of
256 through its internal fringe path. The script derives the tile and column
labels from `target_suffix`, so each output filename identifies the kernel
actually compiled.
Outputs are placed below `generated_pdi_insts/<M>x<K>x<N>/<configuration>/`.

The ordinary aligned path retains its original compact TAPs; fringe transfers
use exact host ranges and bounded per-panel task groups. The generator exits
nonzero if any requested PDI or instruction file fails.
