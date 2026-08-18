# Whole-array implementation

See the [example README](../README.md) for the data contract, configuration
rules, Python API, Make/CMake/XRT commands, tracing, tests, and performance
protocol.

The implementation is split into:

- `whole_array.py`: current IRON whole-array dataflow and direct run/verify CLI.
- `awq_4bit.cc`: local AIE2P packed-INT4 dequantization and 8x8x8 BF16 MMUL.
- `test.cpp`: thin include of the self-contained XRT host in the parent.
- `tests/`: host contract, compile-layout, and hardware lit coverage.
