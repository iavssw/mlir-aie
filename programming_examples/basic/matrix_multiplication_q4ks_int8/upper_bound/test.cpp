//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../common.h"

namespace {

constexpr long long stochastic_threshold = 1024LL * 1024 * 1024;

void write_trace(const char *data, std::size_t trace_size,
                 const std::string &path) {
  std::ofstream output(path);
  const auto *words = reinterpret_cast<const std::uint32_t *>(data);
  for (std::size_t i = 0; i < trace_size / sizeof(*words); ++i)
    output << std::setfill('0') << std::setw(8) << std::hex << words[i] << '\n';
}

std::vector<std::int8_t>
quantize_activations(const q4ks::Layout &layout,
                     const std::vector<q4ks::bf16> &input) {
  std::vector<std::int8_t> output(input.size());
  for (int row = 0; row < layout.M; ++row) {
    for (int group0 = 0; group0 < layout.K; group0 += q4ks::group_size) {
      float maximum = 0.0f;
      for (int x = 0; x < q4ks::group_size; ++x) {
        const float value = q4ks::as_float(
            input[static_cast<std::size_t>(row) * layout.K + group0 + x]);
        maximum = std::max(maximum, std::abs(value));
      }
      const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
      for (int x = 0; x < q4ks::group_size; ++x) {
        const auto index =
            static_cast<std::size_t>(row) * layout.K + group0 + x;
        const float value = q4ks::as_float(input[index]) / scale;
        const int rounded = value >= 0.0f ? static_cast<int>(value + 0.5f)
                                          : static_cast<int>(value - 0.5f);
        output[index] =
            static_cast<std::int8_t>(std::clamp(rounded, -127, 127));
      }
    }
  }
  return output;
}

std::vector<std::int8_t> q4_codes_column_major(const q4ks::Layout &layout,
                                               const q4ks::Decoded &decoded) {
  std::vector<std::int8_t> output(static_cast<std::size_t>(layout.K) *
                                  layout.N);
  for (int column = 0; column < layout.N; ++column)
    for (int inner = 0; inner < layout.K; ++inner)
      output[static_cast<std::size_t>(column) * layout.K + inner] =
          static_cast<std::int8_t>(
              decoded.q[static_cast<std::size_t>(inner) * layout.N + column]);
  return output;
}

std::int32_t reference_dot(const q4ks::Layout &layout,
                           const std::vector<std::int8_t> &activation,
                           const std::vector<std::int8_t> &weights, int row,
                           int column, int inner_begin = 0) {
  std::int32_t sum = 0;
  for (int inner = inner_begin; inner < layout.K; ++inner)
    sum += static_cast<std::int32_t>(
               activation[static_cast<std::size_t>(row) * layout.K + inner]) *
           static_cast<std::int32_t>(
               weights[static_cast<std::size_t>(column) * layout.K + inner]);
  return sum;
}

} // namespace

int main(int argc, const char *argv[]) {
  cxxopts::Options options("Q4_K INT8 Whole-Array Compute Ceiling");
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "kernel prefix",
      cxxopts::value<std::string>()->default_value("MLIR_AIE"))(
      "instr,i", "instruction binary",
      cxxopts::value<std::string>())("verbosity,v", "output verbosity",
                                     cxxopts::value<int>()->default_value("0"))(
      "verify", "verify integer dot products",
      cxxopts::value<bool>()->default_value("true"))(
      "rows,M", "matrix M", cxxopts::value<int>()->default_value("4096"))(
      "inner,K", "matrix K", cxxopts::value<int>()->default_value("4096"))(
      "columns,N", "matrix N", cxxopts::value<int>()->default_value("4096"))(
      "iters", "timed iterations", cxxopts::value<int>()->default_value("1"))(
      "warmup", "warmup iterations", cxxopts::value<int>()->default_value("1"))(
      "verify-samples", "stochastic verification samples",
      cxxopts::value<int>()->default_value("1000"))(
      "seed", "deterministic input seed",
      cxxopts::value<std::uint32_t>()->default_value("1263755060"))(
      "min-gops", "minimum accepted GOP/s",
      cxxopts::value<double>()->default_value("0"))(
      "trace_sz,t", "trace size", cxxopts::value<int>()->default_value("0"))(
      "trace_file", "trace output path",
      cxxopts::value<std::string>()->default_value("trace.txt"))(
      "kernel-repeats", "resident zero+MMUL repetitions per K tile",
      cxxopts::value<int>()->default_value("1"));

  cxxopts::ParseResult args;
  try {
    args = options.parse(argc, argv);
  } catch (const cxxopts::exceptions::parsing &error) {
    std::cerr << error.what() << "\n\n" << options.help() << '\n';
    return 2;
  }
  if (args.count("help")) {
    std::cout << options.help() << '\n';
    return 0;
  }
  if (!args.count("xclbin") || !args.count("instr")) {
    std::cerr << "--xclbin and --instr are required\n\n" << options.help();
    return 2;
  }

  q4ks::Layout layout;
  layout.M = args["M"].as<int>();
  layout.K = args["K"].as<int>();
  layout.N = args["N"].as<int>();
  layout.m_c = 64;
  layout.m_a = 32;
  layout.k = 64;
  layout.n = 64;
  layout.n_aie_cols = 8;
  layout.compute_type = "int8";
  layout.cache_mode = "stream";
  layout.cache_k = layout.K;
  try {
    layout.validate();
  } catch (const std::exception &error) {
    std::cerr << "Invalid configuration: " << error.what() << '\n';
    return 2;
  }

  const int verbosity = args["verbosity"].as<int>();
  const int warmup = args["warmup"].as<int>();
  const int iterations = args["iters"].as<int>();
  const int samples = args["verify-samples"].as<int>();
  const int trace_size = args["trace_sz"].as<int>();
  const double min_gops = args["min-gops"].as<double>();
  const int kernel_repeats = args["kernel-repeats"].as<int>();
  if (verbosity < 0 || warmup < 0 || iterations < 1 || samples < 1 ||
      trace_size < 0 || min_gops < 0 || kernel_repeats < 1) {
    std::cerr << "invalid iteration, verification, trace, or gate value\n";
    return 2;
  }

  if (verbosity >= 1)
    std::cout << "Generating normalized BF16 activations and native Q4_K "
                 "weights, then prequantizing outside the timed region.\n";
  auto inputs = q4ks::make_inputs(layout, args["seed"].as<std::uint32_t>());
  const auto decoded = q4ks::decode(layout, inputs.native_B);
  const auto activation = quantize_activations(layout, inputs.A);
  const auto weights = q4_codes_column_major(layout, decoded);
  std::vector<std::int32_t> output(static_cast<std::size_t>(layout.M) *
                                   layout.N);
  auto instructions =
      test_utils::load_instr_binary(args["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(args["xclbin"].as<std::string>());
  const auto kernel_prefix = args["kernel"].as<std::string>();
  auto kernels = xclbin.get_kernels();
  auto found = std::find_if(
      kernels.begin(), kernels.end(), [&](xrt::xclbin::kernel &candidate) {
        return candidate.get_name().rfind(kernel_prefix, 0) == 0;
      });
  if (found == kernels.end()) {
    std::cerr << "No kernel begins with '" << kernel_prefix << "'\n";
    return 2;
  }
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  xrt::kernel kernel(context, found->get_name());

  xrt::bo bo_instr(device, instructions.size() * sizeof(std::uint32_t),
                   XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  xrt::bo bo_a(device, activation.size(), XRT_BO_FLAGS_HOST_ONLY,
               kernel.group_id(3));
  xrt::bo bo_b(device, weights.size(), XRT_BO_FLAGS_HOST_ONLY,
               kernel.group_id(4));
  xrt::bo bo_c(device, output.size() * sizeof(std::int32_t),
               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  xrt::bo bo_trace;
  if (trace_size)
    bo_trace = xrt::bo(device, static_cast<std::size_t>(trace_size) * 4,
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  std::memcpy(bo_instr.map<void *>(), instructions.data(),
              instructions.size() * sizeof(std::uint32_t));
  std::memcpy(bo_a.map<void *>(), activation.data(), activation.size());
  std::memcpy(bo_b.map<void *>(), weights.data(), weights.size());
  std::memset(bo_c.map<void *>(), 0, output.size() * sizeof(std::int32_t));
  if (trace_size)
    std::memset(bo_trace.map<void *>(), 0,
                static_cast<std::size_t>(trace_size) * 4);
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (trace_size)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  double total_us = 0.0;
  double minimum_us = std::numeric_limits<double>::max();
  double maximum_us = 0.0;
  for (int iteration = 0; iteration < warmup + iterations; ++iteration) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iteration << ").\n";
    auto start = std::chrono::steady_clock::now();
    xrt::run run(kernel);
    run.set_arg(0, 3U);
    run.set_arg(1, bo_instr);
    run.set_arg(2, instructions.size());
    run.set_arg(3, bo_a);
    run.set_arg(4, bo_b);
    run.set_arg(5, bo_c);
    if (trace_size)
      run.set_arg(6, bo_trace);
    run.start();
    const auto state = run.wait();
    const auto stop = std::chrono::steady_clock::now();
    if (state != ERT_CMD_STATE_COMPLETED) {
      std::cerr << "Kernel failed with state " << state << '\n';
      return 1;
    }
    if (iteration >= warmup) {
      const double elapsed =
          std::chrono::duration<double, std::micro>(stop - start).count();
      total_us += elapsed;
      minimum_us = std::min(minimum_us, elapsed);
      maximum_us = std::max(maximum_us, elapsed);
    }
  }

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(output.data(), bo_c.map<void *>(),
              output.size() * sizeof(std::int32_t));
  if (trace_size) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    write_trace(bo_trace.map<const char *>(), trace_size,
                args["trace_file"].as<std::string>());
  }

  const double average_us = total_us / iterations;
  const double operations = 2.0 * static_cast<double>(layout.M) * layout.K *
                            layout.N * kernel_repeats;
  const double gops = operations / (1000.0 * average_us);
  std::cout << "\nQ4-derived prequantized INT8 ceiling " << layout.M << 'x'
            << layout.K << 'x' << layout.N << '\n';
  std::cout << "Avg NPU time: " << average_us << "us\n";
  std::cout << "Avg effective GOP/s: " << gops << '\n';
  std::cout << "Range: " << operations / (1000.0 * maximum_us) << "--"
            << operations / (1000.0 * minimum_us) << " GOP/s\n";
  std::cout << "Excluded: BF16->INT8 quantization, affine Q4_K correction, "
               "and INT32->BF16 output conversion.\n";
  if (kernel_repeats > 1)
    std::cout << "Resident mode: each A/B K tile is held in L1 for "
              << kernel_repeats
              << " zero+MMUL repetitions; host and MemTile traffic are "
                 "amortized.\n";

  int errors = 0;
  if (args["verify"].as<bool>()) {
    const bool stochastic =
        static_cast<long long>(layout.M) * layout.K * layout.N >
        stochastic_threshold;
    const int count = stochastic ? samples : layout.M * layout.N;
    std::mt19937 rng(args["seed"].as<std::uint32_t>() ^ 0x49384345U);
    std::uniform_int_distribution<int> rows(0, layout.M - 1);
    std::uniform_int_distribution<int> columns(0, layout.N - 1);
    for (int sample = 0; sample < count; ++sample) {
      const int row = stochastic ? rows(rng) : sample / layout.N;
      const int column = stochastic ? columns(rng) : sample % layout.N;
      const int inner_begin = kernel_repeats == 1 ? 0 : layout.K - layout.k;
      const auto expected =
          reference_dot(layout, activation, weights, row, column, inner_begin);
      const auto actual =
          output[static_cast<std::size_t>(row) * layout.N + column];
      if (actual != expected) {
        if (errors < 10)
          std::cerr << "Mismatch [" << row << ", " << column
                    << "]: actual=" << actual << ", expected=" << expected
                    << '\n';
        ++errors;
      }
    }
    std::cout << (stochastic ? "Sampled" : "Full")
              << " integer-dot verification: " << (errors ? "FAIL" : "PASS")
              << '\n';
  } else {
    std::cout << "WARNING: verification disabled\n";
  }

  if (errors)
    return 1;
  if (min_gops && gops < min_gops) {
    std::cerr << "Performance gate failed: " << gops << " < " << min_gops
              << " GOP/s\n";
    return 1;
  }
  std::cout << "PASS!\n";
  return 0;
}
