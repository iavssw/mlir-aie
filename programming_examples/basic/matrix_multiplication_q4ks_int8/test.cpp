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

#include "common.h"

namespace {

constexpr long long verify_stochastic_threshold = 1024LL * 1024 * 1024;

struct DistributionStats {
  double mean;
  double rms;
  double l2;
};

template <typename ValueAt>
DistributionStats distribution_stats(std::size_t count, ValueAt value_at) {
  double sum = 0.0;
  double square_sum = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const double value = value_at(index);
    sum += value;
    square_sum += value * value;
  }
  return {sum / count, std::sqrt(square_sum / count), std::sqrt(square_sum)};
}

void write_trace(const char *data, std::size_t trace_size,
                 const std::string &path) {
  std::ofstream output(path);
  const auto *words = reinterpret_cast<const std::uint32_t *>(data);
  for (std::size_t i = 0; i < trace_size / sizeof(*words); ++i)
    output << std::setfill('0') << std::setw(8) << std::hex << words[i] << '\n';
}

std::vector<std::uint8_t> read_bytes(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("cannot open native Q4_K file: " + path);
  const auto size = input.tellg();
  if (size < 0)
    throw std::runtime_error("cannot determine native Q4_K file size");
  input.seekg(0);
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char *>(result.data()), size);
  if (!input)
    throw std::runtime_error("cannot read native Q4_K file");
  return result;
}

void write_bytes(const std::string &path,
                 const std::vector<std::uint8_t> &values) {
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot create prepared Q4_K file: " + path);
  output.write(reinterpret_cast<const char *>(values.data()), values.size());
  if (!output)
    throw std::runtime_error("cannot write prepared Q4_K file");
}

} // namespace

int main(int argc, const char *argv[]) {
  // Match matrix_multiplication/test.cpp's host interface and add only the
  // Q4_K-specific packing, tile, and performance controls.
  cxxopts::Options options("Matrix Matrix Multiplication Q4_K Test");
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN",
      cxxopts::value<std::string>()->default_value("MLIR_AIE"))(
      "verbosity,v", "the verbosity of the output",
      cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions sent to the NPU",
      cxxopts::value<std::string>())(
      "verify", "whether to verify the AIE computed output",
      cxxopts::value<bool>()->default_value("true"))(
      "rows,M", "Matrix size M", cxxopts::value<int>()->default_value("512"))(
      "inner,K", "Matrix size K", cxxopts::value<int>()->default_value("512"))(
      "columns,N", "Matrix size N",
      cxxopts::value<int>()->default_value("512"))(
      "iters", "number of iterations",
      cxxopts::value<int>()->default_value("1"))(
      "warmup", "number of warmup iterations",
      cxxopts::value<int>()->default_value("0"))(
      "trace_sz,t", "trace size", cxxopts::value<int>()->default_value("0"))(
      "trace_file", "where to store trace output",
      cxxopts::value<std::string>()->default_value("trace.txt"))(
      "b_col_maj", "Is B matrix in column-major format?",
      cxxopts::value<int>()->default_value("0"))(
      "c_col_maj", "Is C matrix in column-major format?",
      cxxopts::value<int>()->default_value("0"))(
      "tile-m-c", "C tile M", cxxopts::value<int>()->default_value("64"))(
      "tile-m-a", "A subtile M", cxxopts::value<int>()->default_value("32"))(
      "tile-k", "tile K", cxxopts::value<int>()->default_value("128"))(
      "tile-n", "tile N", cxxopts::value<int>()->default_value("64"))(
      "n-aie-cols", "NPU2 columns", cxxopts::value<int>()->default_value("8"))(
      "compute-type", "bf16, bfp16, or int8",
      cxxopts::value<std::string>()->default_value("bf16"))(
      "cache-mode", "stream, l1-weight, or memtile-weight",
      cxxopts::value<std::string>()->default_value("stream"))(
      "cache-k", "resident K extent (zero means K)",
      cxxopts::value<int>()->default_value("0"))(
      "q4-k-file", "flat native block_q4_K payload",
      cxxopts::value<std::string>()->default_value(""))(
      "prepare-output", "prepare weights, write bytes, and exit",
      cxxopts::value<std::string>()->default_value(""))(
      "verify-samples", "sample count",
      cxxopts::value<int>()->default_value("1000"))(
      "seed", "deterministic seed",
      cxxopts::value<std::uint32_t>()->default_value("1263755060"))(
      "min-gflops", "minimum accepted throughput",
      cxxopts::value<double>()->default_value("0"));

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
  q4ks::Layout layout;
  layout.M = args["M"].as<int>();
  layout.K = args["K"].as<int>();
  layout.N = args["N"].as<int>();
  layout.m_c = args["tile-m-c"].as<int>();
  layout.m_a = args["tile-m-a"].as<int>();
  layout.k = args["tile-k"].as<int>();
  layout.n = args["tile-n"].as<int>();
  layout.n_aie_cols = args["n-aie-cols"].as<int>();
  layout.compute_type = args["compute-type"].as<std::string>();
  layout.cache_mode = args["cache-mode"].as<std::string>();
  layout.cache_k = args["cache-k"].as<int>();
  if (layout.cache_k == 0)
    layout.cache_k = layout.K;
  try {
    layout.validate();
  } catch (const std::exception &error) {
    std::cerr << "Invalid configuration: " << error.what() << '\n';
    return 2;
  }

  const int verbosity = args["verbosity"].as<int>();
  const bool do_verify = args["verify"].as<bool>();
  const int n_warmup_iterations = args["warmup"].as<int>();
  const int n_iterations = args["iters"].as<int>();
  const int trace_size = args["trace_sz"].as<int>();
  const double min_gflops = args["min-gflops"].as<double>();
  const int b_col_maj = args["b_col_maj"].as<int>();
  const int c_col_maj = args["c_col_maj"].as<int>();
  if (verbosity < 0 || n_warmup_iterations < 0 || n_iterations < 1 ||
      trace_size < 0 || min_gflops < 0 ||
      args["verify-samples"].as<int>() < 1) {
    std::cerr << "warmup/trace/gate must be non-negative and iters positive\n";
    return 2;
  }
  if (b_col_maj || c_col_maj) {
    std::cerr << "Q4_K uses a fixed KxN weight contract and row-major C; "
                 "b_col_maj and c_col_maj must both be zero\n";
    return 2;
  }
  if (verbosity >= 1)
    std::cout << "Matrix size " << layout.M << 'x' << layout.K << 'x'
              << layout.N << '\n';

  auto inputs = q4ks::make_inputs(layout, args["seed"].as<std::uint32_t>());
  const auto q4_path = args["q4-k-file"].as<std::string>();
  if (!q4_path.empty())
    inputs.native_B = read_bytes(q4_path);
  std::vector<std::uint8_t> prepared;
  q4ks::Decoded decoded;
  try {
    prepared = q4ks::prepare_weights(layout, inputs.native_B);
    decoded = q4ks::decode(layout, inputs.native_B);
  } catch (const std::exception &error) {
    std::cerr << "Q4_K preparation failed: " << error.what() << '\n';
    return 2;
  }
  if (verbosity >= 1) {
    const auto a_stats = distribution_stats(layout.K, [&](std::size_t inner) {
      return q4ks::as_float(inputs.A[inner]);
    });
    const auto b_stats = distribution_stats(layout.K, [&](std::size_t inner) {
      const auto group = inner / q4ks::group_size;
      const auto parameter = group * static_cast<std::size_t>(layout.N);
      const float scale =
          q4ks::as_float(q4ks::as_bf16(decoded.scales[parameter]));
      const float bias =
          q4ks::as_float(q4ks::as_bf16(decoded.biases[parameter]));
      return decoded.q[inner * layout.N] * scale - bias;
    });
    std::cout << "Normalized input profile: A row 0 mean=" << a_stats.mean
              << ", RMS=" << a_stats.rms
              << "; dequantized B column 0 mean=" << b_stats.mean
              << ", L2=" << b_stats.l2 << '\n';
  }
  const auto prepare_output = args["prepare-output"].as<std::string>();
  if (!prepare_output.empty()) {
    try {
      write_bytes(prepare_output, prepared);
    } catch (const std::exception &error) {
      std::cerr << error.what() << '\n';
      return 2;
    }
    std::cout << "Wrote " << prepared.size() << " prepared bytes\nPASS!\n";
    return 0;
  }
  if (!args.count("xclbin") || !args.count("instr")) {
    std::cerr << "--xclbin and --instr are required unless --prepare-output is "
                 "used\n\n"
              << options.help() << '\n';
    return 2;
  }
  std::vector<q4ks::bf16> output(static_cast<std::size_t>(layout.M) * layout.N);
  auto instructions =
      test_utils::load_instr_binary(args["instr"].as<std::string>());

  if (verbosity >= 1) {
    std::cout << "Sequence instr count: " << instructions.size() << '\n';
    std::cout << "Loading xclbin: " << args["xclbin"].as<std::string>() << '\n';
  }

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(args["xclbin"].as<std::string>());
  const auto kernel_prefix = args["kernel"].as<std::string>();
  auto kernels = xclbin.get_kernels();
  auto found = std::find_if(
      kernels.begin(), kernels.end(), [&](xrt::xclbin::kernel &kernel) {
        return kernel.get_name().rfind(kernel_prefix, 0) == 0;
      });
  if (found == kernels.end()) {
    std::cerr << "No kernel begins with '" << kernel_prefix << "'\n";
    return 2;
  }
  if (verbosity >= 1)
    std::cout << "Using kernel: " << found->get_name() << '\n';
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  xrt::kernel kernel(context, found->get_name());

  xrt::bo bo_instr(device, instructions.size() * sizeof(std::uint32_t),
                   XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  xrt::bo bo_a(device, inputs.A.size() * sizeof(q4ks::bf16),
               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  xrt::bo bo_b(device, prepared.size(), XRT_BO_FLAGS_HOST_ONLY,
               kernel.group_id(4));
  xrt::bo bo_c(device, output.size() * sizeof(q4ks::bf16),
               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  xrt::bo bo_trace;
  if (trace_size)
    bo_trace = xrt::bo(device, static_cast<std::size_t>(trace_size) * 4,
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  std::memcpy(bo_instr.map<void *>(), instructions.data(),
              instructions.size() * sizeof(std::uint32_t));
  std::memcpy(bo_a.map<void *>(), inputs.A.data(),
              inputs.A.size() * sizeof(q4ks::bf16));
  std::memcpy(bo_b.map<void *>(), prepared.data(), prepared.size());
  std::memset(bo_c.map<void *>(), 0, output.size() * sizeof(q4ks::bf16));
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
  double min_us = std::numeric_limits<double>::max();
  double max_us = 0.0;
  for (int iteration = 0; iteration < n_warmup_iterations + n_iterations;
       ++iteration) {
    if (verbosity >= 1)
      std::cout << "Running Kernel (iteration " << iteration << ").\n";
    const auto start = std::chrono::steady_clock::now();
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
    if (iteration >= n_warmup_iterations) {
      const double elapsed =
          std::chrono::duration<double, std::micro>(stop - start).count();
      total_us += elapsed;
      min_us = std::min(min_us, elapsed);
      max_us = std::max(max_us, elapsed);
    }
  }
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(output.data(), bo_c.map<void *>(),
              output.size() * sizeof(q4ks::bf16));
  if (trace_size) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    write_trace(bo_trace.map<const char *>(), trace_size,
                args["trace_file"].as<std::string>());
  }

  const double average_us = total_us / n_iterations;
  const double operations =
      2.0 * static_cast<double>(layout.M) * layout.K * layout.N;
  const double gflops = operations / (1000.0 * average_us);
  std::cout << "Matrix " << layout.M << 'x' << layout.K << 'x' << layout.N
            << ", tile " << layout.m_c << '/' << layout.m_a << 'x' << layout.k
            << 'x' << layout.n << ", " << layout.compute_type << ", "
            << layout.cache_mode << ", " << layout.n_aie_cols << " columns\n";
  std::cout << "Native/prepared B bytes: " << inputs.native_B.size() << " / "
            << prepared.size() << " (" << layout.tile_bytes() << " per tile)\n";
  std::cout << "\nAvg NPU matmul time: " << average_us << "us.\n";
  std::cout << "Avg NPU gflops: " << gflops << '\n';
  std::cout << "\nMin NPU matmul time: " << min_us << "us.\n";
  std::cout << "Max NPU gflops: " << operations / (1000.0 * min_us) << '\n';
  std::cout << "\nMax NPU matmul time: " << max_us << "us.\n";
  std::cout << "Min NPU gflops: " << operations / (1000.0 * max_us) << '\n';

  int errors = 0;
  double max_abs = 0.0;
  double max_rel = 0.0;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  int native_errors = 0;
  double native_max_abs = 0.0;
  double native_max_rel = 0.0;
  double native_squared_error = 0.0;
  double native_squared_reference = 0.0;
  if (do_verify) {
    const bool sampled =
        static_cast<long long>(layout.M) * layout.K * layout.N >
        verify_stochastic_threshold;
    const int sample_count =
        sampled ? args["verify-samples"].as<int>() : layout.M * layout.N;
    std::mt19937 rng(args["seed"].as<std::uint32_t>() ^ 0x5134U);
    std::uniform_int_distribution<int> rows(0, layout.M - 1);
    std::uniform_int_distribution<int> cols(0, layout.N - 1);
    auto native_reference_layout = layout;
    native_reference_layout.compute_type = "bf16";
    for (int sample = 0; sample < sample_count; ++sample) {
      const int row = sampled ? rows(rng) : sample / layout.N;
      const int col = sampled ? cols(rng) : sample % layout.N;
      const float expected =
          q4ks::reference_value(layout, inputs, decoded, row, col);
      const float actual = q4ks::as_float(
          output[static_cast<std::size_t>(row) * layout.N + col]);
      const double difference = std::abs(actual - expected);
      max_abs = std::max(max_abs, difference);
      max_rel = std::max(max_rel,
                         difference / std::max(std::abs(expected), 1.0e-12f));
      squared_error += difference * difference;
      squared_reference += static_cast<double>(expected) * expected;
      if (!q4ks::close(actual, expected)) {
        if (errors < 10)
          std::cerr << "Mismatch [" << row << ", " << col
                    << "]: actual=" << actual << ", expected=" << expected
                    << '\n';
        ++errors;
      }

      const float native_expected = q4ks::reference_value(
          native_reference_layout, inputs, decoded, row, col);
      const double native_difference = std::abs(actual - native_expected);
      native_max_abs = std::max(native_max_abs, native_difference);
      native_max_rel = std::max(
          native_max_rel,
          native_difference / std::max(std::abs(native_expected), 1.0e-12f));
      native_squared_error += native_difference * native_difference;
      native_squared_reference +=
          static_cast<double>(native_expected) * native_expected;
      if (!q4ks::close(actual, native_expected))
        ++native_errors;
    }
    const double nrmse =
        std::sqrt(squared_error / sample_count) /
        std::max(std::sqrt(squared_reference / sample_count), 1.0e-12);
    std::cout << (sampled ? "Sampled" : "Full")
              << " INT8-model verification: max_abs=" << max_abs
              << ", max_rel=" << max_rel << ", nrmse=" << nrmse << '\n';
    const double native_nrmse =
        std::sqrt(native_squared_error / sample_count) /
        std::max(std::sqrt(native_squared_reference / sample_count), 1.0e-12);
    std::cout << (sampled ? "Sampled" : "Full")
              << " dequantized-Q4_K BF16-reference verification: max_abs="
              << native_max_abs << ", max_rel=" << native_max_rel
              << ", nrmse=" << native_nrmse << ", mismatches=" << native_errors
              << '\n';
  } else {
    std::cout << "WARNING: verification disabled\n";
  }
  if (errors || native_errors) {
    std::cerr << "Failed with " << errors << " INT8-model and " << native_errors
              << " native-reference mismatches\n";
    return 1;
  }
  if (min_gflops && gflops < min_gflops) {
    std::cerr << "Performance gate failed: " << gflops << " < " << min_gflops
              << " GFLOP/s\n";
    return 1;
  }
  std::cout << "PASS!\n";
  return 0;
}
