//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cxxopts.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#ifndef DTYPE_OUT
#define DTYPE_OUT test_utils::bfloat16_t
#endif
using C_DATATYPE = DTYPE_OUT;
using A_DATATYPE = test_utils::bfloat16_t;

namespace {

constexpr long long verify_stochastic_threshold = 1024LL * 1024 * 1024;

void write_trace(const char *data, std::size_t trace_size,
                 const std::string &path) {
  std::ofstream output(path);
  const auto *words = reinterpret_cast<const std::uint32_t *>(data);
  for (std::size_t index = 0; index < trace_size / sizeof(*words); ++index)
    output << std::setfill('0') << std::setw(8) << std::hex << words[index]
           << '\n';
}

} // namespace

int main(int argc, const char *argv[]) {
  cxxopts::Options options("AWQ INT4 Whole-Array Matmul");
  options.add_options()("h,help", "show help")("x,xclbin", "xclbin path",
                                               cxxopts::value<std::string>())(
      "i,instr", "instruction binary path", cxxopts::value<std::string>())(
      "kernel", "kernel name prefix",
      cxxopts::value<std::string>()->default_value("MLIR_AIE"))(
      "M", "matrix M", cxxopts::value<int>()->default_value("1024"))(
      "K", "matrix K", cxxopts::value<int>()->default_value("1024"))(
      "N", "matrix N", cxxopts::value<int>()->default_value("2048"))(
      "tile-m", "core tile m", cxxopts::value<int>()->default_value("64"))(
      "tile-k", "core tile k", cxxopts::value<int>()->default_value("128"))(
      "tile-n", "core tile n", cxxopts::value<int>()->default_value("64"))(
      "group-size", "AWQ group size",
      cxxopts::value<int>()->default_value("128"))(
      "n-aie-cols", "NPU2 columns", cxxopts::value<int>()->default_value("8"))(
      "v,verbosity", "verbosity", cxxopts::value<int>()->default_value("1"))(
      "verify", "verify output", cxxopts::value<bool>()->default_value("true"))(
      "verify-samples", "sample count",
      cxxopts::value<int>()->default_value("1000"))(
      "warmup", "warmup iterations", cxxopts::value<int>()->default_value("1"))(
      "iters", "timed iterations", cxxopts::value<int>()->default_value("1"))(
      "seed", "deterministic seed",
      cxxopts::value<std::uint32_t>()->default_value("1726250518"))(
      "t,trace_sz", "trace size in bytes",
      cxxopts::value<int>()->default_value("0"))(
      "trace-file", "trace output path",
      cxxopts::value<std::string>()->default_value("trace.txt"));

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
    std::cerr << "--xclbin and --instr are required\n\n"
              << options.help() << '\n';
    return 2;
  }

  awq4::Layout layout{args["M"].as<int>(),
                      args["K"].as<int>(),
                      args["N"].as<int>(),
                      args["tile-m"].as<int>(),
                      args["tile-k"].as<int>(),
                      args["tile-n"].as<int>(),
                      args["group-size"].as<int>(),
                      args["n-aie-cols"].as<int>(),
                      std::is_same_v<C_DATATYPE, float>};
  try {
    layout.validate();
  } catch (const std::exception &error) {
    std::cerr << "Invalid configuration: " << error.what() << '\n';
    return 2;
  }

  const int verbosity = args["verbosity"].as<int>();
  const int warmup = args["warmup"].as<int>();
  const int iterations = args["iters"].as<int>();
  const int trace_size = args["trace_sz"].as<int>();
  if (warmup < 0 || iterations < 1 || trace_size < 0) {
    std::cerr
        << "warmup/trace must be non-negative and iters must be positive\n";
    return 2;
  }

  auto inputs = awq4::make_inputs(layout, args["seed"].as<std::uint32_t>());
  auto packed_b = awq4::pack(layout, inputs);
  std::vector<C_DATATYPE> output(static_cast<std::size_t>(layout.M) * layout.N);
  auto instructions =
      test_utils::load_instr_binary(args["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(args["xclbin"].as<std::string>());
  const std::string kernel_prefix = args["kernel"].as<std::string>();
  auto kernels = xclbin.get_kernels();
  auto found = std::find_if(
      kernels.begin(), kernels.end(), [&](xrt::xclbin::kernel &kernel) {
        return kernel.get_name().rfind(kernel_prefix, 0) == 0;
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
  xrt::bo bo_a(device, inputs.A.size() * sizeof(A_DATATYPE),
               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  xrt::bo bo_b(device, packed_b.size(), XRT_BO_FLAGS_HOST_ONLY,
               kernel.group_id(4));
  xrt::bo bo_c(device, output.size() * sizeof(C_DATATYPE),
               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  xrt::bo bo_trace;
  if (trace_size > 0)
    bo_trace = xrt::bo(device, static_cast<std::size_t>(trace_size) * 4,
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  std::memcpy(bo_instr.map<void *>(), instructions.data(),
              instructions.size() * sizeof(std::uint32_t));
  std::memcpy(bo_a.map<void *>(), inputs.A.data(),
              inputs.A.size() * sizeof(A_DATATYPE));
  std::memcpy(bo_b.map<void *>(), packed_b.data(), packed_b.size());
  std::memset(bo_c.map<void *>(), 0, output.size() * sizeof(C_DATATYPE));
  if (trace_size > 0)
    std::memset(bo_trace.map<void *>(), 0,
                static_cast<std::size_t>(trace_size) * 4);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (trace_size > 0)
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  double total_us = 0.0;
  double min_us = std::numeric_limits<double>::max();
  double max_us = 0.0;
  for (int iteration = 0; iteration < warmup + iterations; ++iteration) {
    const auto start = std::chrono::steady_clock::now();
    xrt::run run(kernel);
    run.set_arg(0, 3U);
    run.set_arg(1, bo_instr);
    run.set_arg(2, instructions.size());
    run.set_arg(3, bo_a);
    run.set_arg(4, bo_b);
    run.set_arg(5, bo_c);
    if (trace_size > 0)
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
      min_us = std::min(min_us, elapsed);
      max_us = std::max(max_us, elapsed);
    }
  }
  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::memcpy(output.data(), bo_c.map<void *>(),
              output.size() * sizeof(C_DATATYPE));
  if (trace_size > 0) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    write_trace(bo_trace.map<const char *>(), trace_size,
                args["trace-file"].as<std::string>());
  }

  const double avg_us = total_us / iterations;
  const double operations =
      2.0 * static_cast<double>(layout.M) * layout.K * layout.N;
  std::cout << "Matrix " << layout.M << 'x' << layout.K << 'x' << layout.N
            << ", tile " << layout.m << 'x' << layout.k << 'x' << layout.n
            << ", group " << layout.group_size << ", columns "
            << layout.n_aie_cols << '\n';
  std::cout << "Packed B bytes: " << packed_b.size() << " ("
            << layout.tile_bytes() << " per tile)\n";
  std::cout << "Avg NPU matmul time: " << avg_us << " us\n";
  std::cout << "Avg NPU gflops: " << operations / (1000.0 * avg_us) << '\n';
  std::cout << "Min/Max NPU time: " << min_us << " / " << max_us << " us\n";

  int errors = 0;
  if (args["verify"].as<bool>()) {
    const long long products =
        static_cast<long long>(layout.M) * layout.K * layout.N;
    const bool sampled = products > verify_stochastic_threshold;
    const int requested_samples = args["verify-samples"].as<int>();
    const int samples = sampled ? requested_samples : layout.M * layout.N;
    std::mt19937 verify_rng(args["seed"].as<std::uint32_t>() ^ 0x4A17U);
    std::uniform_int_distribution<int> row_dist(0, layout.M - 1);
    std::uniform_int_distribution<int> col_dist(0, layout.N - 1);
    for (int sample = 0; sample < samples; ++sample) {
      const int row = sampled ? row_dist(verify_rng) : sample / layout.N;
      const int column = sampled ? col_dist(verify_rng) : sample % layout.N;
      const float expected =
          awq4::reference_value<C_DATATYPE>(layout, inputs, row, column);
      const float actual = awq4::to_float(
          output[static_cast<std::size_t>(row) * layout.N + column]);
      if (!awq4::close(actual, expected)) {
        if (errors < 10)
          std::cerr << "Mismatch [" << row << ", " << column
                    << "]: actual=" << actual << ", expected=" << expected
                    << '\n';
        ++errors;
      }
    }
    if (verbosity)
      std::cout << (sampled ? "Sampled" : "Full") << " verification checked "
                << samples << " outputs\n";
  } else {
    std::cout << "WARNING: verification disabled\n";
  }

  if (errors) {
    std::cerr << "Failed with " << errors << " mismatches\n";
    return 1;
  }
  std::cout << "PASS!\n";
  return 0;
}
