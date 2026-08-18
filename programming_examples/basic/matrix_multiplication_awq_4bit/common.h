//===- common.h -------------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATRIX_MULTIPLICATION_AWQ_4BIT_COMMON_H
#define MATRIX_MULTIPLICATION_AWQ_4BIT_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "test_utils.h"

namespace awq4 {

constexpr int n_aie_rows = 4;

inline std::size_t ceildiv(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1) / divisor;
}

struct Layout {
  int M;
  int K;
  int N;
  int m;
  int k;
  int n;
  int group_size;
  int n_aie_cols;
  bool output_f32;

  void validate() const {
    if (M <= 0 || K <= 0 || N <= 0 || m <= 0 || k <= 0 || n <= 0)
      throw std::invalid_argument(
          "matrix and tile dimensions must be positive");
    if (group_size != 32 && group_size != 64 && group_size != 128)
      throw std::invalid_argument("group size must be 32, 64, or 128");
    if (n_aie_cols != 1 && n_aie_cols != 2 && n_aie_cols != 4 &&
        n_aie_cols != 8)
      throw std::invalid_argument("NPU2 columns must be 1, 2, 4, or 8");
    if (m % 32 || k % 8 || n % 16)
      throw std::invalid_argument("tile must satisfy m%32=0, k%8=0, n%16=0");
    if (k % group_size || K % group_size)
      throw std::invalid_argument("group size must divide k and K");
    if (M % (m * n_aie_rows) || K % k || N % (n * n_aie_cols))
      throw std::invalid_argument(
          "matrix dimensions are incompatible with tiling");
    if ((M / (m * n_aie_rows)) % 2)
      throw std::invalid_argument("M/(m*4) must be even");
    const std::size_t rows_per_a_shim =
        n_aie_cols < n_aie_rows ? n_aie_rows / n_aie_cols : 1;
    if (static_cast<std::size_t>(K / k) > 1023 ||
        static_cast<std::size_t>(N / (n * n_aie_cols)) > 1023 ||
        packed_rows() > 1023 || static_cast<std::size_t>(k) > 1023 ||
        static_cast<std::size_t>(m) * rows_per_a_shim > 1023 ||
        static_cast<std::size_t>(n_aie_rows) * m > 1023 ||
        static_cast<std::size_t>(n) > 1023)
      throw std::invalid_argument("DMA size exceeds the 10-bit limit");
    if (core_memory_bytes() > 64 * 1024)
      throw std::invalid_argument("configuration exceeds 64-KiB core memory");
  }

  std::size_t groups_per_tile() const { return k / group_size; }
  std::size_t weight_bytes_per_tile() const { return k * n / 2; }
  std::size_t scale_bytes_per_tile() const {
    return groups_per_tile() * n * sizeof(test_utils::bfloat16_t);
  }
  std::size_t zero_bytes_per_tile() const { return groups_per_tile() * n * 2; }
  std::size_t raw_tile_bytes() const {
    return weight_bytes_per_tile() + scale_bytes_per_tile() +
           zero_bytes_per_tile();
  }
  std::size_t packed_rows() const { return ceildiv(raw_tile_bytes(), k); }
  std::size_t tile_bytes() const { return packed_rows() * k; }
  std::size_t packed_bytes() const {
    return static_cast<std::size_t>(K / k) * (N / n) * tile_bytes();
  }
  std::size_t core_memory_bytes() const {
    const std::size_t a_fifo = 2ULL * (m / 2) * k * 2;
    const std::size_t b_fifo = 2ULL * tile_bytes();
    const std::size_t c_depth = output_f32 ? 1 : 2;
    const std::size_t c_fifo =
        c_depth * static_cast<std::size_t>(m) * n * (output_f32 ? 4 : 2);
    const std::size_t scratch = static_cast<std::size_t>(k) * n * 2;
    return a_fifo + b_fifo + c_fifo + scratch + 0xD00 + 4 * 1024;
  }
};

struct Inputs {
  std::vector<test_utils::bfloat16_t> A;
  std::vector<std::uint8_t> qweight;
  std::vector<test_utils::bfloat16_t> scales;
  std::vector<std::uint8_t> zeros;
};

inline Inputs make_inputs(const Layout &layout,
                          std::uint32_t seed = 1726250518U) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> a_dist(-0.25f, 0.25f);
  std::uniform_real_distribution<float> scale_dist(0.005f, 0.05f);
  std::uniform_int_distribution<int> nibble_dist(0, 15);
  Inputs result;
  result.A.resize(static_cast<std::size_t>(layout.M) * layout.K);
  result.qweight.resize(static_cast<std::size_t>(layout.K) * layout.N);
  result.scales.resize(static_cast<std::size_t>(layout.K / layout.group_size) *
                       layout.N);
  result.zeros.resize(result.scales.size());
  for (auto &value : result.A)
    value = test_utils::bfloat16_from_float(a_dist(rng));
  for (auto &value : result.qweight)
    value = static_cast<std::uint8_t>(nibble_dist(rng));
  for (auto &value : result.scales)
    value = test_utils::bfloat16_from_float(scale_dist(rng));
  for (auto &value : result.zeros)
    value = static_cast<std::uint8_t>(nibble_dist(rng));
  return result;
}

inline std::vector<std::uint8_t> pack(const Layout &layout,
                                      const Inputs &inputs) {
  layout.validate();
  const std::size_t q_size = static_cast<std::size_t>(layout.K) * layout.N;
  const std::size_t parameter_size =
      static_cast<std::size_t>(layout.K / layout.group_size) * layout.N;
  if (inputs.qweight.size() != q_size ||
      inputs.scales.size() != parameter_size ||
      inputs.zeros.size() != parameter_size)
    throw std::invalid_argument("logical AWQ input size mismatch");
  if (std::any_of(inputs.qweight.begin(), inputs.qweight.end(),
                  [](std::uint8_t value) { return value > 15; }) ||
      std::any_of(inputs.zeros.begin(), inputs.zeros.end(),
                  [](std::uint8_t value) { return value > 15; }))
    throw std::invalid_argument("weights and zero points must be in 0..15");

  std::vector<std::uint8_t> packed(layout.packed_bytes(), 0);
  const int n_k_tiles = layout.K / layout.k;
  const int n_n_tiles = layout.N / layout.n;
  const int n_rounds = n_n_tiles / layout.n_aie_cols;
  std::size_t tile_index = 0;
  for (int col = 0; col < layout.n_aie_cols; ++col) {
    for (int n_round = 0; n_round < n_rounds; ++n_round) {
      const int n_tile = col + n_round * layout.n_aie_cols;
      for (int k_tile = 0; k_tile < n_k_tiles; ++k_tile, ++tile_index) {
        const int k0 = k_tile * layout.k;
        const int n0 = n_tile * layout.n;
        const std::size_t tile_start = tile_index * layout.tile_bytes();
        std::size_t cursor = tile_start;

        for (int micro_k = 0; micro_k < layout.k; micro_k += 8) {
          for (int micro_n = 0; micro_n < layout.n; micro_n += 8) {
            for (int row = 0; row < 8; ++row) {
              for (int lane = 0; lane < 8; lane += 2) {
                const std::size_t logical =
                    static_cast<std::size_t>(k0 + micro_k + row) * layout.N +
                    n0 + micro_n + lane;
                const auto low = inputs.qweight[logical];
                const auto high = inputs.qweight[logical + 1];
                packed[cursor++] = low | static_cast<std::uint8_t>(high << 4);
              }
            }
          }
        }

        const int first_group = k0 / layout.group_size;
        for (int group = 0; group < layout.k / layout.group_size; ++group) {
          for (int column = 0; column < layout.n; ++column) {
            const auto scale =
                inputs.scales[static_cast<std::size_t>(first_group + group) *
                                  layout.N +
                              n0 + column];
            std::memcpy(packed.data() + cursor, &scale, sizeof(scale));
            cursor += sizeof(scale);
          }
        }

        for (int group = 0; group < layout.k / layout.group_size; ++group) {
          for (int micro_n = 0; micro_n < layout.n; micro_n += 8) {
            const std::size_t logical =
                static_cast<std::size_t>(first_group + group) * layout.N + n0 +
                micro_n;
            for (int repeat = 0; repeat < 2; ++repeat)
              for (int lane = 0; lane < 8; ++lane)
                packed[cursor++] = inputs.zeros[logical + lane];
          }
        }
        if (cursor - tile_start != layout.raw_tile_bytes())
          throw std::logic_error("internal AWQ packing size mismatch");
      }
    }
  }
  return packed;
}

template <typename T>
inline float to_float(T value) {
  if constexpr (std::is_same_v<T, test_utils::bfloat16_t>)
    return test_utils::bfloat16_to_float(value);
  else
    return static_cast<float>(value);
}

template <typename C>
inline float reference_value(const Layout &layout, const Inputs &inputs,
                             int row, int column) {
  float stored = 0.0f;
  for (int k0 = 0; k0 < layout.K; k0 += layout.k) {
    float accumulator = stored;
    for (int inner = k0; inner < k0 + layout.k; ++inner) {
      const std::size_t weight_index =
          static_cast<std::size_t>(inner) * layout.N + column;
      const std::size_t parameter_index =
          static_cast<std::size_t>(inner / layout.group_size) * layout.N +
          column;
      const int centered = static_cast<int>(inputs.qweight[weight_index]) -
                           static_cast<int>(inputs.zeros[parameter_index]);
      const float scaled = centered * test_utils::bfloat16_to_float(
                                          inputs.scales[parameter_index]);
      // The core stores dequantized B in BF16 scratch before the MMUL.
      const auto b = test_utils::bfloat16_from_float(scaled);
      accumulator +=
          test_utils::bfloat16_to_float(
              inputs.A[static_cast<std::size_t>(row) * layout.K + inner]) *
          test_utils::bfloat16_to_float(b);
    }
    if constexpr (std::is_same_v<C, test_utils::bfloat16_t>)
      stored = test_utils::bfloat16_to_float(
          test_utils::bfloat16_from_float(accumulator));
    else
      stored = accumulator;
  }
  return stored;
}

inline bool close(float actual, float expected, float atol = 0.5f,
                  float rtol = 0.05f) {
  return std::abs(actual - expected) <=
         atol + rtol * std::max(std::abs(actual), std::abs(expected));
}

} // namespace awq4

#endif
