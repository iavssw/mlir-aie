//===- common.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATRIX_MULTIPLICATION_Q4KS_COMMON_H
#define MATRIX_MULTIPLICATION_Q4KS_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_utils.h"

namespace q4ks {

using bf16 = test_utils::bfloat16_t;

constexpr int qk_k = 256;
constexpr int block_bytes = 144;
constexpr int group_size = 32;
constexpr int n_aie_rows = 4;

inline std::size_t ceildiv(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1) / divisor;
}

inline bf16 as_bf16(float value) {
  return test_utils::bfloat16_from_float(value);
}

inline float as_float(bf16 value) {
  return test_utils::bfloat16_to_float(value);
}

inline std::uint16_t fp16_bits(float value) {
  _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

inline float fp16_from_bits(const std::uint8_t *bytes) {
  const std::uint16_t bits = static_cast<std::uint16_t>(bytes[0]) |
                             (static_cast<std::uint16_t>(bytes[1]) << 8);
  _Float16 half;
  std::memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
}

struct Layout {
  int M = 1024;
  int K = 1024;
  int N = 2048;
  int m_c = 64;
  int m_a = 32;
  int k = 128;
  int n = 64;
  int n_aie_cols = 8;
  std::string compute_type = "bf16";
  std::string accumulation_mode = "bf16";
  std::string cache_mode = "stream";
  int cache_k = 1024;

  std::size_t groups_per_tile() const { return k / group_size; }
  std::size_t weight_bytes_per_tile() const {
    return static_cast<std::size_t>(k) * n / 2;
  }
  std::size_t metadata_bytes_per_tile() const {
    return groups_per_tile() * n * 4;
  }
  std::size_t raw_tile_bytes() const {
    return weight_bytes_per_tile() + metadata_bytes_per_tile();
  }
  std::size_t packed_rows() const { return ceildiv(raw_tile_bytes(), k); }
  std::size_t tile_bytes() const { return packed_rows() * k; }
  std::size_t native_bytes() const {
    return static_cast<std::size_t>(N) * (K / qk_k) * block_bytes;
  }
  std::size_t prepared_bytes() const {
    return static_cast<std::size_t>(K / k) * (N / n) * tile_bytes();
  }
  int c_depth() const { return m_c >= 128 ? 1 : 2; }
  int a_depth() const {
    if (accumulation_mode == "cascade")
      return 2;
    if (accumulation_mode == "cascade-hybrid")
      return 1;
    return m_a >= 64 ? 1 : 2;
  }
  std::size_t core_memory_bytes() const {
    if (accumulation_mode == "cascade-hybrid")
      return static_cast<std::size_t>(m_a) * k * 2 + tile_bytes() +
             static_cast<std::size_t>(k) * n * 9 / 8 +
             static_cast<std::size_t>(m_c) * n * 2 + 0xD00 + 4 * 1024;
    const std::size_t a_fifo =
        static_cast<std::size_t>(a_depth()) * m_a * k * 2;
    const std::size_t b_fifo = tile_bytes();
    const std::size_t c_fifo =
        static_cast<std::size_t>(c_depth()) * m_c * n * 2;
    std::size_t weight_scratch = static_cast<std::size_t>(k) * n * 2;
    std::size_t activation_scratch = 0;
    if (compute_type == "bfp16") {
      weight_scratch = static_cast<std::size_t>(k) * n * 9 / 8;
    } else if (compute_type == "int8") {
      weight_scratch = static_cast<std::size_t>(k) * n;
      activation_scratch = static_cast<std::size_t>(m_a) * k +
                           static_cast<std::size_t>(m_a) * (k / 32) * 6;
    }
    const std::size_t fp32_scratch =
        accumulation_mode == "fp32" || accumulation_mode == "cascade"
            ? static_cast<std::size_t>(m_c) * n * 4
            : 0;
    return a_fifo + b_fifo + c_fifo + weight_scratch + activation_scratch +
           fp32_scratch + 0xD00 + 4 * 1024;
  }

  std::size_t memtile_bytes() const {
    std::size_t resident = 0;
    if (cache_mode == "memtile-weight")
      resident = static_cast<std::size_t>(cache_k / k) * tile_bytes();
    if (accumulation_mode == "cascade-hybrid")
      return resident + 2ULL * m_c * k * 2 +
             static_cast<std::size_t>(m_c) * n * 2 + 32 * 1024;
    return resident + static_cast<std::size_t>(n_aie_rows) * m_c * n * 2 +
           2ULL * (static_cast<std::size_t>(m_a) * k * 2 + tile_bytes()) +
           32 * 1024;
  }

  void validate() const {
    if (M <= 0 || K <= 0 || N <= 0 || m_c <= 0 || m_a <= 0 || k <= 0 || n <= 0)
      throw std::invalid_argument(
          "matrix and tile dimensions must be positive");
    if (K % qk_k)
      throw std::invalid_argument("K must be divisible by 256");
    if (k % group_size || K % k)
      throw std::invalid_argument("k must be a 32-aligned divisor of K");
    if (m_c % m_a || m_a % 16 || n % 16)
      throw std::invalid_argument(
          "tile must satisfy m_a|m_c, m_a%16=0, n%16=0");
    if (n_aie_cols != 1 && n_aie_cols != 2 && n_aie_cols != 4 &&
        n_aie_cols != 8)
      throw std::invalid_argument("NPU2 columns must be 1, 2, 4, or 8");
    if (M % (m_c * n_aie_rows) || N % (n * n_aie_cols))
      throw std::invalid_argument(
          "matrix dimensions are incompatible with tiling");
    if ((M / (m_c * n_aie_rows)) % 2)
      throw std::invalid_argument("M/(m_c*4) must be even");
    if (compute_type != "bf16" && compute_type != "bfp16" &&
        compute_type != "int8")
      throw std::invalid_argument("compute type must be bf16, bfp16, or int8");
    if (accumulation_mode != "bf16" && accumulation_mode != "fp32" &&
        accumulation_mode != "cascade" && accumulation_mode != "cascade-hybrid")
      throw std::invalid_argument(
          "accumulation mode must be bf16, fp32, cascade, or cascade-hybrid");
    if (accumulation_mode != "bf16" && compute_type != "bfp16")
      throw std::invalid_argument(
          "FP32 and cascade accumulation require bfp16 compute");
    if (accumulation_mode == "cascade") {
      if (m_a != m_c)
        throw std::invalid_argument("cascade accumulation requires m_a == m_c");
      if (k != 64)
        throw std::invalid_argument(
            "cascade accumulation currently requires k == 64; larger "
            "specializations exceed AIE program memory");
      if (K % (n_aie_rows * k))
        throw std::invalid_argument(
            "cascade accumulation requires 4*k to divide K");
      if (cache_mode != "stream" && cache_mode != "l1-weight")
        throw std::invalid_argument(
            "cascade accumulation supports stream or l1-weight");
    }
    if (accumulation_mode == "cascade-hybrid") {
      const bool valid_tile =
          (m_c == 128 && m_a == 32 && k == 64 && n == 128) ||
          (m_c == 128 && m_a == 64 && k == 128 && n == 64) ||
          (m_c == 256 && (m_a == 32 || m_a == 64) && k == 64 && n == 64) ||
          (m_c == 256 && m_a == 32 && k == 128 && n == 64) ||
          (m_c == 512 && (m_a == 32 || m_a == 64) && k == 64 && n == 32);
      if (!valid_tile)
        throw std::invalid_argument(
            "cascade-hybrid tile must be 128x32x64x128, "
            "128x64x128x64, 256x(32|64)x64x64, "
            "256x32x128x64, or 512x(32|64)x64x32");
      if (K % (2 * k))
        throw std::invalid_argument("cascade-hybrid requires 2*k to divide K");
      if (cache_mode != "l1-weight" && cache_mode != "memtile-weight")
        throw std::invalid_argument(
            "cascade-hybrid requires l1-weight or memtile-weight");
    }
    if (cache_mode != "stream" && cache_mode != "l1-weight" &&
        cache_mode != "memtile-weight")
      throw std::invalid_argument(
          "C++ execution supports stream, l1-weight, or memtile-weight");
    if (cache_k <= 0 || cache_k % qk_k || K % cache_k)
      throw std::invalid_argument("cache_k must be a 256-aligned divisor of K");
    if (cache_mode == "memtile-weight" && cache_k != K)
      throw std::invalid_argument("memtile-weight requires cache_k == K");
    if (packed_rows() > 1023 ||
        (accumulation_mode == "cascade" && n_aie_rows * packed_rows() > 1023) ||
        K / k > 1023 || N / (n * n_aie_cols) > 1023 || k > 1023 || m_a > 1023 ||
        m_c > 1023 || n > 1023)
      throw std::invalid_argument("DMA size exceeds the 10-bit limit");
    if (core_memory_bytes() > 64 * 1024)
      throw std::invalid_argument("configuration exceeds 64-KiB core memory");
    if (cache_mode == "memtile-weight" && memtile_bytes() > 512 * 1024)
      throw std::invalid_argument(
          "configuration exceeds 512-KiB MemTile memory");
  }
};

struct Inputs {
  std::vector<bf16> A;
  std::vector<std::uint8_t> native_B;
};

struct Decoded {
  std::vector<std::uint8_t> q;
  std::vector<float> scales;
  std::vector<float> biases;
};

inline void pack_scale_min(const std::uint8_t scales[8],
                           const std::uint8_t mins[8],
                           std::uint8_t packed[12]) {
  std::fill(packed, packed + 12, 0);
  for (int j = 0; j < 4; ++j) {
    packed[j] = scales[j];
    packed[j + 4] = mins[j];
  }
  for (int j = 4; j < 8; ++j) {
    packed[j + 4] = (scales[j] & 0x0f) | ((mins[j] & 0x0f) << 4);
    packed[j - 4] |= (scales[j] >> 4) << 6;
    packed[j] |= (mins[j] >> 4) << 6;
  }
}

inline void unpack_scale_min(const std::uint8_t packed[12],
                             std::uint8_t scales[8], std::uint8_t mins[8]) {
  for (int j = 0; j < 8; ++j) {
    if (j < 4) {
      scales[j] = packed[j] & 0x3f;
      mins[j] = packed[j + 4] & 0x3f;
    } else {
      scales[j] = (packed[j + 4] & 0x0f) | ((packed[j - 4] >> 6) << 4);
      mins[j] = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
    }
  }
}

inline Inputs make_inputs(const Layout &layout,
                          std::uint32_t seed = 0x4b535f34U) {
  layout.validate();
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  Inputs result;
  result.A.resize(static_cast<std::size_t>(layout.M) * layout.K);
  std::vector<float> values(layout.K);

  // Model the output of LayerNorm/RMSNorm: every token row is centered and
  // has unit RMS before it enters the projection matmul.
  for (int row = 0; row < layout.M; ++row) {
    double mean = 0.0;
    for (float &value : values) {
      value = normal(rng);
      mean += value;
    }
    mean /= layout.K;
    double square_sum = 0.0;
    for (float value : values) {
      const double centered = value - mean;
      square_sum += centered * centered;
    }
    const float inverse_rms =
        static_cast<float>(1.0 / std::sqrt(square_sum / layout.K + 1.0e-12));
    for (int inner = 0; inner < layout.K; ++inner)
      result.A[static_cast<std::size_t>(row) * layout.K + inner] =
          as_bf16(static_cast<float>(values[inner] - mean) * inverse_rms);
  }

  result.native_B.assign(layout.native_bytes(), 0);
  const int blocks_per_row = layout.K / qk_k;

  // A normalized projection row has L2 norm one (RMS 1/sqrt(K)).  Generate
  // that float-domain row first, then quantize it into the actual llama.cpp
  // block_q4_K hierarchy instead of choosing Q4 codes and metadata
  // independently.
  const float weight_target_rms =
      1.0f / std::sqrt(static_cast<float>(layout.K));
  for (int col = 0; col < layout.N; ++col) {
    double mean = 0.0;
    for (float &value : values) {
      value = normal(rng);
      mean += value;
    }
    mean /= layout.K;
    double square_sum = 0.0;
    for (float value : values) {
      const double centered = value - mean;
      square_sum += centered * centered;
    }
    const float row_rms =
        static_cast<float>(std::sqrt(square_sum / layout.K + 1.0e-20));
    const float row_scale = weight_target_rms / row_rms;
    for (float &value : values)
      value = static_cast<float>(value - mean) * row_scale;

    for (int bi = 0; bi < blocks_per_row; ++bi) {
      auto *block =
          result.native_B.data() +
          (static_cast<std::size_t>(col) * blocks_per_row + bi) * block_bytes;
      float desired_scales[8];
      float desired_biases[8];
      float maximum_scale = 0.0f;
      float maximum_bias = 0.0f;
      for (int group = 0; group < 8; ++group) {
        const auto first = values.begin() + bi * qk_k + group * group_size;
        const auto last = first + group_size;
        const auto [minimum_it, maximum_it] = std::minmax_element(first, last);
        const float minimum = std::min(*minimum_it, 0.0f);
        const float maximum = std::max(*maximum_it, 0.0f);
        desired_scales[group] = std::max((maximum - minimum) / 15.0f, 1.0e-12f);
        desired_biases[group] = -minimum;
        maximum_scale = std::max(maximum_scale, desired_scales[group]);
        maximum_bias = std::max(maximum_bias, desired_biases[group]);
      }

      const auto d_bits = fp16_bits(std::max(maximum_scale, 1.0e-7f) / 63.0f);
      const auto dmin_bits =
          fp16_bits(maximum_bias == 0.0f ? 0.0f : maximum_bias / 63.0f);
      block[0] = d_bits & 0xff;
      block[1] = d_bits >> 8;
      block[2] = dmin_bits & 0xff;
      block[3] = dmin_bits >> 8;
      const float d = fp16_from_bits(block);
      const float dmin = fp16_from_bits(block + 2);
      std::uint8_t scales[8], mins[8], packed[12];
      for (int group = 0; group < 8; ++group) {
        scales[group] = static_cast<std::uint8_t>(std::clamp(
            static_cast<int>(std::lround(desired_scales[group] / d)), 1, 63));
        mins[group] = dmin == 0.0f ? 0
                                   : static_cast<std::uint8_t>(std::clamp(
                                         static_cast<int>(std::lround(
                                             desired_biases[group] / dmin)),
                                         0, 63));
      }
      pack_scale_min(scales, mins, packed);
      std::memcpy(block + 4, packed, 12);
      std::uint8_t quantized[qk_k];
      for (int group = 0; group < 8; ++group) {
        const float effective_scale = d * scales[group];
        const float effective_bias = dmin * mins[group];
        for (int inner = 0; inner < group_size; ++inner) {
          const int index = group * group_size + inner;
          const float value = values[bi * qk_k + index];
          quantized[index] = static_cast<std::uint8_t>(
              std::clamp(static_cast<int>(std::lround((value + effective_bias) /
                                                      effective_scale)),
                         0, 15));
        }
      }
      for (int region = 0; region < 4; ++region)
        for (int x = 0; x < 32; ++x)
          block[16 + region * 32 + x] =
              static_cast<std::uint8_t>(quantized[region * 64 + x] |
                                        (quantized[region * 64 + 32 + x] << 4));
    }
  }
  return result;
}

inline Decoded decode(const Layout &layout,
                      const std::vector<std::uint8_t> &native) {
  layout.validate();
  if (native.size() != layout.native_bytes())
    throw std::invalid_argument("native Q4_K payload has the wrong byte size");
  Decoded result;
  result.q.resize(static_cast<std::size_t>(layout.K) * layout.N);
  result.scales.resize(static_cast<std::size_t>(layout.K / group_size) *
                       layout.N);
  result.biases.resize(result.scales.size());
  const int blocks_per_row = layout.K / qk_k;
  for (int col = 0; col < layout.N; ++col) {
    for (int bi = 0; bi < blocks_per_row; ++bi) {
      const auto *block =
          native.data() +
          (static_cast<std::size_t>(col) * blocks_per_row + bi) * block_bytes;
      const float d = fp16_from_bits(block);
      const float dmin = fp16_from_bits(block + 2);
      if (!std::isfinite(d) || !std::isfinite(dmin))
        throw std::invalid_argument("native Q4_K factors must be finite");
      std::uint8_t scales[8], mins[8];
      unpack_scale_min(block + 4, scales, mins);
      for (int group = 0; group < 8; ++group) {
        const auto index =
            static_cast<std::size_t>(bi * 8 + group) * layout.N + col;
        result.scales[index] = d * scales[group];
        result.biases[index] = dmin * mins[group];
      }
      for (int region = 0; region < 4; ++region) {
        for (int x = 0; x < 32; ++x) {
          const auto byte = block[16 + region * 32 + x];
          const int k0 = bi * qk_k + region * 64;
          result.q[static_cast<std::size_t>(k0 + x) * layout.N + col] =
              byte & 0x0f;
          result.q[static_cast<std::size_t>(k0 + 32 + x) * layout.N + col] =
              byte >> 4;
        }
      }
    }
  }
  return result;
}

inline std::vector<std::uint8_t>
prepare_weights(const Layout &layout, const std::vector<std::uint8_t> &native) {
  const auto decoded = decode(layout, native);
  std::vector<std::uint8_t> prepared(layout.prepared_bytes(), 0);
  const int n_k_tiles = layout.K / layout.k;
  const int n_n_tiles = layout.N / layout.n;
  const int n_rounds = n_n_tiles / layout.n_aie_cols;
  const int cascade_rows =
      layout.accumulation_mode == "cascade-hybrid" ? 2 : n_aie_rows;
  const int chunks_per_row = n_k_tiles / cascade_rows;
  std::size_t tile_index = 0;
  for (int col = 0; col < layout.n_aie_cols; ++col) {
    for (int n_round = 0; n_round < n_rounds; ++n_round) {
      const int nt = col + n_round * layout.n_aie_cols;
      for (int stored_kt = 0; stored_kt < n_k_tiles;
           ++stored_kt, ++tile_index) {
        const int kt = layout.accumulation_mode == "cascade-hybrid"
                           ? (stored_kt % chunks_per_row) * cascade_rows +
                                 stored_kt / chunks_per_row
                           : stored_kt;
        auto *tile = prepared.data() + tile_index * layout.tile_bytes();
        std::size_t cursor = 0;
        const int k0 = kt * layout.k;
        const int n0 = nt * layout.n;
        for (int mk = 0; mk < layout.k; mk += 8) {
          for (int mn = 0; mn < layout.n; mn += 8) {
            for (int row = 0; row < 8; ++row) {
              for (int lane = 0; lane < 8; lane += 2) {
                const auto logical =
                    static_cast<std::size_t>(k0 + mk + row) * layout.N + n0 +
                    mn + lane;
                tile[cursor++] =
                    decoded.q[logical] | (decoded.q[logical + 1] << 4);
              }
            }
          }
        }
        const int first_group = k0 / group_size;
        for (int group = 0; group < layout.k / group_size; ++group) {
          for (int mn = 0; mn < layout.n; mn += 8) {
            for (int lane = 0; lane < 8; ++lane) {
              const auto logical =
                  static_cast<std::size_t>(first_group + group) * layout.N +
                  n0 + mn + lane;
              const bf16 value = as_bf16(decoded.scales[logical]);
              std::memcpy(tile + cursor + lane * sizeof(value), &value,
                          sizeof(value));
            }
            cursor += 16;
            for (int lane = 0; lane < 8; ++lane) {
              const auto logical =
                  static_cast<std::size_t>(first_group + group) * layout.N +
                  n0 + mn + lane;
              const bf16 value = as_bf16(decoded.biases[logical]);
              std::memcpy(tile + cursor + lane * sizeof(value), &value,
                          sizeof(value));
            }
            cursor += 16;
          }
        }
        if (cursor != layout.raw_tile_bytes())
          throw std::logic_error("internal Q4_K tile packing size mismatch");
      }
    }
  }
  return prepared;
}

inline void round_bfp16ebs8(const float input[8], float output[8]) {
  std::uint32_t bits[8];
  unsigned maximum_exponent = 0;
  for (int lane = 0; lane < 8; ++lane) {
    std::memcpy(&bits[lane], &input[lane], sizeof(bits[lane]));
    maximum_exponent = std::max(maximum_exponent, (bits[lane] >> 23) & 0xffU);
  }
  const float multiplier =
      std::ldexp(1.0f / 64.0f, static_cast<int>(maximum_exponent) - 127);
  for (int lane = 0; lane < 8; ++lane) {
    const unsigned exponent = (bits[lane] >> 23) & 0xffU;
    int mantissa = static_cast<int>(bits[lane] & 0x7fffffU);
    if (exponent)
      mantissa |= 0x800000;
    if (bits[lane] & 0x80000000U)
      mantissa = -mantissa;
    int quantized = mantissa >> 17;
    const unsigned delta = maximum_exponent - exponent;
    quantized = delta >= 32 ? (quantized < 0 ? -1 : 0) : quantized >> delta;
    const auto byte = static_cast<std::uint8_t>(quantized & 0xff);
    std::int8_t signed_byte;
    std::memcpy(&signed_byte, &byte, sizeof(byte));
    output[lane] = static_cast<float>(signed_byte) * multiplier;
  }
}

inline void reference_accumulate_tile(const Layout &layout,
                                      const Inputs &inputs,
                                      const Decoded &decoded, int row, int col,
                                      int kt, float &accumulator) {
  for (int group0 = kt; group0 < kt + layout.k; group0 += group_size) {
    const auto parameter =
        static_cast<std::size_t>(group0 / group_size) * layout.N + col;
    const float ws = as_float(as_bf16(decoded.scales[parameter]));
    const float wb = as_float(as_bf16(decoded.biases[parameter]));
    if (layout.compute_type == "int8") {
      float maximum = 0.0f;
      for (int x = 0; x < group_size; ++x)
        maximum = std::max(
            maximum, std::abs(as_float(
                         inputs.A[static_cast<std::size_t>(row) * layout.K +
                                  group0 + x])));
      const float quant_scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
      const float stored_scale = as_float(as_bf16(quant_scale));
      int dot = 0;
      int sum = 0;
      for (int x = 0; x < group_size; ++x) {
        const float a = as_float(
            inputs.A[static_cast<std::size_t>(row) * layout.K + group0 + x]);
        int q = static_cast<int>(a / quant_scale + (a >= 0.0f ? 0.5f : -0.5f));
        q = std::clamp(q, -127, 127);
        sum += q;
        dot += q *
               decoded.q[static_cast<std::size_t>(group0 + x) * layout.N + col];
      }
      accumulator += stored_scale * (ws * dot - wb * sum);
    } else if (layout.compute_type == "bfp16") {
      for (int x = 0; x < group_size; x += 8) {
        float activation[8], weights[8], rounded_a[8], rounded_b[8];
        for (int lane = 0; lane < 8; ++lane) {
          const int inner = group0 + x + lane;
          activation[lane] = as_float(
              inputs.A[static_cast<std::size_t>(row) * layout.K + inner]);
          const auto q =
              decoded.q[static_cast<std::size_t>(inner) * layout.N + col];
          weights[lane] = as_float(as_bf16(static_cast<float>(q) * ws - wb));
        }
        round_bfp16ebs8(activation, rounded_a);
        round_bfp16ebs8(weights, rounded_b);
        for (int lane = 0; lane < 8; ++lane)
          accumulator += rounded_a[lane] * rounded_b[lane];
      }
    } else {
      for (int x = 0; x < group_size; ++x) {
        const auto a =
            inputs.A[static_cast<std::size_t>(row) * layout.K + group0 + x];
        const auto q =
            decoded.q[static_cast<std::size_t>(group0 + x) * layout.N + col];
        const float weight = as_float(as_bf16(static_cast<float>(q) * ws - wb));
        accumulator += as_float(a) * weight;
      }
    }
  }
}

inline float reference_value(const Layout &layout, const Inputs &inputs,
                             const Decoded &decoded, int row, int col,
                             bool round_tile_partials = true) {
  if (round_tile_partials && layout.accumulation_mode == "cascade-hybrid") {
    constexpr int cascade_rows = 2;
    const int chunks_per_row = layout.K / (cascade_rows * layout.k);
    float row_partials[cascade_rows] = {};
    for (int cascade_row = 0; cascade_row < cascade_rows; ++cascade_row) {
      for (int chunk = 0; chunk < chunks_per_row; ++chunk) {
        const int kt = (chunk * cascade_rows + cascade_row) * layout.k;
        reference_accumulate_tile(layout, inputs, decoded, row, col, kt,
                                  row_partials[cascade_row]);
        if (chunk + 1 != chunks_per_row)
          row_partials[cascade_row] =
              as_float(as_bf16(row_partials[cascade_row]));
      }
    }
    float accumulator = row_partials[cascade_rows - 1];
    for (int cascade_row = cascade_rows - 2; cascade_row >= 0; --cascade_row)
      accumulator += row_partials[cascade_row];
    return as_float(as_bf16(accumulator));
  }

  float accumulator = 0.0f;
  for (int kt = 0; kt < layout.K; kt += layout.k) {
    reference_accumulate_tile(layout, inputs, decoded, row, col, kt,
                              accumulator);
    if (round_tile_partials)
      accumulator = as_float(as_bf16(accumulator));
  }
  return as_float(as_bf16(accumulator));
}

inline bool close(float actual, float expected, float atol = 0.5f,
                  float rtol = 0.05f) {
  return std::abs(actual - expected) <=
         atol + rtol * std::max(std::abs(actual), std::abs(expected));
}

} // namespace q4ks

#endif
