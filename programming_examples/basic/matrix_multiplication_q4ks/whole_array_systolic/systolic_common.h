//===- systolic_common.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MATRIX_MULTIPLICATION_Q4KS_SYSTOLIC_COMMON_H
#define MATRIX_MULTIPLICATION_Q4KS_SYSTOLIC_COMMON_H

#include "../common.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace q4ks_systolic {

using bf16 = q4ks::bf16;
using Inputs = q4ks::Inputs;
using Decoded = q4ks::Decoded;

constexpr int group_size = q4ks::group_size;
constexpr int n_aie_cols_required = 8;
constexpr int n_aie_rows = 4;

struct Layout {
  int M = 4096;
  int K = 4096;
  int N = 4096;
  int m_c = 32;
  int m_a = 32;
  int k = 512;
  int n = 32;
  int n_aie_cols = 8;
  std::string compute_type = "bfp16";
  std::string accumulation_mode = "cascade";
  std::string cache_mode = "q4-local";
  int cache_k = 4096;

  std::size_t groups_per_tile() const { return k / group_size; }
  std::size_t q4_weight_bytes() const {
    return static_cast<std::size_t>(k) * n / 2;
  }
  std::size_t q4_metadata_bytes() const { return groups_per_tile() * n * 4; }
  std::size_t q4_tile_bytes() const {
    return q4_weight_bytes() + q4_metadata_bytes();
  }
  std::size_t bfp_tile_bytes() const {
    return static_cast<std::size_t>(k) * n * 9 / 8;
  }
  bool uses_bfp_payload() const {
    return cache_mode == "bfp-prepared" || cache_mode == "resident";
  }
  std::size_t tile_bytes() const {
    return uses_bfp_payload() ? bfp_tile_bytes() : q4_tile_bytes();
  }
  std::size_t native_bytes() const {
    return static_cast<std::size_t>(N) * (K / q4ks::qk_k) * q4ks::block_bytes;
  }
  std::size_t prepared_bytes() const {
    return static_cast<std::size_t>(n_aie_cols) * (N / n) * tile_bytes();
  }

  void validate() const {
    if (M <= 0 || K <= 0 || N <= 0)
      throw std::invalid_argument("M, K, and N must be positive");
    if ((m_a != 32 && m_a != 64) || m_c != m_a)
      throw std::invalid_argument(
          "the systolic hardware requires m_c=m_a=32 or 64");
    if (M % (n_aie_rows * m_a))
      throw std::invalid_argument("M must be divisible by four times tile-m-a");
    if (K % 2048)
      throw std::invalid_argument("K must be divisible by 2048");
    if (n != 16 && n != 32 && n != 64)
      throw std::invalid_argument("n must be 16, 32, or 64");
    if (N % n)
      throw std::invalid_argument("N must be divisible by n");
    // The shared normalized-data generator uses the parent's eight-column
    // validation. The Python callable itself supports every N divisible by n.
    if (N % 128)
      throw std::invalid_argument(
          "the C++ normalized-data harness needs N%128=0");
    if (n_aie_cols != n_aie_cols_required)
      throw std::invalid_argument("the systolic POC requires eight columns");
    if (k != K / n_aie_cols)
      throw std::invalid_argument("tile-k must equal K/8");
    if (k > 1023)
      throw std::invalid_argument("K/8 exceeds the 10-bit DMA dimension");
    if (compute_type != "bfp16")
      throw std::invalid_argument("native BFP16 MMUL is the supported compute");
    if (accumulation_mode != "cascade")
      throw std::invalid_argument(
          "the systolic array requires cascade accumulation");
    if (cache_mode != "q4-local" && cache_mode != "q4-expand-once" &&
        cache_mode != "q4-direct" && cache_mode != "bfp-prepared" &&
        cache_mode != "resident")
      throw std::invalid_argument("invalid weight-flow/cache-mode");
    if (m_a == 64 &&
        ((cache_mode != "q4-direct" && cache_mode != "q4-expand-once") ||
         n != 32))
      throw std::invalid_argument(
          "tile-m-a=64 requires q4-direct or q4-expand-once with "
          "tile-n=32");
    if (cache_k != K)
      throw std::invalid_argument("cache-k must equal K");
    if (q4_tile_bytes() % 64 || bfp_tile_bytes() % 64)
      throw std::invalid_argument(
          "weight tiles must use complete 64-byte beats");
  }
};

inline q4ks::Layout source_layout(const Layout &layout,
                                  std::string compute_type = "bfp16") {
  q4ks::Layout source;
  source.M = layout.M;
  source.K = layout.K;
  source.N = layout.N;
  source.m_c = 16;
  source.m_a = 16;
  source.k = 256;
  source.n = 16;
  source.n_aie_cols = 8;
  source.compute_type = std::move(compute_type);
  source.accumulation_mode = source.compute_type == "bfp16" ? "fp32" : "bf16";
  source.cache_mode = "l1-weight";
  source.cache_k = layout.K;
  source.validate();
  return source;
}

inline bf16 as_bf16(float value) { return q4ks::as_bf16(value); }
inline float as_float(bf16 value) { return q4ks::as_float(value); }

inline Inputs make_inputs(const Layout &layout,
                          std::uint32_t seed = 0x53595354U) {
  layout.validate();
  return q4ks::make_inputs(source_layout(layout), seed);
}

inline Decoded decode(const Layout &layout,
                      const std::vector<std::uint8_t> &native) {
  layout.validate();
  return q4ks::decode(source_layout(layout), native);
}

inline void store_bf16(std::uint8_t *output, bf16 value) {
  std::memcpy(output, &value, sizeof(value));
}

inline void encode_bfp16ebs8(const float input[8], std::uint8_t output[9]) {
  std::uint32_t words[8];
  unsigned maximum_exponent = 0;
  for (int lane = 0; lane < 8; ++lane) {
    std::memcpy(&words[lane], &input[lane], sizeof(words[lane]));
    maximum_exponent = std::max(maximum_exponent, (words[lane] >> 23) & 0xffU);
  }
  output[0] = static_cast<std::uint8_t>(maximum_exponent);
  for (int lane = 0; lane < 8; ++lane) {
    const unsigned exponent = (words[lane] >> 23) & 0xffU;
    int mantissa = static_cast<int>(words[lane] & 0x7fffffU);
    if (exponent)
      mantissa |= 0x800000;
    if (words[lane] & 0x80000000U)
      mantissa = -mantissa;
    int quantized = mantissa >> 17;
    const unsigned delta = maximum_exponent - exponent;
    quantized = delta >= 32 ? (quantized < 0 ? -1 : 0) : quantized >> delta;
    output[lane + 1] = static_cast<std::uint8_t>(quantized & 0xff);
  }
}

inline void pack_q4_tile(std::uint8_t *tile, const Decoded &decoded, int k0,
                         int n0, const Layout &layout) {
  std::size_t cursor = 0;
  for (int mk = 0; mk < layout.k; mk += 8) {
    for (int mn = 0; mn < layout.n; mn += 8) {
      for (int row = 0; row < 8; ++row) {
        for (int lane = 0; lane < 8; lane += 2) {
          const auto index =
              static_cast<std::size_t>(k0 + mk + row) * layout.N + n0 + mn +
              lane;
          tile[cursor++] = decoded.q[index] | (decoded.q[index + 1] << 4);
        }
      }
    }
  }
  const int first_group = k0 / group_size;
  for (int group = 0; group < layout.k / group_size; ++group) {
    for (int mn = 0; mn < layout.n; mn += 8) {
      for (int lane = 0; lane < 8; ++lane) {
        const auto index =
            static_cast<std::size_t>(first_group + group) * layout.N + n0 + mn +
            lane;
        store_bf16(tile + cursor + lane * 2, as_bf16(decoded.scales[index]));
      }
      cursor += 16;
      for (int lane = 0; lane < 8; ++lane) {
        const auto index =
            static_cast<std::size_t>(first_group + group) * layout.N + n0 + mn +
            lane;
        store_bf16(tile + cursor + lane * 2, as_bf16(decoded.biases[index]));
      }
      cursor += 16;
    }
  }
  if (cursor != layout.q4_tile_bytes())
    throw std::logic_error("internal systolic Q4 tile size mismatch");
}

inline void pack_bfp_tile(std::uint8_t *tile, const Decoded &decoded, int k0,
                          int n0, const Layout &layout) {
  std::size_t cursor = 0;
  for (int mn = 0; mn < layout.n; mn += 8) {
    for (int mk = 0; mk < layout.k; mk += 8) {
      for (int column = 0; column < 8; ++column) {
        float values[8];
        for (int inner = 0; inner < 8; ++inner) {
          const int k = k0 + mk + inner;
          const int n = n0 + mn + column;
          const auto parameter =
              static_cast<std::size_t>(k / group_size) * layout.N + n;
          const float scale = as_float(as_bf16(decoded.scales[parameter]));
          const float bias = as_float(as_bf16(decoded.biases[parameter]));
          const auto q = decoded.q[static_cast<std::size_t>(k) * layout.N + n];
          values[inner] =
              as_float(as_bf16(static_cast<float>(q) * scale - bias));
        }
        encode_bfp16ebs8(values, tile + cursor);
        cursor += 9;
      }
    }
  }
  if (cursor != layout.bfp_tile_bytes())
    throw std::logic_error("internal systolic BFP tile size mismatch");
}

inline std::vector<std::uint8_t>
prepare_weights(const Layout &layout, const std::vector<std::uint8_t> &native) {
  const auto decoded = decode(layout, native);
  std::vector<std::uint8_t> result(layout.prepared_bytes());
  std::size_t tile_index = 0;
  for (int stage = 0; stage < layout.n_aie_cols; ++stage) {
    for (int panel = 0; panel < layout.N / layout.n; ++panel, ++tile_index) {
      auto *tile = result.data() + tile_index * layout.tile_bytes();
      const int k0 = stage * layout.k;
      const int n0 = panel * layout.n;
      if (layout.uses_bfp_payload())
        pack_bfp_tile(tile, decoded, k0, n0, layout);
      else
        pack_q4_tile(tile, decoded, k0, n0, layout);
    }
  }
  return result;
}

inline float reference_value(const Layout &layout, const Inputs &inputs,
                             const Decoded &decoded, int row, int col,
                             bool /*round_tile_partials*/ = false) {
  return q4ks::reference_value(source_layout(layout, layout.compute_type),
                               inputs, decoded, row, col, false);
}

inline bool close(float actual, float expected, float atol = 0.5f,
                  float rtol = 0.05f) {
  return q4ks::close(actual, expected, atol, rtol);
}

} // namespace q4ks_systolic

#endif
