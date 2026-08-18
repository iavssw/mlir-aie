//===- q4ks.cc --------------------------------------------------*- C++ -*-===//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define NOCPP

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M_C
#define DIM_M_C 64
#endif
#ifndef DIM_M_A
#define DIM_M_A 32
#endif
#ifndef DIM_K
#define DIM_K 128
#endif
#ifndef DIM_N
#define DIM_N 64
#endif
#ifndef PACKED_TILE_BYTES
#define PACKED_TILE_BYTES (DIM_K * DIM_N * 5 / 8)
#endif

namespace {

constexpr unsigned kM = 8;
constexpr unsigned kK = 8;
constexpr unsigned kN = 8;
constexpr unsigned kGroup = 32;
constexpr unsigned kKBlocks = DIM_K / kK;
constexpr unsigned kNBlocks = DIM_N / kN;
constexpr unsigned kGroups = DIM_K / kGroup;
constexpr unsigned kWeightBytes = DIM_K * DIM_N / 2;
constexpr unsigned kMetadataBytes = kGroups * DIM_N * 4;
constexpr unsigned kRawBytes = kWeightBytes + kMetadataBytes;

static_assert(DIM_M_C % DIM_M_A == 0, "A subtile must divide C tile M");
static_assert(DIM_M_A % 16 == 0, "A subtile must be divisible by 16");
static_assert(DIM_K % 32 == 0, "Q4_K groups contain 32 K values");
static_assert(DIM_N % 16 == 0, "N must contain pairs of 8-column tiles");
static_assert(kRawBytes <= PACKED_TILE_BYTES, "packed Q4_K tile is too small");

static inline const bfloat16 *scale_vector(const uint8 *__restrict packed,
                                           unsigned group, unsigned nb) {
  const unsigned vector = group * kNBlocks + nb;
  return reinterpret_cast<const bfloat16 *>(packed + kWeightBytes +
                                            vector * 32);
}

static inline const bfloat16 *bias_vector(const uint8 *__restrict packed,
                                          unsigned group, unsigned nb) {
  return scale_vector(packed, group, nb) + 8;
}

static inline aie::vector<int8, 64>
unpack_q_microtile(const uint8 *__restrict packed, unsigned block) {
  const auto bytes = aie::load_v<32>(packed + block * 32);
  return bytes.template cast_to<uint4>()
      .template unpack()
      .template cast_to<int8>();
}

static inline void zero_bf16(bfloat16 *__restrict output) {
  const auto zero = aie::zeros<bfloat16, 32>();
  for (unsigned i = 0; i < DIM_M_C * DIM_N; i += 32)
    aie::store_v(output + i, zero);
}

#ifdef COMPUTE_BF16
alignas(aie::vector_decl_align) static bfloat16 b_bf16[DIM_K * DIM_N];

static inline void prepare_bf16(const uint8 *__restrict packed) {
  for (unsigned group = 0; group < kGroups; ++group)
    chess_prepare_for_pipelining {
      for (unsigned nb = 0; nb < kNBlocks; ++nb) {
        const auto scale8 = aie::load_v<8>(scale_vector(packed, group, nb));
        const auto bias8 = aie::load_v<8>(bias_vector(packed, group, nb));
        const auto scale64 = scale8.template grow_replicate<64>();
        const auto bias64 = bias8.template grow_replicate<64>();
        for (unsigned inner = 0; inner < 4; ++inner) {
          const unsigned kb = group * 4 + inner;
          const unsigned block = kb * kNBlocks + nb;
          const auto q8 = unpack_q_microtile(packed, block);
          const auto qbf16 = aie::to_float<bfloat16>(q8);
          const auto affine = aie::sub(aie::mul(qbf16, scale64), bias64);
          aie::store_v(b_bf16 + block * 64,
                       affine.template to_vector<bfloat16>());
        }
      }
    }
}

static inline void matmul_bf16(const bfloat16 *__restrict input_a,
                               bfloat16 *__restrict output_c,
                               unsigned subtile) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  output_c += subtile * DIM_M_A * DIM_N;
  constexpr unsigned row_blocks = DIM_M_A / 8;
  for (unsigned rb = 0; rb < row_blocks; rb += 2)
    chess_prepare_for_pipelining {
      bfloat16 *__restrict c0 = output_c + rb * kNBlocks * 64;
      bfloat16 *__restrict c1 = output_c + (rb + 1) * kNBlocks * 64;
      for (unsigned nb = 0; nb < kNBlocks; nb += 2)
        chess_flatten_loop {
          const bfloat16 *__restrict a0 = input_a + rb * kKBlocks * 64;
          const bfloat16 *__restrict a1 = input_a + (rb + 1) * kKBlocks * 64;
          const bfloat16 *__restrict b0 = b_bf16 + nb * 64;
          const bfloat16 *__restrict b1 = b_bf16 + (nb + 1) * 64;
          MMUL c00(aie::load_v<64>(c0));
          MMUL c01(aie::load_v<64>(c0 + 64));
          MMUL c10(aie::load_v<64>(c1));
          MMUL c11(aie::load_v<64>(c1 + 64));
          for (unsigned kb = 0; kb < kKBlocks; ++kb)
            chess_flatten_loop {
              const auto av0 = aie::load_v<64>(a0);
              const auto av1 = aie::load_v<64>(a1);
              const auto bv0 = aie::load_v<64>(b0);
              const auto bv1 = aie::load_v<64>(b1);
              c00.mac(av0, bv0);
              c01.mac(av0, bv1);
              c10.mac(av1, bv0);
              c11.mac(av1, bv1);
              a0 += 64;
              a1 += 64;
              b0 += kNBlocks * 64;
              b1 += kNBlocks * 64;
            }
          aie::store_v(c0, c00.template to_vector<bfloat16>());
          aie::store_v(c0 + 64, c01.template to_vector<bfloat16>());
          aie::store_v(c1, c10.template to_vector<bfloat16>());
          aie::store_v(c1 + 64, c11.template to_vector<bfloat16>());
          c0 += 128;
          c1 += 128;
        }
    }
}
#endif

#ifdef COMPUTE_BFP16
alignas(aie::vector_decl_align) static bfp16ebs8 b_bfp[DIM_K * DIM_N / 8];

static inline void prepare_bfp16(const uint8 *__restrict packed) {
  for (unsigned group = 0; group < kGroups; ++group)
    chess_prepare_for_pipelining {
      for (unsigned nb = 0; nb < kNBlocks; ++nb) {
        const auto scale8 = aie::load_v<8>(scale_vector(packed, group, nb));
        const auto bias8 = aie::load_v<8>(bias_vector(packed, group, nb));
        const auto scale64 = scale8.template grow_replicate<64>();
        const auto bias64 = bias8.template grow_replicate<64>();
        for (unsigned inner = 0; inner < 4; ++inner) {
          const unsigned kb = group * 4 + inner;
          const unsigned block = kb * kNBlocks + nb;
          const auto qbf16 =
              aie::to_float<bfloat16>(unpack_q_microtile(packed, block));
          const auto affine = aie::sub(aie::mul(qbf16, scale64), bias64);
          const auto column_major =
              aie::transpose(affine.template to_vector<bfloat16>(), 8, 8);
          aie::accum<accfloat, 64> transposed;
          transposed = column_major;
          aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(b_bfp);
          out.seek(block);
          out.push(transposed.template to_vector<bfp16ebs8>());
        }
      }
    }
}

static inline void matmul_bfp16(const bfloat16 *__restrict input_a,
                                bfloat16 *__restrict output_c,
                                unsigned subtile) {
  output_c += subtile * DIM_M_A * DIM_N;
  constexpr unsigned row_blocks = DIM_M_A / 8;
  for (unsigned rb = 0; rb < row_blocks; rb += 2)
    chess_prepare_for_pipelining {
      bfloat16 *__restrict c0 = output_c + rb * kNBlocks * 64;
      bfloat16 *__restrict c1 = output_c + (rb + 1) * kNBlocks * 64;
      for (unsigned nb = 0; nb < kNBlocks; nb += 2)
        chess_flatten_loop {
          const bfloat16 *__restrict a0 = input_a + rb * kKBlocks * 64;
          const bfloat16 *__restrict a1 = input_a + (rb + 1) * kKBlocks * 64;
          aie::accum<accfloat, 64> c00(aie::load_v<64>(c0));
          aie::accum<accfloat, 64> c01(aie::load_v<64>(c0 + 64));
          aie::accum<accfloat, 64> c10(aie::load_v<64>(c1));
          aie::accum<accfloat, 64> c11(aie::load_v<64>(c1 + 64));
          for (unsigned kb = 0; kb < kKBlocks; ++kb)
            chess_flatten_loop {
              const auto av0 = aie::load_v<64>(a0);
              const auto av1 = aie::load_v<64>(a1);
              aie::accum<accfloat, 64> aa0;
              aie::accum<accfloat, 64> aa1;
              aa0 = av0;
              aa1 = av1;
              const auto ab0 = aa0.template to_vector<bfp16ebs8>();
              const auto ab1 = aa1.template to_vector<bfp16ebs8>();
              aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs0(b_bfp);
              aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs1(b_bfp);
              bs0.seek(kb * kNBlocks + nb);
              bs1.seek(kb * kNBlocks + nb + 1);
              const auto bv0 = bs0.pop();
              const auto bv1 = bs1.pop();
              c00 = mac_8x8_8x8T(ab0, bv0, c00);
              c01 = mac_8x8_8x8T(ab0, bv1, c01);
              c10 = mac_8x8_8x8T(ab1, bv0, c10);
              c11 = mac_8x8_8x8T(ab1, bv1, c11);
              a0 += 64;
              a1 += 64;
            }
          aie::store_v(c0, c00.template to_vector<bfloat16>());
          aie::store_v(c0 + 64, c01.template to_vector<bfloat16>());
          aie::store_v(c1, c10.template to_vector<bfloat16>());
          aie::store_v(c1 + 64, c11.template to_vector<bfloat16>());
          c0 += 128;
          c1 += 128;
        }
    }
}
#endif

#ifdef COMPUTE_INT8
alignas(aie::vector_decl_align) static int8 b_i8[DIM_K * DIM_N];
alignas(aie::vector_decl_align) static int8 a_i8[DIM_M_A * DIM_K];
alignas(aie::vector_decl_align) static bfloat16 a_scale[DIM_M_A * kGroups];
alignas(aie::vector_decl_align) static int32 a_sum[DIM_M_A * kGroups];

static inline void prepare_i8_weights(const uint8 *__restrict packed) {
  for (unsigned block = 0; block < kKBlocks * kNBlocks; ++block)
    aie::store_v(b_i8 + block * 64, unpack_q_microtile(packed, block));
}

static inline unsigned a_index(unsigned row, unsigned col) {
  return ((row / 8) * kKBlocks + col / 8) * 64 + (row % 8) * 8 + col % 8;
}

static inline void quantize_a(const bfloat16 *__restrict input_a) {
  for (unsigned row = 0; row < DIM_M_A; ++row) {
    for (unsigned group = 0; group < kGroups; ++group) {
      float maximum = 0.0f;
      for (unsigned x = 0; x < kGroup; ++x) {
        float value =
            static_cast<float>(input_a[a_index(row, group * kGroup + x)]);
        float magnitude = value < 0.0f ? -value : value;
        maximum = magnitude > maximum ? magnitude : maximum;
      }
      const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
      a_scale[row * kGroups + group] = static_cast<bfloat16>(scale);
      int32 sum = 0;
      for (unsigned x = 0; x < kGroup; ++x) {
        const unsigned index = a_index(row, group * kGroup + x);
        const float scaled = static_cast<float>(input_a[index]) / scale;
        int value = scaled >= 0.0f ? static_cast<int>(scaled + 0.5f)
                                   : static_cast<int>(scaled - 0.5f);
        value = value > 127 ? 127 : (value < -127 ? -127 : value);
        a_i8[index] = static_cast<int8>(value);
        sum += value;
      }
      a_sum[row * kGroups + group] = sum;
    }
  }
}

static inline void matmul_i8(const bfloat16 *__restrict input_a,
                             const uint8 *__restrict packed,
                             bfloat16 *__restrict output_c, unsigned subtile) {
  using MMUL = aie::mmul<4, 8, 8, int8, int8, acc32>;
  quantize_a(input_a);
  output_c += subtile * DIM_M_A * DIM_N;
  constexpr unsigned row_blocks = DIM_M_A / 8;
  for (unsigned rb = 0; rb < row_blocks; ++rb) {
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      float accum[64];
      bfloat16 *__restrict c = output_c + (rb * kNBlocks + nb) * 64;
      for (unsigned lane = 0; lane < 64; ++lane)
        accum[lane] = static_cast<float>(c[lane]);
      for (unsigned group = 0; group < kGroups; ++group) {
        MMUL top = aie::zeros<acc32, 32>();
        MMUL bottom = aie::zeros<acc32, 32>();
        for (unsigned inner = 0; inner < 4; ++inner) {
          const unsigned kb = group * 4 + inner;
          const int8 *__restrict ap = a_i8 + (rb * kKBlocks + kb) * 64;
          const int8 *__restrict bp = b_i8 + (kb * kNBlocks + nb) * 64;
          const auto bv = aie::load_v<64>(bp);
          top.mac(aie::load_v<32>(ap), bv);
          bottom.mac(aie::load_v<32>(ap + 32), bv);
        }
        const auto dt = top.template to_vector<int32>();
        const auto db = bottom.template to_vector<int32>();
        const bfloat16 *__restrict ws = scale_vector(packed, group, nb);
        const bfloat16 *__restrict wb = bias_vector(packed, group, nb);
        for (unsigned r4 = 0; r4 < 4; ++r4) {
          const unsigned logical_row = rb * 8 + r4;
          const float as =
              static_cast<float>(a_scale[logical_row * kGroups + group]);
          const float sum =
              static_cast<float>(a_sum[logical_row * kGroups + group]);
          for (unsigned col = 0; col < 8; ++col) {
            const unsigned lane = r4 * 8 + col;
            accum[lane] += as * (static_cast<float>(ws[col]) *
                                     static_cast<float>(dt[lane]) -
                                 static_cast<float>(wb[col]) * sum);
          }
        }
        for (unsigned r4 = 0; r4 < 4; ++r4) {
          const unsigned logical_row = rb * 8 + 4 + r4;
          const float as =
              static_cast<float>(a_scale[logical_row * kGroups + group]);
          const float sum =
              static_cast<float>(a_sum[logical_row * kGroups + group]);
          for (unsigned col = 0; col < 8; ++col) {
            const unsigned lane = 32 + r4 * 8 + col;
            const unsigned dot_lane = r4 * 8 + col;
            accum[lane] += as * (static_cast<float>(ws[col]) *
                                     static_cast<float>(db[dot_lane]) -
                                 static_cast<float>(wb[col]) * sum);
          }
        }
      }
      for (unsigned lane = 0; lane < 64; ++lane)
        c[lane] = static_cast<bfloat16>(accum[lane]);
    }
  }
}
#endif

} // namespace

extern "C" {

void q4ks_zero_bf16(bfloat16 *output) { zero_bf16(output); }

#ifdef COMPUTE_BF16
void q4ks_matmul_bf16(bfloat16 *a, uint8 *b, bfloat16 *c, int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    prepare_bf16(b);
  matmul_bf16(a, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}
#endif

#ifdef COMPUTE_BFP16
void q4ks_matmul_bfp16(bfloat16 *a, uint8 *b, bfloat16 *c, int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    prepare_bfp16(b);
  matmul_bfp16(a, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}
#endif

#ifdef COMPUTE_INT8
void q4ks_matmul_int8(bfloat16 *a, uint8 *b, bfloat16 *c, int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    prepare_i8_weights(b);
  matmul_i8(a, b, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}
#endif
}
