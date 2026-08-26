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
#ifndef DIM_SHARD_K
#define DIM_SHARD_K DIM_K
#endif
#ifndef DIM_CHUNK_K
#define DIM_CHUNK_K 256
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
#if !defined(ACCUM_CASCADE_RESIDENT) && !defined(ACCUM_CASCADE_REGISTER) &&    \
    !defined(ACCUM_CASCADE_CHUNKED)
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
#endif

#if !defined(ACCUM_FP32) && !defined(ACCUM_CASCADE) &&                         \
    !defined(ACCUM_CASCADE_RESIDENT) && !defined(ACCUM_CASCADE_REGISTER) &&    \
    !defined(ACCUM_CASCADE_CHUNKED) && !defined(ACCUM_CASCADE_SHARED)
static inline void matmul_bfp16_from(const bfloat16 *__restrict input_a,
                                     const bfp16ebs8 *__restrict weights,
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
              aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs0(weights);
              aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs1(weights);
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

static inline void matmul_bfp16(const bfloat16 *__restrict input_a,
                                bfloat16 *__restrict output_c,
                                unsigned subtile) {
  matmul_bfp16_from(input_a, b_bfp, output_c, subtile);
}
#endif

#if defined(ACCUM_FP32) || defined(ACCUM_CASCADE) ||                           \
    defined(ACCUM_CASCADE_RESIDENT) || defined(ACCUM_CASCADE_REGISTER) ||      \
    defined(ACCUM_CASCADE_CHUNKED) || defined(ACCUM_CASCADE_SHARED) ||         \
    defined(ACCUM_CASCADE_HYBRID)
static inline aie::accum<accfloat, 64>
load_fp32_accumulator(const float *__restrict input) {
  aie::accum<accfloat, 64> result;
  result.insert(0, aie::accum<accfloat, 32>(aie::load_v<32>(input)));
  result.insert(1, aie::accum<accfloat, 32>(aie::load_v<32>(input + 32)));
  return result;
}

static inline void
store_fp32_accumulator(float *__restrict output,
                       const aie::accum<accfloat, 64> &value) {
  aie::store_v(output, value.template extract<32>(0).to_vector<float>());
  aie::store_v(output + 32, value.template extract<32>(1).to_vector<float>());
}

#if defined(ACCUM_CASCADE) || defined(ACCUM_CASCADE_RESIDENT) ||               \
    defined(ACCUM_CASCADE_REGISTER) || defined(ACCUM_CASCADE_CHUNKED) ||       \
    defined(ACCUM_CASCADE_SHARED) || defined(ACCUM_CASCADE_HYBRID)
static inline aie::accum<accfloat, 64> load_cascade_accumulator() {
  return aie::accum<accfloat, 64>(get_scd_v64accfloat(1));
}

static inline void
put_cascade_accumulator(const aie::accum<accfloat, 64> &value) {
  const auto native = value.to_native();
  put_mcd(extract_v16accfloat(native, 0));
  put_mcd(extract_v16accfloat(native, 1));
  put_mcd(extract_v16accfloat(native, 2));
  put_mcd(extract_v16accfloat(native, 3));
}
#endif

#if !defined(ACCUM_CASCADE_RESIDENT) && !defined(ACCUM_CASCADE_REGISTER) &&    \
    !defined(ACCUM_CASCADE_CHUNKED)
static inline void
mac_bfp16_2x2(const bfloat16 *__restrict input_a, unsigned rb, unsigned nb,
              aie::accum<accfloat, 64> &c00, aie::accum<accfloat, 64> &c01,
              aie::accum<accfloat, 64> &c10, aie::accum<accfloat, 64> &c11) {
  const bfloat16 *__restrict a0 = input_a + rb * kKBlocks * 64;
  const bfloat16 *__restrict a1 = input_a + (rb + 1) * kKBlocks * 64;
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
}
#endif

static inline void zero_fp32(float *__restrict output) {
  const auto zero = aie::zeros<float, 32>();
  for (unsigned i = 0; i < DIM_M_C * DIM_N; i += 32)
    aie::store_v(output + i, zero);
}

static inline void store_bf16(const float *__restrict input,
                              bfloat16 *__restrict output) {
  for (unsigned i = 0; i < DIM_M_C * DIM_N; i += 32) {
    const aie::accum<accfloat, 32> value(aie::load_v<32>(input + i));
    aie::store_v(output + i, value.template to_vector<bfloat16>());
  }
}

#ifdef ACCUM_FP32
static inline void matmul_bfp16_fp32(const bfloat16 *__restrict input_a,
                                     float *__restrict output_c,
                                     unsigned subtile) {
  output_c += subtile * DIM_M_A * DIM_N;
  constexpr unsigned row_blocks = DIM_M_A / 8;
  for (unsigned rb = 0; rb < row_blocks; rb += 2)
    chess_prepare_for_pipelining {
      float *__restrict c0 = output_c + rb * kNBlocks * 64;
      float *__restrict c1 = output_c + (rb + 1) * kNBlocks * 64;
      for (unsigned nb = 0; nb < kNBlocks; nb += 2)
        chess_flatten_loop {
          const bfloat16 *__restrict a0 = input_a + rb * kKBlocks * 64;
          const bfloat16 *__restrict a1 = input_a + (rb + 1) * kKBlocks * 64;
          auto c00 = load_fp32_accumulator(c0);
          auto c01 = load_fp32_accumulator(c0 + 64);
          auto c10 = load_fp32_accumulator(c1);
          auto c11 = load_fp32_accumulator(c1 + 64);
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
          store_fp32_accumulator(c0, c00);
          store_fp32_accumulator(c0 + 64, c01);
          store_fp32_accumulator(c1, c10);
          store_fp32_accumulator(c1 + 64, c11);
          c0 += 128;
          c1 += 128;
        }
    }
}
#endif
#ifdef ACCUM_CASCADE_RESIDENT
static inline void
matmul_bfp16_fp32_resident(const bfloat16 *__restrict input_a,
                           const uint8 *__restrict input_b,
                           float *__restrict output_c) {
  auto *__restrict weights =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(input_b));
  constexpr unsigned row_blocks = DIM_M_A / 8;
  for (unsigned rb = 0; rb < row_blocks; rb += 2)
    chess_prepare_for_pipelining {
      float *__restrict c0 = output_c + rb * kNBlocks * 64;
      float *__restrict c1 = output_c + (rb + 1) * kNBlocks * 64;
      for (unsigned nb = 0; nb < kNBlocks; nb += 2)
        chess_flatten_loop {
          const bfloat16 *__restrict a0 = input_a + rb * kKBlocks * 64;
          const bfloat16 *__restrict a1 = input_a + (rb + 1) * kKBlocks * 64;
          auto c00 = load_fp32_accumulator(c0);
          auto c01 = load_fp32_accumulator(c0 + 64);
          auto c10 = load_fp32_accumulator(c1);
          auto c11 = load_fp32_accumulator(c1 + 64);
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
              aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs0(weights);
              aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs1(weights);
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
          store_fp32_accumulator(c0, c00);
          store_fp32_accumulator(c0 + 64, c01);
          store_fp32_accumulator(c1, c10);
          store_fp32_accumulator(c1 + 64, c11);
          c0 += 128;
          c1 += 128;
        }
    }
}

static inline void cascade_resident_put(const float *__restrict local) {
  for (unsigned i = 0; i < DIM_M_C * DIM_N; i += 64)
    put_cascade_accumulator(load_fp32_accumulator(local + i));
}

static inline void cascade_resident_put_get(const float *__restrict local) {
  for (unsigned i = 0; i < DIM_M_C * DIM_N; i += 64) {
    const auto sum =
        aie::add(load_cascade_accumulator(), load_fp32_accumulator(local + i));
    put_cascade_accumulator(sum);
  }
}

static inline void cascade_resident_get_stream(float *__restrict local) {
  auto *__restrict compact = reinterpret_cast<bfloat16 *>(local);
  for (unsigned i = 0; i < DIM_M_C * DIM_N; i += 64) {
    const auto sum =
        aie::add(load_cascade_accumulator(), load_fp32_accumulator(local + i));
    aie::store_v(compact + i, sum.template to_vector<bfloat16>());
  }

  for (unsigned row = 0; row < DIM_M_C; ++row) {
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      const unsigned index = ((row / 8) * kNBlocks + nb) * 64 + (row % 8) * 8;
      const bool last = row + 1 == DIM_M_C && nb + 1 == kNBlocks;
      put_ms(aie::load_v<8>(compact + index).to_native(), last);
    }
  }
}
#endif

#ifdef ACCUM_CASCADE_REGISTER
constexpr unsigned kRegisterKBlocks = DIM_SHARD_K / 8;
static_assert(DIM_SHARD_K % 8 == 0, "cascade K shard must be 8-aligned");

static inline void cascade_register_mac(const bfloat16 *__restrict input_a,
                                        const uint8 *__restrict input_b,
                                        aie::accum<accfloat, 64> &c00,
                                        aie::accum<accfloat, 64> &c01,
                                        aie::accum<accfloat, 64> &c10,
                                        aie::accum<accfloat, 64> &c11) {
  const bfloat16 *__restrict a0 = input_a;
  const bfloat16 *__restrict a1 = input_a + kRegisterKBlocks * 64;
  auto *__restrict weights =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(input_b));

  for (unsigned kb = 0; kb < kRegisterKBlocks; ++kb)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      const auto av0 = aie::load_v<64>(a0);
      const auto av1 = aie::load_v<64>(a1);
      aie::accum<accfloat, 64> aa0;
      aie::accum<accfloat, 64> aa1;
      aa0 = av0;
      aa1 = av1;
      const auto ab0 = aa0.template to_vector<bfp16ebs8>();
      const auto ab1 = aa1.template to_vector<bfp16ebs8>();
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs0(weights);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs1(weights);
      bs0.seek(kb * 2);
      bs1.seek(kb * 2 + 1);
      const auto bv0 = bs0.pop();
      const auto bv1 = bs1.pop();
      c00 = mac_8x8_8x8T(ab0, bv0, c00);
      c01 = mac_8x8_8x8T(ab0, bv1, c01);
      c10 = mac_8x8_8x8T(ab1, bv0, c10);
      c11 = mac_8x8_8x8T(ab1, bv1, c11);
      a0 += 64;
      a1 += 64;
    }
}

static inline void cascade_register_add_input(aie::accum<accfloat, 64> &c00,
                                              aie::accum<accfloat, 64> &c01,
                                              aie::accum<accfloat, 64> &c10,
                                              aie::accum<accfloat, 64> &c11) {
  c00 = aie::add(c00, load_cascade_accumulator());
  c01 = aie::add(c01, load_cascade_accumulator());
  c10 = aie::add(c10, load_cascade_accumulator());
  c11 = aie::add(c11, load_cascade_accumulator());
}

static inline void cascade_register_put_output(
    const aie::accum<accfloat, 64> &c00, const aie::accum<accfloat, 64> &c01,
    const aie::accum<accfloat, 64> &c10, const aie::accum<accfloat, 64> &c11) {
  put_cascade_accumulator(c00);
  put_cascade_accumulator(c01);
  put_cascade_accumulator(c10);
  put_cascade_accumulator(c11);
}

static inline void
cascade_register_stream_pair(const aie::accum<accfloat, 64> &left,
                             const aie::accum<accfloat, 64> &right,
                             bool last_pair) {
  const auto left_bf16 = left.template to_vector<bfloat16>();
  const auto right_bf16 = right.template to_vector<bfloat16>();
  for (unsigned row = 0; row < 8; ++row) {
    put_ms(left_bf16.template extract<8>(row).to_native(), false);
    put_ms(right_bf16.template extract<8>(row).to_native(),
           last_pair && row == 7);
  }
}

static inline void cascade_register_local(const bfloat16 *__restrict input_a,
                                          const uint8 *__restrict input_b,
                                          aie::accum<accfloat, 64> &c00,
                                          aie::accum<accfloat, 64> &c01,
                                          aie::accum<accfloat, 64> &c10,
                                          aie::accum<accfloat, 64> &c11) {
  c00 = aie::zeros<accfloat, 64>();
  c01 = aie::zeros<accfloat, 64>();
  c10 = aie::zeros<accfloat, 64>();
  c11 = aie::zeros<accfloat, 64>();
  cascade_register_mac(input_a, input_b, c00, c01, c10, c11);
}
#endif

#ifdef ACCUM_CASCADE_CHUNKED
constexpr unsigned kChunkKBlocks = DIM_CHUNK_K / 8;
static_assert(DIM_CHUNK_K % 32 == 0, "cascade chunk must contain Q4_K groups");

static inline void cascade_chunk_mac(const bfloat16 *__restrict input_a,
                                     const uint8 *__restrict input_b,
                                     unsigned nb, aie::accum<accfloat, 64> &c00,
                                     aie::accum<accfloat, 64> &c01,
                                     aie::accum<accfloat, 64> &c10,
                                     aie::accum<accfloat, 64> &c11) {
  const bfloat16 *__restrict a0 = input_a;
  const bfloat16 *__restrict a1 = input_a + kChunkKBlocks * 64;
  auto *__restrict weights =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(input_b));

  for (unsigned kb = 0; kb < kChunkKBlocks; ++kb)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      const auto av0 = aie::load_v<64>(a0);
      const auto av1 = aie::load_v<64>(a1);
      aie::accum<accfloat, 64> aa0;
      aie::accum<accfloat, 64> aa1;
      aa0 = av0;
      aa1 = av1;
      const auto ab0 = aa0.template to_vector<bfp16ebs8>();
      const auto ab1 = aa1.template to_vector<bfp16ebs8>();
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs0(weights);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs1(weights);
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
}

static inline void cascade_chunk_local(
    const bfloat16 *__restrict input_a, const uint8 *__restrict input_b,
    unsigned nb, aie::accum<accfloat, 64> &c00, aie::accum<accfloat, 64> &c01,
    aie::accum<accfloat, 64> &c10, aie::accum<accfloat, 64> &c11) {
  c00 = aie::zeros<accfloat, 64>();
  c01 = aie::zeros<accfloat, 64>();
  c10 = aie::zeros<accfloat, 64>();
  c11 = aie::zeros<accfloat, 64>();
  cascade_chunk_mac(input_a, input_b, nb, c00, c01, c10, c11);
}

static inline void cascade_chunk_add_input(aie::accum<accfloat, 64> &c00,
                                           aie::accum<accfloat, 64> &c01,
                                           aie::accum<accfloat, 64> &c10,
                                           aie::accum<accfloat, 64> &c11) {
  c00 = aie::add(c00, load_cascade_accumulator());
  c01 = aie::add(c01, load_cascade_accumulator());
  c10 = aie::add(c10, load_cascade_accumulator());
  c11 = aie::add(c11, load_cascade_accumulator());
}

static inline void cascade_chunk_put_output(
    const aie::accum<accfloat, 64> &c00, const aie::accum<accfloat, 64> &c01,
    const aie::accum<accfloat, 64> &c10, const aie::accum<accfloat, 64> &c11) {
  put_cascade_accumulator(c00);
  put_cascade_accumulator(c01);
  put_cascade_accumulator(c10);
  put_cascade_accumulator(c11);
}

static inline void cascade_chunk_accumulate(const bfloat16 *__restrict input_a,
                                            const uint8 *__restrict input_b,
                                            float *__restrict output_c,
                                            unsigned rb) {
  for (unsigned nb = 0; nb < kNBlocks; nb += 2) {
    aie::accum<accfloat, 64> c00, c01, c10, c11;
    cascade_chunk_local(input_a, input_b, nb, c00, c01, c10, c11);
    cascade_chunk_add_input(c00, c01, c10, c11);
    float *__restrict c0 = output_c + ((rb * 2) * kNBlocks + nb) * 64;
    float *__restrict c1 = c0 + kNBlocks * 64;
    c00 = aie::add(c00, load_fp32_accumulator(c0));
    c01 = aie::add(c01, load_fp32_accumulator(c0 + 64));
    c10 = aie::add(c10, load_fp32_accumulator(c1));
    c11 = aie::add(c11, load_fp32_accumulator(c1 + 64));
    store_fp32_accumulator(c0, c00);
    store_fp32_accumulator(c0 + 64, c01);
    store_fp32_accumulator(c1, c10);
    store_fp32_accumulator(c1 + 64, c11);
  }
}

static inline void cascade_chunk_stream(float *__restrict output_c) {
  auto *__restrict compact = reinterpret_cast<bfloat16 *>(output_c);
  for (unsigned i = 0; i < DIM_M_C * DIM_N; i += 64) {
    const auto value = load_fp32_accumulator(output_c + i);
    aie::store_v(compact + i, value.template to_vector<bfloat16>());
  }
  for (unsigned row = 0; row < DIM_M_C; ++row) {
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      const unsigned index = ((row / 8) * kNBlocks + nb) * 64 + (row % 8) * 8;
      const bool last = row + 1 == DIM_M_C && nb + 1 == kNBlocks;
      put_ms(aie::load_v<8>(compact + index).to_native(), last);
    }
  }
}
#endif

#ifdef ACCUM_CASCADE_SHARED
constexpr unsigned kSharedHalfRows = DIM_M_C / 2;
constexpr unsigned kSharedHalfValues = kSharedHalfRows * DIM_N;
static_assert(DIM_M_C % 32 == 0,
              "shared C tile needs two 16-row-aligned halves");

static inline void cascade_shared_local(const bfloat16 *__restrict input_a,
                                        unsigned nb,
                                        aie::accum<accfloat, 64> &c00,
                                        aie::accum<accfloat, 64> &c01,
                                        aie::accum<accfloat, 64> &c10,
                                        aie::accum<accfloat, 64> &c11) {
  c00 = aie::zeros<accfloat, 64>();
  c01 = aie::zeros<accfloat, 64>();
  c10 = aie::zeros<accfloat, 64>();
  c11 = aie::zeros<accfloat, 64>();
  mac_bfp16_2x2(input_a, 0, nb, c00, c01, c10, c11);
}

static inline void cascade_shared_add_input(aie::accum<accfloat, 64> &c00,
                                            aie::accum<accfloat, 64> &c01,
                                            aie::accum<accfloat, 64> &c10,
                                            aie::accum<accfloat, 64> &c11) {
  c00 = aie::add(c00, load_cascade_accumulator());
  c01 = aie::add(c01, load_cascade_accumulator());
  c10 = aie::add(c10, load_cascade_accumulator());
  c11 = aie::add(c11, load_cascade_accumulator());
}

static inline void cascade_shared_put_output(
    const aie::accum<accfloat, 64> &c00, const aie::accum<accfloat, 64> &c01,
    const aie::accum<accfloat, 64> &c10, const aie::accum<accfloat, 64> &c11) {
  put_cascade_accumulator(c00);
  put_cascade_accumulator(c01);
  put_cascade_accumulator(c10);
  put_cascade_accumulator(c11);
}

static inline void cascade_shared_accumulate(const bfloat16 *__restrict input_a,
                                             float *__restrict c_low,
                                             float *__restrict c_high,
                                             unsigned rb) {
  constexpr unsigned pairs_per_half = kSharedHalfRows / 16;
  float *__restrict output_c = rb < pairs_per_half ? c_low : c_high;
  const unsigned local_rb = rb % pairs_per_half;
  for (unsigned nb = 0; nb < kNBlocks; nb += 2) {
    aie::accum<accfloat, 64> c00, c01, c10, c11;
    cascade_shared_local(input_a, nb, c00, c01, c10, c11);
    cascade_shared_add_input(c00, c01, c10, c11);
    float *__restrict c0 = output_c + ((local_rb * 2) * kNBlocks + nb) * 64;
    float *__restrict c1 = c0 + kNBlocks * 64;
    c00 = aie::add(c00, load_fp32_accumulator(c0));
    c01 = aie::add(c01, load_fp32_accumulator(c0 + 64));
    c10 = aie::add(c10, load_fp32_accumulator(c1));
    c11 = aie::add(c11, load_fp32_accumulator(c1 + 64));
    store_fp32_accumulator(c0, c00);
    store_fp32_accumulator(c0 + 64, c01);
    store_fp32_accumulator(c1, c10);
    store_fp32_accumulator(c1 + 64, c11);
  }
}

static inline void cascade_shared_zero(float *__restrict c_low,
                                       float *__restrict c_high) {
  const auto zero = aie::zeros<float, 32>();
  for (unsigned i = 0; i < kSharedHalfValues; i += 32) {
    aie::store_v(c_low + i, zero);
    aie::store_v(c_high + i, zero);
  }
}

static inline void cascade_shared_compact(float *__restrict input) {
  auto *__restrict compact = reinterpret_cast<bfloat16 *>(input);
  for (unsigned i = 0; i < kSharedHalfValues; i += 64) {
    const auto value = load_fp32_accumulator(input + i);
    aie::store_v(compact + i, value.template to_vector<bfloat16>());
  }
}

static inline void cascade_shared_stream(float *__restrict c_low,
                                         float *__restrict c_high) {
  cascade_shared_compact(c_low);
  cascade_shared_compact(c_high);
  auto *__restrict low = reinterpret_cast<bfloat16 *>(c_low);
  auto *__restrict high = reinterpret_cast<bfloat16 *>(c_high);
  for (unsigned row = 0; row < DIM_M_C; ++row) {
    const unsigned local_row = row % kSharedHalfRows;
    const bfloat16 *__restrict half = row < kSharedHalfRows ? low : high;
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      const unsigned index =
          ((local_row / 8) * kNBlocks + nb) * 64 + (local_row % 8) * 8;
      const bool last = row + 1 == DIM_M_C && nb + 1 == kNBlocks;
      put_ms(aie::load_v<8>(half + index).to_native(), last);
    }
  }
}
#endif

#ifdef ACCUM_CASCADE_HYBRID
template <bool GetCascade, bool PutCascade>
static inline void cascade_hybrid_finish(aie::accum<accfloat, 64> value,
                                         bfloat16 *__restrict output) {
  if constexpr (GetCascade)
    value = aie::add(load_cascade_accumulator(), value);
  if constexpr (PutCascade)
    put_cascade_accumulator(value);
  else
    aie::store_v(output, value.template to_vector<bfloat16>());
}

template <bool GetCascade, bool PutCascade>
static inline void
matmul_bfp16_cascade_finish(const bfloat16 *__restrict input_a,
                            bfloat16 *__restrict output_c, unsigned subtile) {
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
          cascade_hybrid_finish<GetCascade, PutCascade>(c00, c0);
          cascade_hybrid_finish<GetCascade, PutCascade>(c01, c0 + 64);
          cascade_hybrid_finish<GetCascade, PutCascade>(c10, c1);
          cascade_hybrid_finish<GetCascade, PutCascade>(c11, c1 + 64);
          c0 += 128;
          c1 += 128;
        }
    }
}
#endif

#ifdef ACCUM_CASCADE
static inline void cascade_put_only(const bfloat16 *__restrict input_a) {
  constexpr unsigned row_blocks = DIM_M_A / 8;
  for (unsigned rb = 0; rb < row_blocks; rb += 2)
    chess_prepare_for_pipelining {
      for (unsigned nb = 0; nb < kNBlocks; nb += 2)
        chess_flatten_loop {
          auto c00 = aie::zeros<accfloat, 64>();
          auto c01 = aie::zeros<accfloat, 64>();
          auto c10 = aie::zeros<accfloat, 64>();
          auto c11 = aie::zeros<accfloat, 64>();
          mac_bfp16_2x2(input_a, rb, nb, c00, c01, c10, c11);
          put_cascade_accumulator(c00);
          put_cascade_accumulator(c01);
          put_cascade_accumulator(c10);
          put_cascade_accumulator(c11);
        }
    }
}

static inline void cascade_put_get(const bfloat16 *__restrict input_a) {
  constexpr unsigned row_blocks = DIM_M_A / 8;
  for (unsigned rb = 0; rb < row_blocks; rb += 2)
    chess_prepare_for_pipelining {
      for (unsigned nb = 0; nb < kNBlocks; nb += 2)
        chess_flatten_loop {
          auto c00 = load_cascade_accumulator();
          auto c01 = load_cascade_accumulator();
          auto c10 = load_cascade_accumulator();
          auto c11 = load_cascade_accumulator();
          mac_bfp16_2x2(input_a, rb, nb, c00, c01, c10, c11);
          put_cascade_accumulator(c00);
          put_cascade_accumulator(c01);
          put_cascade_accumulator(c10);
          put_cascade_accumulator(c11);
        }
    }
}

static inline void cascade_get_only(const bfloat16 *__restrict input_a,
                                    float *__restrict output_c) {
  constexpr unsigned row_blocks = DIM_M_A / 8;
  for (unsigned rb = 0; rb < row_blocks; rb += 2)
    chess_prepare_for_pipelining {
      float *__restrict c0 = output_c + rb * kNBlocks * 64;
      float *__restrict c1 = output_c + (rb + 1) * kNBlocks * 64;
      for (unsigned nb = 0; nb < kNBlocks; nb += 2)
        chess_flatten_loop {
          auto c00 = load_cascade_accumulator();
          auto c01 = load_cascade_accumulator();
          auto c10 = load_cascade_accumulator();
          auto c11 = load_cascade_accumulator();
          mac_bfp16_2x2(input_a, rb, nb, c00, c01, c10, c11);
          c00 = aie::add(c00, load_fp32_accumulator(c0));
          c01 = aie::add(c01, load_fp32_accumulator(c0 + 64));
          c10 = aie::add(c10, load_fp32_accumulator(c1));
          c11 = aie::add(c11, load_fp32_accumulator(c1 + 64));
          store_fp32_accumulator(c0, c00);
          store_fp32_accumulator(c0 + 64, c01);
          store_fp32_accumulator(c1, c10);
          store_fp32_accumulator(c1 + 64, c11);
          c0 += 128;
          c1 += 128;
        }
    }
}
#endif
#endif
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
#if !defined(ACCUM_FP32) && !defined(ACCUM_CASCADE) &&                         \
    !defined(ACCUM_CASCADE_RESIDENT) && !defined(ACCUM_CASCADE_REGISTER) &&    \
    !defined(ACCUM_CASCADE_CHUNKED) && !defined(ACCUM_CASCADE_SHARED) &&       \
    !defined(ACCUM_CASCADE_HYBRID)
void q4ks_matmul_bfp16(bfloat16 *a, uint8 *b, bfloat16 *c, int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    prepare_bfp16(b);
  matmul_bfp16(a, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}

// A 64-row C stream is the internal fringe ABI for the fixed DIM_M_C=128
// kernel.  Full waves hold two such streams and reuse the one prepared B tile
// across both; a 256-row array fringe holds only the first stream.  Keeping
// both entry points in this object shares the same aligned BFP16 scratch.
void q4ks_matmul_bfp16_mhalf(bfloat16 *a, uint8 *b, bfloat16 *c, int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    prepare_bfp16(b);
  matmul_bfp16(a, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}

void q4ks_matmul_bfp16_mhalf_continue(bfloat16 *a, uint8 *b, bfloat16 *c,
                                      int subtile) {
  (void)b;
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  matmul_bfp16(a, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}

void q4ks_zero_bf16_mhalf(bfloat16 *output) {
  const auto zero = aie::zeros<bfloat16, 32>();
  for (unsigned i = 0; i < (DIM_M_C / 2) * DIM_N; i += 32)
    aie::store_v(output + i, zero);
}

void q4ks_copy_bf16_mhalf_buffer(bfloat16 *input, bfloat16 *output) {
  constexpr unsigned kHalfElements = (DIM_M_C / 2) * DIM_N;
  for (unsigned i = 0; i < kHalfElements; i += 32)
    aie::store_v(output + i, aie::load_v<32>(input + i));
}

// Paired-m64 schedule: one 128-row local result holds all four 32-row A
// subtiles.  Prepare B only for segment zero and use the segment directly as
// the output offset.  This removes the second Q4-to-BFP16 conversion without
// mutable global schedule state or a second C producer stream.
void q4ks_matmul_bfp16_pair(bfloat16 *a, uint8 *b, bfloat16 *c, int segment) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (segment == 0)
    prepare_bfp16(b);
  matmul_bfp16(a, c, static_cast<unsigned>(segment));
  aie::set_rounding(saved);
}

void q4ks_zero_bf16_pair(bfloat16 *output) {
  zero_bf16(output);
  zero_bf16(output + DIM_M_C * DIM_N);
}

// An odd 256-row array wave uses all four AIE rows for distinct 64-row
// results.  The paired C transport still carries two 64-row halves per core,
// so duplicate the completed half before the MemTile drain writes both views
// to the same host rows.  This copy is far cheaper than the removed duplicate
// MMUL work and keeps the high-throughput full-pair ABI unchanged.
void q4ks_copy_bf16_pair(bfloat16 *output) {
  constexpr unsigned kHalfElements = DIM_M_C * DIM_N;
  for (unsigned i = 0; i < kHalfElements; i += 32)
    aie::store_v(output + kHalfElements + i, aie::load_v<32>(output + i));
}

// The fixed 128-row high-throughput kernel also serves a final 256-row array
// half-wave.  Each physical row computes one selected pair of 32-row A
// subtiles; duplicate those 64 completed rows into the unused half solely so
// the existing 128-row producer object can be released in predictable chunks.
// Both transport views target the same logical host rows.  No duplicate MMUL
// or Q4-to-BFP16 conversion is performed.
void q4ks_copy_bf16_mhalf(bfloat16 *output) {
  constexpr unsigned kHalfElements = (DIM_M_C / 2) * DIM_N;
  for (unsigned i = 0; i < kHalfElements; i += 32)
    aie::store_v(output + kHalfElements + i, aie::load_v<32>(output + i));
}
#endif

#if defined(ACCUM_FP32) || defined(ACCUM_CASCADE) ||                           \
    defined(ACCUM_CASCADE_RESIDENT) || defined(ACCUM_CASCADE_CHUNKED)
void q4ks_zero_f32(float *output) { zero_fp32(output); }

void q4ks_store_bf16(float *input, bfloat16 *output) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  store_bf16(input, output);
  aie::set_rounding(saved);
}
#endif

#ifdef ACCUM_FP32
void q4ks_matmul_bfp16_fp32(bfloat16 *a, uint8 *b, float *c, int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    prepare_bfp16(b);
  matmul_bfp16_fp32(a, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}
#endif

#ifdef ACCUM_CASCADE_RESIDENT
void q4ks_cascade_resident_accumulate(bfloat16 *a, uint8 *b, float *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  matmul_bfp16_fp32_resident(a, b, c);
  aie::set_rounding(saved);
}

void q4ks_cascade_resident_put(float *c) { cascade_resident_put(c); }

void q4ks_cascade_resident_put_get(float *c) { cascade_resident_put_get(c); }

void q4ks_cascade_resident_get_stream(float *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  cascade_resident_get_stream(c);
  aie::set_rounding(saved);
}
#endif

#ifdef ACCUM_CASCADE_REGISTER
void q4ks_cascade_register_put(bfloat16 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  aie::accum<accfloat, 64> c00;
  aie::accum<accfloat, 64> c01;
  aie::accum<accfloat, 64> c10;
  aie::accum<accfloat, 64> c11;
  cascade_register_local(a, b, c00, c01, c10, c11);
  cascade_register_put_output(c00, c01, c10, c11);
  aie::set_rounding(saved);
}

void q4ks_cascade_register_put_get(bfloat16 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  aie::accum<accfloat, 64> c00;
  aie::accum<accfloat, 64> c01;
  aie::accum<accfloat, 64> c10;
  aie::accum<accfloat, 64> c11;
  cascade_register_local(a, b, c00, c01, c10, c11);
  cascade_register_add_input(c00, c01, c10, c11);
  cascade_register_put_output(c00, c01, c10, c11);
  aie::set_rounding(saved);
}

void q4ks_cascade_register_get_stream(bfloat16 *a, uint8 *b, int last) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  aie::accum<accfloat, 64> c00;
  aie::accum<accfloat, 64> c01;
  aie::accum<accfloat, 64> c10;
  aie::accum<accfloat, 64> c11;
  cascade_register_local(a, b, c00, c01, c10, c11);
  cascade_register_add_input(c00, c01, c10, c11);
  cascade_register_stream_pair(c00, c01, false);
  cascade_register_stream_pair(c10, c11, last != 0);
  aie::set_rounding(saved);
}
#endif

#ifdef ACCUM_CASCADE_CHUNKED
void q4ks_cascade_chunk_put(bfloat16 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned nb = 0; nb < kNBlocks; nb += 2) {
    aie::accum<accfloat, 64> c00, c01, c10, c11;
    cascade_chunk_local(a, b, nb, c00, c01, c10, c11);
    cascade_chunk_put_output(c00, c01, c10, c11);
  }
  aie::set_rounding(saved);
}

void q4ks_cascade_chunk_put_get(bfloat16 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  for (unsigned nb = 0; nb < kNBlocks; nb += 2) {
    aie::accum<accfloat, 64> c00, c01, c10, c11;
    cascade_chunk_local(a, b, nb, c00, c01, c10, c11);
    cascade_chunk_add_input(c00, c01, c10, c11);
    cascade_chunk_put_output(c00, c01, c10, c11);
  }
  aie::set_rounding(saved);
}

void q4ks_cascade_chunk_accumulate(bfloat16 *a, uint8 *b, float *c, int rb) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  cascade_chunk_accumulate(a, b, c, static_cast<unsigned>(rb));
  aie::set_rounding(saved);
}

void q4ks_cascade_chunk_stream(float *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  cascade_chunk_stream(c);
  aie::set_rounding(saved);
}
#endif

#ifdef ACCUM_CASCADE_SHARED
void q4ks_cascade_shared_put(bfloat16 *a, uint8 *b, int rb) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (rb == 0)
    prepare_bfp16(b);
  for (unsigned nb = 0; nb < kNBlocks; nb += 2) {
    aie::accum<accfloat, 64> c00, c01, c10, c11;
    cascade_shared_local(a, nb, c00, c01, c10, c11);
    cascade_shared_put_output(c00, c01, c10, c11);
  }
  aie::set_rounding(saved);
}

void q4ks_cascade_shared_put_get(bfloat16 *a, uint8 *b, int rb) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (rb == 0)
    prepare_bfp16(b);
  for (unsigned nb = 0; nb < kNBlocks; nb += 2) {
    aie::accum<accfloat, 64> c00, c01, c10, c11;
    cascade_shared_local(a, nb, c00, c01, c10, c11);
    cascade_shared_add_input(c00, c01, c10, c11);
    cascade_shared_put_output(c00, c01, c10, c11);
  }
  aie::set_rounding(saved);
}

void q4ks_cascade_shared_accumulate(bfloat16 *a, uint8 *b, float *c_low,
                                    float *c_high, int rb) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (rb == 0)
    prepare_bfp16(b);
  cascade_shared_accumulate(a, c_low, c_high, static_cast<unsigned>(rb));
  aie::set_rounding(saved);
}

void q4ks_cascade_shared_zero(float *c_low, float *c_high) {
  cascade_shared_zero(c_low, c_high);
}

void q4ks_cascade_shared_stream(float *c_low, float *c_high) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  cascade_shared_stream(c_low, c_high);
  aie::set_rounding(saved);
}
#endif

#ifdef ACCUM_CASCADE_HYBRID
static inline void hybrid_accumulate_entry(bfloat16 *a, uint8 *b, bfloat16 *c,
                                           int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    prepare_bfp16(b);
  matmul_bfp16(a, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}

static inline void hybrid_final_entry(bfloat16 *a, uint8 *b, bfloat16 *c,
                                      int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    prepare_bfp16(b);
#if CASCADE_ROLE == 0
  matmul_bfp16_cascade_finish<false, true>(a, c,
                                           static_cast<unsigned>(subtile));
#elif CASCADE_ROLE == 1
  matmul_bfp16_cascade_finish<true, true>(a, c, static_cast<unsigned>(subtile));
#elif CASCADE_ROLE == 2
  matmul_bfp16_cascade_finish<true, false>(a, c,
                                           static_cast<unsigned>(subtile));
#endif
  aie::set_rounding(saved);
}

#if CASCADE_ROLE == 0
void q4ks_zero_bf16_bottom(bfloat16 *c) { zero_bf16(c); }

void q4ks_cascade_hybrid_bottom_accumulate(bfloat16 *a, uint8 *b, bfloat16 *c,
                                           int subtile) {
  hybrid_accumulate_entry(a, b, c, subtile);
}

void q4ks_cascade_hybrid_bottom_final(bfloat16 *a, uint8 *b, bfloat16 *c,
                                      int subtile) {
  hybrid_final_entry(a, b, c, subtile);
}
#elif CASCADE_ROLE == 1
void q4ks_zero_bf16_middle(bfloat16 *c) { zero_bf16(c); }

void q4ks_cascade_hybrid_middle_accumulate(bfloat16 *a, uint8 *b, bfloat16 *c,
                                           int subtile) {
  hybrid_accumulate_entry(a, b, c, subtile);
}

void q4ks_cascade_hybrid_middle_final(bfloat16 *a, uint8 *b, bfloat16 *c,
                                      int subtile) {
  hybrid_final_entry(a, b, c, subtile);
}
#elif CASCADE_ROLE == 2
void q4ks_zero_bf16_top(bfloat16 *c) { zero_bf16(c); }

void q4ks_cascade_hybrid_top_accumulate(bfloat16 *a, uint8 *b, bfloat16 *c,
                                        int subtile) {
  hybrid_accumulate_entry(a, b, c, subtile);
}

void q4ks_cascade_hybrid_top_final(bfloat16 *a, uint8 *b, bfloat16 *c,
                                   int subtile) {
  hybrid_final_entry(a, b, c, subtile);
}
#else
#error "CASCADE_ROLE must be 0 (bottom), 1 (middle), or 2 (top)"
#endif
#endif

#ifdef ACCUM_CASCADE
void q4ks_cascade_put_only(bfloat16 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  prepare_bfp16(b);
  cascade_put_only(a);
  aie::set_rounding(saved);
}

void q4ks_cascade_put_get(bfloat16 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  prepare_bfp16(b);
  cascade_put_get(a);
  aie::set_rounding(saved);
}

void q4ks_cascade_get_only(bfloat16 *a, uint8 *b, float *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  prepare_bfp16(b);
  cascade_get_only(a, c);
  aie::set_rounding(saved);
}
#endif
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
