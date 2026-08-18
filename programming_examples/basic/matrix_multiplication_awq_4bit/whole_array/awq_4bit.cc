//===- awq_4bit.cc ----------------------------------------000---*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <type_traits>

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_K
#define DIM_K 128
#endif
#ifndef DIM_N
#define DIM_N 64
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif

namespace {

constexpr unsigned kMmulM = 8;
constexpr unsigned kMmulK = 8;
constexpr unsigned kMmulN = 8;
constexpr unsigned kHalfRows = DIM_M / 2;
constexpr unsigned kWeightBytes = DIM_K * DIM_N / 2;
constexpr unsigned kGroups = DIM_K / GROUP_SIZE;
constexpr unsigned kScaleBytes = kGroups * DIM_N * sizeof(bfloat16);
constexpr unsigned kZeroBytes = kGroups * DIM_N * 2;
constexpr unsigned kRawTileBytes = kWeightBytes + kScaleBytes + kZeroBytes;

static_assert(DIM_M % 32 == 0,
              "DIM_M must split into two even pairs of 8-row MMUL tiles");
static_assert(DIM_K % kMmulK == 0, "DIM_K must be divisible by 8");
static_assert(DIM_N % (2 * kMmulN) == 0, "DIM_N must be divisible by 16");
static_assert(GROUP_SIZE == 32 || GROUP_SIZE == 64 || GROUP_SIZE == 128,
              "GROUP_SIZE must be 32, 64, or 128");
static_assert(DIM_K % GROUP_SIZE == 0, "GROUP_SIZE must divide DIM_K");
#ifdef PACKED_TILE_BYTES
static_assert(kRawTileBytes <= PACKED_TILE_BYTES,
              "packed tile is smaller than its weights/scales/zeros");
static_assert(PACKED_TILE_BYTES % DIM_K == 0,
              "packed tile must be padded to a whole number of K-byte rows");
#endif

// Each compute tile owns its own aligned scratch allocation.  It is populated
// once per packed B tile (on explicit A half 0) and reused by half 1.
alignas(aie::vector_decl_align) static bfloat16 b_dequantized[DIM_K * DIM_N];

template <typename T>
static inline void zero_tile(T *__restrict output) {
  constexpr unsigned lanes = 512 / (sizeof(T) * 8);
  static_assert((DIM_M * DIM_N) % lanes == 0);
  const aie::vector<T, lanes> zeros = aie::zeros<T, lanes>();
  T *__restrict end = output + DIM_M * DIM_N;
  event0();
  for (; output < end; output += lanes)
    aie::store_v(output, zeros);
  event1();
}

static inline void dequantize_tile(const uint8 *__restrict packed) {
  constexpr unsigned k_blocks = DIM_K / kMmulK;
  constexpr unsigned n_blocks = DIM_N / kMmulN;
  constexpr unsigned values_per_microtile = kMmulK * kMmulN;
  constexpr unsigned bytes_per_microtile = values_per_microtile / 2;

  const uint8 *__restrict packed_weights = packed;
  const bfloat16 *__restrict scales =
      reinterpret_cast<const bfloat16 *>(packed + kWeightBytes);
  const uint8 *__restrict zeros = packed + kWeightBytes + kScaleBytes;

  for (unsigned kb = 0; kb < k_blocks; ++kb)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      const unsigned group = (kb * kMmulK) / GROUP_SIZE;
      for (unsigned nb = 0; nb < n_blocks; ++nb)
        chess_flatten_loop {
          const unsigned block = kb * n_blocks + nb;
          const auto packed_bytes = aie::load_v<bytes_per_microtile>(
              packed_weights + block * bytes_per_microtile);
          const auto q4 = packed_bytes.template cast_to<uint4>();
          const auto q8 = q4.template unpack();
          const auto q8_signed = q8.template cast_to<int8>();

          // Scale layout is [group][N].  Repeating one 8-column vector eight
          // times matches the row-major 8x8 weight microtile.
          const auto scale8 =
              aie::load_v<kMmulN>(scales + group * DIM_N + nb * kMmulN);
          const auto scale64 =
              scale8.template grow_replicate<values_per_microtile>();

          // Every logical 8-byte zero vector is stored twice.  A native
          // 16-byte load followed by grow_replicate therefore yields the same
          // per-column pattern over all eight microtile rows.
          const auto zero16 =
              aie::load_v<16>(zeros + group * DIM_N * 2 + nb * 16);
          const auto zero64_u8 =
              zero16.template grow_replicate<values_per_microtile>();
          const auto zero64 = zero64_u8.template cast_to<int8>();

          const auto centered = aie::sub(q8_signed, zero64);
          const auto centered_bf16 = aie::to_float<bfloat16>(centered);
          const auto scaled = aie::mul(centered_bf16, scale64);
          const auto result = scaled.template to_vector<bfloat16>();
          aie::store_v(b_dequantized + block * values_per_microtile, result);
        }
    }
}

template <typename T_out>
static inline void matmul_half(const bfloat16 *__restrict input_a,
                               T_out *__restrict output_c,
                               unsigned half_index) {
  using MMUL = aie::mmul<kMmulM, kMmulK, kMmulN, bfloat16, bfloat16, accauto>;
  constexpr unsigned row_blocks = kHalfRows / kMmulM;
  constexpr unsigned k_blocks = DIM_K / kMmulK;
  constexpr unsigned n_blocks = DIM_N / kMmulN;

  output_c += half_index * kHalfRows * DIM_N;

  for (unsigned rb = 0; rb < row_blocks; rb += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      T_out *__restrict c_top = output_c + rb * n_blocks * MMUL::size_C;
      T_out *__restrict c_bottom =
          output_c + (rb + 1) * n_blocks * MMUL::size_C;

      for (unsigned nb = 0; nb < n_blocks; nb += 2)
#ifdef OPT_PERF_ENABLED
        chess_flatten_loop
#endif
        {
          const bfloat16 *__restrict a_top =
              input_a + rb * k_blocks * MMUL::size_A;
          const bfloat16 *__restrict a_bottom =
              input_a + (rb + 1) * k_blocks * MMUL::size_A;
          const bfloat16 *__restrict b_left = b_dequantized + nb * MMUL::size_B;
          const bfloat16 *__restrict b_right =
              b_dequantized + (nb + 1) * MMUL::size_B;

          MMUL c00(aie::load_v<MMUL::size_C>(c_top));
          MMUL c01(aie::load_v<MMUL::size_C>(c_top + MMUL::size_C));
          MMUL c10(aie::load_v<MMUL::size_C>(c_bottom));
          MMUL c11(aie::load_v<MMUL::size_C>(c_bottom + MMUL::size_C));

          for (unsigned kb = 0; kb < k_blocks; ++kb)
#ifdef OPT_PERF_ENABLED
            chess_flatten_loop
#endif
            {
              const auto a0 = aie::load_v<MMUL::size_A>(a_top);
              const auto a1 = aie::load_v<MMUL::size_A>(a_bottom);
              const auto b0 = aie::load_v<MMUL::size_B>(b_left);
              const auto b1 = aie::load_v<MMUL::size_B>(b_right);
              c00.mac(a0, b0);
              c01.mac(a0, b1);
              c10.mac(a1, b0);
              c11.mac(a1, b1);
              a_top += MMUL::size_A;
              a_bottom += MMUL::size_A;
              b_left += n_blocks * MMUL::size_B;
              b_right += n_blocks * MMUL::size_B;
            }

          aie::store_v(c_top, c00.template to_vector<T_out>());
          aie::store_v(c_top + MMUL::size_C, c01.template to_vector<T_out>());
          aie::store_v(c_bottom, c10.template to_vector<T_out>());
          aie::store_v(c_bottom + MMUL::size_C,
                       c11.template to_vector<T_out>());
          c_top += 2 * MMUL::size_C;
          c_bottom += 2 * MMUL::size_C;
        }
    }
}

template <typename T_out>
static inline void awq_matmul(const bfloat16 *__restrict input_a,
                              const uint8 *__restrict packed_b,
                              T_out *__restrict output_c, int half_index) {
  event0();
  const aie::rounding_mode saved_rounding =
      aie::swap_rounding(aie::rounding_mode::conv_even);
  if (half_index == 0)
    dequantize_tile(packed_b);
  matmul_half(input_a, output_c, static_cast<unsigned>(half_index));
  aie::set_rounding(saved_rounding);
  event1();
}

} // namespace

extern "C" {

#ifdef AWQ_OUTPUT_BF16
void awq_matmul_bf16_bf16(bfloat16 *input_a, uint8 *packed_b,
                          bfloat16 *output_c, int half_index) {
  awq_matmul(input_a, packed_b, output_c, half_index);
}

void awq_zero_bf16(bfloat16 *output_c) { zero_tile(output_c); }
#endif

#ifdef AWQ_OUTPUT_F32
void awq_matmul_bf16_f32(bfloat16 *input_a, uint8 *packed_b, float *output_c,
                         int half_index) {
  awq_matmul(input_a, packed_b, output_c, half_index);
}

void awq_zero_f32(float *output_c) { zero_tile(output_c); }
#endif

} // extern "C"
