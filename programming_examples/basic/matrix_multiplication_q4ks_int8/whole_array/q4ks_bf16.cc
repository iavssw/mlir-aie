// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define NOCPP
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
constexpr unsigned KBLKS = DIM_K / 8;
constexpr unsigned NBLKS = DIM_N / 8;
constexpr unsigned GROUPS = DIM_K / 32;
constexpr unsigned WEIGHT_BYTES = DIM_K * DIM_N / 2;
constexpr unsigned RAW_BYTES = WEIGHT_BYTES + GROUPS * DIM_N * 4;
static_assert(DIM_M_C % DIM_M_A == 0);
static_assert(DIM_M_A % 16 == 0);
static_assert(DIM_K % 32 == 0);
static_assert(DIM_N % 16 == 0);
static_assert(RAW_BYTES <= PACKED_TILE_BYTES);

alignas(aie::vector_decl_align) static bfloat16 weights[DIM_K * DIM_N];

static inline const bfloat16 *metadata(const uint8 *packed, unsigned group,
                                       unsigned nb) {
  return reinterpret_cast<const bfloat16 *>(packed + WEIGHT_BYTES +
                                            (group * NBLKS + nb) * 32);
}

static inline void dequantize(const uint8 *__restrict packed) {
  for (unsigned kb = 0; kb < KBLKS; ++kb)
    chess_prepare_for_pipelining chess_loop_range(4, ) {
      const unsigned group = kb / 4;
      for (unsigned nb = 0; nb < NBLKS; ++nb)
        chess_flatten_loop {
          const unsigned block = kb * NBLKS + nb;
          const auto bytes = aie::load_v<32>(packed + block * 32);
          const auto q = bytes.template cast_to<uint4>()
                             .template unpack()
                             .template cast_to<int8>();
          const bfloat16 *params = metadata(packed, group, nb);
          const auto scale =
              aie::load_v<8>(params).template grow_replicate<64>();
          const auto bias =
              aie::load_v<8>(params + 8).template grow_replicate<64>();
          const auto qbf = aie::to_float<bfloat16>(q);
          const auto result = aie::sub(aie::mul(qbf, scale), bias);
          aie::store_v(weights + block * 64,
                       result.template to_vector<bfloat16>());
        }
    }
}

static inline void matmul(const bfloat16 *__restrict a, bfloat16 *__restrict c,
                          unsigned subtile) {
  using MMUL = aie::mmul<8, 8, 8, bfloat16, bfloat16, accauto>;
  c += subtile * DIM_M_A * DIM_N;
  for (unsigned rb = 0; rb < DIM_M_A / 8; rb += 2)
    chess_prepare_for_pipelining chess_loop_range(2, ) {
      bfloat16 *c0 = c + rb * NBLKS * 64;
      bfloat16 *c1 = c + (rb + 1) * NBLKS * 64;
      for (unsigned nb = 0; nb < NBLKS; nb += 2)
        chess_flatten_loop {
          const bfloat16 *a0 = a + rb * KBLKS * 64;
          const bfloat16 *a1 = a + (rb + 1) * KBLKS * 64;
          const bfloat16 *b0 = weights + nb * 64;
          const bfloat16 *b1 = weights + (nb + 1) * 64;
          MMUL c00(aie::load_v<64>(c0));
          MMUL c01(aie::load_v<64>(c0 + 64));
          MMUL c10(aie::load_v<64>(c1));
          MMUL c11(aie::load_v<64>(c1 + 64));
          for (unsigned kb = 0; kb < KBLKS; ++kb)
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
              b0 += NBLKS * 64;
              b1 += NBLKS * 64;
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
} // namespace

extern "C" {
void q4ks_zero_bf16(bfloat16 *c) {
  const auto zero = aie::zeros<bfloat16, 32>();
  for (unsigned i = 0; i < DIM_M_C * DIM_N; i += 32)
    aie::store_v(c + i, zero);
}

void q4ks_matmul_bf16(bfloat16 *a, uint8 *b, bfloat16 *c, int subtile) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  if (subtile == 0)
    dequantize(b);
  matmul(a, c, static_cast<unsigned>(subtile));
  aie::set_rounding(saved);
}
}
