//===- q4ks_systolic.cc -----------------------------------------*- C++ -*-===//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define NOCPP

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K_STAGE
#define DIM_K_STAGE 512
#endif
#ifndef DIM_N
#define DIM_N 32
#endif
#ifndef A_CHUNK_K
#define A_CHUNK_K 64
#endif
#ifndef Q4_TILE_BYTES
#define Q4_TILE_BYTES (DIM_K_STAGE * DIM_N * 5 / 8)
#endif
#ifndef N_PANELS
#define N_PANELS 1
#endif
#ifndef C_PANEL_SLAB
#define C_PANEL_SLAB 1
#endif
#ifndef KERNEL_STORAGE
#define KERNEL_STORAGE 0
#endif
#ifndef KERNEL_ROLE
#define KERNEL_ROLE 0
#endif
#ifndef DIRECT_C_STREAM
#define DIRECT_C_STREAM 0
#endif

namespace {

constexpr unsigned kMicro = 8;
constexpr unsigned kGroup = 32;
constexpr unsigned kRowBlocks = DIM_M / kMicro;
constexpr unsigned kKBlocks = DIM_K_STAGE / kMicro;
constexpr unsigned kNBlocks = DIM_N / kMicro;
constexpr unsigned kGroups = DIM_K_STAGE / kGroup;
constexpr unsigned kWeightBytes = DIM_K_STAGE * DIM_N / 2;
constexpr unsigned kMetadataBytes = kGroups * DIM_N * 4;
constexpr unsigned kBfpBytes = DIM_K_STAGE * DIM_N * 9 / 8;
constexpr unsigned kStreamBatchBlocks = 16;
constexpr unsigned kStreamBatchBytes = kStreamBatchBlocks * 64 * 9 / 8;

static_assert(DIM_M == 32 || DIM_M == 64,
              "the POC supports 32- or 64-row stationary tiles");
static_assert(DIM_K_STAGE % 256 == 0, "each stage must own Q4_K blocks");
static_assert(DIM_N == 16 || DIM_N == 32 || DIM_N == 64,
              "N panel must be 16, 32, or 64");
static_assert(A_CHUNK_K == 16 || A_CHUNK_K == 32 || A_CHUNK_K == 64,
              "activation conversion uses 16-, 32-, or 64-K chunks");
static_assert(C_PANEL_SLAB == 1 || C_PANEL_SLAB == 2 ||
                  C_PANEL_SLAB == 4 || C_PANEL_SLAB == 8,
              "C panel slab must contain 1, 2, 4, or 8 panels");
static_assert(KERNEL_ROLE <= 4, "invalid helper/compute kernel role");
static_assert(kWeightBytes + kMetadataBytes == Q4_TILE_BYTES,
              "unexpected Q4 tile size");
static_assert(Q4_TILE_BYTES % 64 == 0, "Q4 forwarding uses 64-byte beats");
static_assert(kBfpBytes % 64 == 0, "BFP forwarding uses 64-byte beats");
static_assert(kKBlocks % kStreamBatchBlocks == 0,
              "K stage must contain complete streamed BFP batches");

#if KERNEL_STORAGE == 0 || KERNEL_STORAGE == 3 ||                         \
    (KERNEL_STORAGE == 8 && DIM_N == 32)
alignas(aie::vector_decl_align) static bfp16ebs8 b_bfp[DIM_K_STAGE * DIM_N / 8];
#endif

#if KERNEL_STORAGE == 0 || KERNEL_STORAGE == 3 || KERNEL_STORAGE == 4 ||      \
    KERNEL_STORAGE == 5 || KERNEL_STORAGE == 8 || KERNEL_STORAGE == 9
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

#endif // Q4 metadata helpers

#if KERNEL_STORAGE == 0 || KERNEL_STORAGE == 3 ||                         \
    (KERNEL_STORAGE == 8 && DIM_N == 32)
static inline void prepare_q4(const uint8 *__restrict packed) {
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
          aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(b_bfp);
          const auto affine = aie::sub(aie::mul(qbf16, scale64), bias64);
          const auto transposed =
              aie::transpose(affine.template to_vector<bfloat16>(), 8, 8);
          aie::accum<accfloat, 64> value;
          value = transposed;
          out.seek(nb * kKBlocks + kb);
          out.push(value.template to_vector<bfp16ebs8>());
        }
      }
    }
}

#endif // Q4 to BFP preparation

static inline aie::vector<bfloat16, 64>
load_row_major_a_block(const bfloat16 *__restrict input, unsigned rb,
                       unsigned kb) {
  const bfloat16 *__restrict base = input + rb * 8 * A_CHUNK_K + kb * 8;
  const auto r0 = aie::load_v<8>(base + 0 * A_CHUNK_K);
  const auto r1 = aie::load_v<8>(base + 1 * A_CHUNK_K);
  const auto r2 = aie::load_v<8>(base + 2 * A_CHUNK_K);
  const auto r3 = aie::load_v<8>(base + 3 * A_CHUNK_K);
  const auto r4 = aie::load_v<8>(base + 4 * A_CHUNK_K);
  const auto r5 = aie::load_v<8>(base + 5 * A_CHUNK_K);
  const auto r6 = aie::load_v<8>(base + 6 * A_CHUNK_K);
  const auto r7 = aie::load_v<8>(base + 7 * A_CHUNK_K);
  return aie::concat(aie::concat(aie::concat(r0, r1), aie::concat(r2, r3)),
                     aie::concat(aie::concat(r4, r5), aie::concat(r6, r7)));
}

static inline void convert_a_chunk(const bfloat16 *__restrict input,
                                   uint8 *__restrict output, unsigned chunk) {
  auto *__restrict blocks = reinterpret_cast<bfp16ebs8 *>(output);
  constexpr unsigned chunk_blocks = A_CHUNK_K / 8;
  for (unsigned rb = 0; rb < kRowBlocks; ++rb) {
    for (unsigned kb = 0; kb < chunk_blocks; ++kb) {
      aie::accum<accfloat, 64> value;
      value = load_row_major_a_block(input, rb, kb);
      aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(blocks);
      out.seek(rb * kKBlocks + chunk * chunk_blocks + kb);
      out.push(value.template to_vector<bfp16ebs8>());
    }
  }
}

static inline aie::accum<accfloat, 64> load_cascade() {
  return aie::accum<accfloat, 64>(get_scd_v64accfloat(1));
}

static inline void put_cascade(const aie::accum<accfloat, 64> &value) {
  const auto native = value.to_native();
  put_mcd(extract_v16accfloat(native, 0));
  put_mcd(extract_v16accfloat(native, 1));
  put_mcd(extract_v16accfloat(native, 2));
  put_mcd(extract_v16accfloat(native, 3));
}

static inline void forward_bytes(const uint8 *__restrict input, unsigned bytes,
                                 bool end_of_object = true) {
  for (unsigned offset = 0; offset < bytes; offset += 64) {
    const bool last = end_of_object && offset + 64 == bytes;
    put_ms(aie::load_v<64>(input + offset).to_native(), last);
  }
}

#if KERNEL_STORAGE == 9
// A complete BFP tile would consume 18 KiB for the default 512x32 shard and
// cannot coexist with stationary A, Q4 input, grouped C, and the compute B
// FIFO on the north/east core.  Eight BFP 8x8 blocks occupy exactly nine
// 64-byte stream beats, so encode and emit that bounded batch instead.
alignas(aie::vector_decl_align) static bfp16ebs8 bfp_stream_batch[64];
static_assert(sizeof(bfp_stream_batch) == 576,
              "eight BFP 8x8 blocks must occupy nine 64-byte beats");

static inline void stream_q4_as_bfp(const uint8 *__restrict packed,
                                    bool last_panel) {
  unsigned batch_block = 0;
  for (unsigned nb = 0; nb < kNBlocks; ++nb) {
    for (unsigned group = 0; group < kGroups; ++group)
      chess_prepare_for_pipelining {
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
          const auto transposed =
              aie::transpose(affine.template to_vector<bfloat16>(), 8, 8);
          aie::accum<accfloat, 64> value;
          value = transposed;
          aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(
              bfp_stream_batch);
          out.seek(batch_block++);
          out.push(value.template to_vector<bfp16ebs8>());

          if (batch_block == 8) {
            const bool final_microtile = nb + 1 == kNBlocks &&
                                         group + 1 == kGroups && inner == 3;
            forward_bytes(reinterpret_cast<const uint8 *>(bfp_stream_batch),
                          sizeof(bfp_stream_batch),
                          last_panel && final_microtile);
            batch_block = 0;
          }
        }
      }
  }
}
#endif

static inline void mac_local(const uint8 *__restrict a_storage,
                             const uint8 *__restrict b_storage, unsigned rb,
                             unsigned nb,
                             aie::accum<accfloat, 64> &accumulator) {
  auto *__restrict a_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(a_storage));
  auto *__restrict b_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(b_storage));
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as(a_blocks);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs(b_blocks);
  as.seek(rb * kKBlocks);
  bs.seek(nb * kKBlocks);
  auto odd = aie::zeros<accfloat, 64>();
  for (unsigned kb = 0; kb < kKBlocks; kb += 2)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      accumulator = mac_8x8_8x8T(as.pop(), bs.pop(), accumulator);
      odd = mac_8x8_8x8T(as.pop(), bs.pop(), odd);
    }
  accumulator = aie::add(accumulator, odd);
}

static inline void store_8x8(bfloat16 *__restrict output,
                             const aie::accum<accfloat, 64> &accumulator) {
  const auto value = accumulator.template to_vector<bfloat16>();
  for (unsigned row = 0; row < 8; ++row) {
    aie::store_v(output + row * 16, value.template extract<8>(row));
  }
}

// The direct-output ceiling avoids an L1 C ObjectFIFO.  Cascade tokens still
// arrive and accumulate as accfloat; only the completed eastern result is
// converted.  Retaining one complete 8-row band lets the emitted stream be
// ordinary row-major data for any supported N tile.
static inline void run_panel_east_stream(const uint8 *__restrict a,
                                         const uint8 *__restrict b) {
  for (unsigned rb = 0; rb < kRowBlocks; ++rb) {
    aie::accum<accfloat, 64> accumulators[kNBlocks];
    aie::vector<bfloat16, 64> values[kNBlocks];
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      accumulators[nb] = load_cascade();
      mac_local(a, b, rb, nb, accumulators[nb]);
      values[nb] = accumulators[nb].template to_vector<bfloat16>();
    }
    for (unsigned row = 0; row < 8; ++row) {
      for (unsigned nb = 0; nb < kNBlocks; ++nb) {
        const bool last = rb + 1 == kRowBlocks && row == 7 &&
                          nb + 1 == kNBlocks;
        put_ms(values[nb].template extract<8>(row).to_native(), last);
      }
    }
  }
}

enum class Role { West, Middle, East };

template <Role role>
static inline void run_panel(const uint8 *__restrict a,
                             const uint8 *__restrict b,
                             bfloat16 *__restrict c,
                             unsigned c_panel = 0) {
  for (unsigned rb = 0; rb < kRowBlocks; ++rb) {
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      aie::accum<accfloat, 64> accumulator;
      if constexpr (role != Role::West) {
        accumulator = load_cascade();
      } else {
        accumulator = aie::zeros<accfloat, 64>();
      }
      mac_local(a, b, rb, nb, accumulator);
      if constexpr (role != Role::East) {
        put_cascade(accumulator);
      } else {
        constexpr unsigned panel_tokens = kNBlocks / 2;
        const unsigned macro_token =
            (rb / 2) * (C_PANEL_SLAB * panel_tokens) +
            c_panel * panel_tokens + (nb / 2);
        bfloat16 *__restrict output =
            c + macro_token * 256 + (rb % 2) * 128 + (nb % 2) * 8;
        store_8x8(output, accumulator);
      }
    }
  }
}

// Reuse each BFP K block across four 8-row activation blocks.  Four
// independent accumulators provide enough dependency distance for native
// BFP16 MMUL issue without the even/odd partial-accumulator split used by the
// scalar microtile schedule.
template <Role role>
static inline void run_panel_bfp_blocked(const uint8 *__restrict a,
                                         const uint8 *__restrict b,
                                         bfloat16 *__restrict c,
                                         unsigned c_panel = 0) {
  static_assert(kRowBlocks % 4 == 0, "BFP blocked path requires M/8 % 4 == 0");
  auto *__restrict a_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(a));
  auto *__restrict b_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(b));

  for (unsigned rb_base = 0; rb_base < kRowBlocks; rb_base += 4) {
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      auto acc0 = aie::zeros<accfloat, 64>();
      auto acc1 = aie::zeros<accfloat, 64>();
      auto acc2 = aie::zeros<accfloat, 64>();
      auto acc3 = aie::zeros<accfloat, 64>();
      if constexpr (role != Role::West) {
        acc0 = load_cascade();
        acc1 = load_cascade();
        acc2 = load_cascade();
        acc3 = load_cascade();
      }

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as0(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as1(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as2(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as3(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs(b_blocks);
      as0.seek((rb_base + 0) * kKBlocks);
      as1.seek((rb_base + 1) * kKBlocks);
      as2.seek((rb_base + 2) * kKBlocks);
      as3.seek((rb_base + 3) * kKBlocks);
      bs.seek(nb * kKBlocks);

      for (unsigned kb = 0; kb < kKBlocks; ++kb)
        chess_prepare_for_pipelining chess_loop_range(32, ) {
          const auto b_value = bs.pop();
          acc0 = mac_8x8_8x8T(as0.pop(), b_value, acc0);
          acc1 = mac_8x8_8x8T(as1.pop(), b_value, acc1);
          acc2 = mac_8x8_8x8T(as2.pop(), b_value, acc2);
          acc3 = mac_8x8_8x8T(as3.pop(), b_value, acc3);
        }

      if constexpr (role != Role::East) {
        put_cascade(acc0);
        put_cascade(acc1);
        put_cascade(acc2);
        put_cascade(acc3);
      } else {
        constexpr unsigned panel_tokens = kNBlocks / 2;
        const aie::accum<accfloat, 64> values[4] = {acc0, acc1, acc2, acc3};
        for (unsigned lane = 0; lane < 4; ++lane) {
          const unsigned rb = rb_base + lane;
          const unsigned macro_token =
              (rb / 2) * (C_PANEL_SLAB * panel_tokens) +
              c_panel * panel_tokens + (nb / 2);
          bfloat16 *__restrict output =
              c + macro_token * 256 + (rb % 2) * 128 + (nb % 2) * 8;
          store_8x8(output, values[lane]);
        }
      }
    }
  }
}

#if KERNEL_STORAGE == 0 || KERNEL_STORAGE == 3
template <Role role, bool forward>
static inline void run_q4(const uint8 *__restrict a,
                          const uint8 *__restrict packed,
                          bfloat16 *__restrict c) {
#if KERNEL_STORAGE == 0
  if constexpr (forward)
    forward_bytes(packed, Q4_TILE_BYTES);
#endif
  prepare_q4(packed);
#if KERNEL_STORAGE == 3
  if constexpr (forward)
    forward_bytes(reinterpret_cast<const uint8 *>(b_bfp), kBfpBytes);
#endif
  run_panel<role>(a, reinterpret_cast<const uint8 *>(b_bfp), c);
}

#endif
#if KERNEL_STORAGE == 4 || KERNEL_STORAGE == 5
template <Role role>
static inline void reduce_direct(aie::accum<accfloat, 64> local, unsigned rb,
                                 unsigned nb, bfloat16 *__restrict c,
                                 unsigned c_panel = 0) {
  auto accumulator = local;
  if constexpr (role != Role::West)
    accumulator = aie::add(load_cascade(), accumulator);

  if constexpr (role != Role::East) {
    put_cascade(accumulator);
  } else {
    constexpr unsigned panel_tokens = kNBlocks / 2;
    const unsigned macro_token =
        (rb / 2) * (C_PANEL_SLAB * panel_tokens) +
        c_panel * panel_tokens + (nb / 2);
    bfloat16 *__restrict output =
        c + macro_token * 256 + (rb % 2) * 128 + (nb % 2) * 8;
    store_8x8(output, accumulator);
  }
}

template <Role role, bool forward>
static inline void run_q4_direct(const uint8 *__restrict a,
                                 const uint8 *__restrict packed,
                                 bfloat16 *__restrict c,
                                 unsigned c_panel = 0) {
  if constexpr (forward)
    forward_bytes(packed, Q4_TILE_BYTES);

  auto *__restrict a_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(a));
  for (unsigned rb_base = 0; rb_base < kRowBlocks; rb_base += 4) {
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as0(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as1(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as2(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as3(a_blocks);
      as0.seek((rb_base + 0) * kKBlocks);
      as1.seek((rb_base + 1) * kKBlocks);
      as2.seek((rb_base + 2) * kKBlocks);
      as3.seek((rb_base + 3) * kKBlocks);

      auto acc0 = aie::zeros<accfloat, 64>();
      auto acc1 = aie::zeros<accfloat, 64>();
      auto acc2 = aie::zeros<accfloat, 64>();
      auto acc3 = aie::zeros<accfloat, 64>();

      for (unsigned group = 0; group < kGroups; ++group)
        chess_prepare_for_pipelining {
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
            const auto transposed =
                aie::transpose(affine.template to_vector<bfloat16>(), 8, 8);
            aie::accum<accfloat, 64> b_value;
            b_value = transposed;
            const auto b_block = b_value.template to_vector<bfp16ebs8>();
            acc0 = mac_8x8_8x8T(as0.pop(), b_block, acc0);
            acc1 = mac_8x8_8x8T(as1.pop(), b_block, acc1);
            acc2 = mac_8x8_8x8T(as2.pop(), b_block, acc2);
            acc3 = mac_8x8_8x8T(as3.pop(), b_block, acc3);
          }
        }

      reduce_direct<role>(acc0, rb_base + 0, nb, c, c_panel);
      reduce_direct<role>(acc1, rb_base + 1, nb, c, c_panel);
      reduce_direct<role>(acc2, rb_base + 2, nb, c, c_panel);
      reduce_direct<role>(acc3, rb_base + 3, nb, c, c_panel);
    }
  }
}

// Native-Q4 ceiling for the 64x64 tile.  The west/middle roles retain the
// four-row Q4 reuse schedule above.  East emits each 32-row band in the same
// [rb_group, N/8, row, lane] order described by the output ObjectFIFO's
// dimensionsFromStream descriptor, avoiding a full 8-KiB C tile in L1.
static inline void emit_q4_direct_east_stream(
    aie::accum<accfloat, 64> local, bool last_accumulator) {
  const auto result = aie::add(load_cascade(), local);
  const auto value = result.template to_vector<bfloat16>();
  for (unsigned row = 0; row < 8; ++row) {
    put_ms(value.template extract<8>(row).to_native(),
           last_accumulator && row == 7);
  }
}

static inline void run_q4_direct_east_stream(const uint8 *__restrict a,
                                             const uint8 *__restrict packed) {
  auto *__restrict a_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(a));
  for (unsigned rb_base = 0; rb_base < kRowBlocks; rb_base += 4) {
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as0(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as1(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as2(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as3(a_blocks);
      as0.seek((rb_base + 0) * kKBlocks);
      as1.seek((rb_base + 1) * kKBlocks);
      as2.seek((rb_base + 2) * kKBlocks);
      as3.seek((rb_base + 3) * kKBlocks);

      auto acc0 = aie::zeros<accfloat, 64>();
      auto acc1 = aie::zeros<accfloat, 64>();
      auto acc2 = aie::zeros<accfloat, 64>();
      auto acc3 = aie::zeros<accfloat, 64>();
      for (unsigned group = 0; group < kGroups; ++group)
        chess_prepare_for_pipelining {
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
            const auto transposed =
                aie::transpose(affine.template to_vector<bfloat16>(), 8, 8);
            aie::accum<accfloat, 64> b_value;
            b_value = transposed;
            const auto b_block = b_value.template to_vector<bfp16ebs8>();
            acc0 = mac_8x8_8x8T(as0.pop(), b_block, acc0);
            acc1 = mac_8x8_8x8T(as1.pop(), b_block, acc1);
            acc2 = mac_8x8_8x8T(as2.pop(), b_block, acc2);
            acc3 = mac_8x8_8x8T(as3.pop(), b_block, acc3);
          }
        }

      emit_q4_direct_east_stream(acc0, false);
      emit_q4_direct_east_stream(acc1, false);
      emit_q4_direct_east_stream(acc2, false);
      emit_q4_direct_east_stream(
          acc3, rb_base + 4 == kRowBlocks && nb + 1 == kNBlocks);
    }
  }
}
#endif

#if KERNEL_STORAGE == 5
alignas(aie::vector_decl_align) static uint8 q4_stream_scratch[Q4_TILE_BYTES];

template <bool forward>
static inline void read_q4_stream() {
  for (unsigned offset = 0; offset < Q4_TILE_BYTES; offset += 64) {
    const auto native = get_ss_v64uint8();
    const auto value = aie::vector<uint8, 64>(native);
    aie::store_v(q4_stream_scratch + offset, value);
    if constexpr (forward) {
      const bool last = offset + 64 == Q4_TILE_BYTES;
      put_ms(native, last);
    }
  }
}

template <Role role, bool forward>
static inline void run_q4_stream(const uint8 *__restrict a,
                                 bfloat16 *__restrict c) {
  for (unsigned panel = 0; panel < C_PANEL_SLAB; ++panel) {
    read_q4_stream<forward>();
    run_q4_direct<role, false>(a, q4_stream_scratch, c, panel);
  }
}
#endif

#if KERNEL_STORAGE == 7
#if DIM_M == 32 && DIM_N == 32
// This geometry can retain the complete expanded tile alongside stationary
// A.  It remains faster than the bounded direct-consume schedule because the
// proven mac_local loop sustains a substantially higher native BFP issue rate.
alignas(aie::vector_decl_align) static uint8 bfp_stream_scratch[kBfpBytes];

template <bool forward>
static inline void read_bfp_stream() {
  for (unsigned offset = 0; offset < kBfpBytes; offset += 64) {
    const auto native = get_ss_v64uint8();
    const auto value = aie::vector<uint8, 64>(native);
    aie::store_v(bfp_stream_scratch + offset, value);
    if constexpr (forward) {
      const bool last = offset + 64 == kBfpBytes;
      put_ms(native, last);
    }
  }
}

template <Role role, bool forward>
static inline void run_bfp_stream(const uint8 *__restrict a,
                                  bfloat16 *__restrict c) {
  for (unsigned panel = 0; panel < C_PANEL_SLAB; ++panel) {
    read_bfp_stream<forward>();
    run_panel<role>(a, bfp_stream_scratch, c, panel);
  }
}
#else
// Consume expanded weights in bounded K batches instead of materializing a
// complete Kstage x N BFP tile on every core.  A 16-block batch is 18 native
// 64-byte stream beats.  This is small enough to coexist with a stationary
// 64x512 activation shard and large enough to amortize the FP32 partial spill
// used to reuse each B block across all eight 8-row activation blocks.
alignas(aie::vector_decl_align) static bfp16ebs8
    bfp_receive_batch[kStreamBatchBlocks * 8];
#if DIM_M == 64
alignas(aie::vector_decl_align) static float
    bfp_stream_partials[kRowBlocks * 64];
#endif
static_assert(sizeof(bfp_receive_batch) == kStreamBatchBytes,
              "unexpected streamed BFP batch size");

template <bool forward>
static inline void read_bfp_batch(bool last_batch) {
  auto *__restrict bytes = reinterpret_cast<uint8 *>(bfp_receive_batch);
  for (unsigned offset = 0; offset < kStreamBatchBytes; offset += 64) {
    const auto native = get_ss_v64uint8();
    const auto value = aie::vector<uint8, 64>(native);
    aie::store_v(bytes + offset, value);
    if constexpr (forward) {
      const bool last = last_batch && offset + 64 == kStreamBatchBytes;
      put_ms(native, last);
    }
  }
}

template <Role role>
static inline void finish_bfp_stream_accumulator(
    aie::accum<accfloat, 64> local, unsigned rb, unsigned nb,
    bfloat16 *__restrict c, unsigned c_panel) {
  auto accumulator = local;
  if constexpr (role != Role::West)
    accumulator = aie::add(load_cascade(), accumulator);

  if constexpr (role != Role::East) {
    put_cascade(accumulator);
  } else {
    constexpr unsigned panel_tokens = kNBlocks / 2;
    const unsigned macro_token =
        (rb / 2) * (C_PANEL_SLAB * panel_tokens) +
        c_panel * panel_tokens + (nb / 2);
    bfloat16 *__restrict output =
        c + macro_token * 256 + (rb % 2) * 128 + (nb % 2) * 8;
    store_8x8(output, accumulator);
  }
}

#if DIM_M == 32
// Four 8-row activation blocks let all FP32 accumulators stay live while the
// BFP stream advances.  Every received B block is used by all four row blocks
// before the next block is consumed, so this path has neither a full B tile
// nor an intermediate-accumulator spill.
template <Role role, bool forward>
static inline void run_bfp_stream_m32_panel(const uint8 *__restrict a,
                                            bfloat16 *__restrict c,
                                            unsigned c_panel) {
  auto *__restrict a_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(a));
  for (unsigned nb = 0; nb < kNBlocks; ++nb) {
    auto acc0 = aie::zeros<accfloat, 64>();
    auto acc1 = aie::zeros<accfloat, 64>();
    auto acc2 = aie::zeros<accfloat, 64>();
    auto acc3 = aie::zeros<accfloat, 64>();

    for (unsigned kb_base = 0; kb_base < kKBlocks;
         kb_base += kStreamBatchBlocks) {
      const bool final_batch = kb_base + kStreamBatchBlocks == kKBlocks;
      read_bfp_batch<forward>(final_batch && nb + 1 == kNBlocks);

      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as0(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as1(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as2(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as3(a_blocks);
      aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs(
          bfp_receive_batch);
      as0.seek(0 * kKBlocks + kb_base);
      as1.seek(1 * kKBlocks + kb_base);
      as2.seek(2 * kKBlocks + kb_base);
      as3.seek(3 * kKBlocks + kb_base);
      bs.seek(0);

      for (unsigned inner = 0; inner < kStreamBatchBlocks; ++inner)
        chess_prepare_for_pipelining chess_loop_range(16, ) {
          const auto b_value = bs.pop();
          acc0 = mac_8x8_8x8T(as0.pop(), b_value, acc0);
          acc1 = mac_8x8_8x8T(as1.pop(), b_value, acc1);
          acc2 = mac_8x8_8x8T(as2.pop(), b_value, acc2);
          acc3 = mac_8x8_8x8T(as3.pop(), b_value, acc3);
        }
    }

    finish_bfp_stream_accumulator<role>(acc0, 0, nb, c, c_panel);
    finish_bfp_stream_accumulator<role>(acc1, 1, nb, c, c_panel);
    finish_bfp_stream_accumulator<role>(acc2, 2, nb, c, c_panel);
    finish_bfp_stream_accumulator<role>(acc3, 3, nb, c, c_panel);
  }
}
#endif

#if DIM_M == 64
template <Role role>
static inline void mac_bfp_receive_group(
    const uint8 *__restrict a, unsigned rb_base, unsigned nb,
    unsigned kb_base, bool first_batch, bool final_batch,
    bfloat16 *__restrict c, unsigned c_panel) {
  auto *__restrict a_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(a));

  aie::accum<accfloat, 64> acc0;
  aie::accum<accfloat, 64> acc1;
  aie::accum<accfloat, 64> acc2;
  aie::accum<accfloat, 64> acc3;
  if (first_batch) {
    acc0 = aie::zeros<accfloat, 64>();
    acc1 = aie::zeros<accfloat, 64>();
    acc2 = aie::zeros<accfloat, 64>();
    acc3 = aie::zeros<accfloat, 64>();
  } else {
    acc0 = aie::load_v<64>(bfp_stream_partials + (rb_base + 0) * 64);
    acc1 = aie::load_v<64>(bfp_stream_partials + (rb_base + 1) * 64);
    acc2 = aie::load_v<64>(bfp_stream_partials + (rb_base + 2) * 64);
    acc3 = aie::load_v<64>(bfp_stream_partials + (rb_base + 3) * 64);
  }

  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as0(a_blocks);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as1(a_blocks);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as2(a_blocks);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as3(a_blocks);
  aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs(
      bfp_receive_batch);
  as0.seek((rb_base + 0) * kKBlocks + kb_base);
  as1.seek((rb_base + 1) * kKBlocks + kb_base);
  as2.seek((rb_base + 2) * kKBlocks + kb_base);
  as3.seek((rb_base + 3) * kKBlocks + kb_base);
  bs.seek(0);

  for (unsigned inner = 0; inner < kStreamBatchBlocks; ++inner)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      const auto b_value = bs.pop();
      acc0 = mac_8x8_8x8T(as0.pop(), b_value, acc0);
      acc1 = mac_8x8_8x8T(as1.pop(), b_value, acc1);
      acc2 = mac_8x8_8x8T(as2.pop(), b_value, acc2);
      acc3 = mac_8x8_8x8T(as3.pop(), b_value, acc3);
    }

  if (!final_batch) {
    aie::store_v(bfp_stream_partials + (rb_base + 0) * 64,
                 acc0.template to_vector<float>());
    aie::store_v(bfp_stream_partials + (rb_base + 1) * 64,
                 acc1.template to_vector<float>());
    aie::store_v(bfp_stream_partials + (rb_base + 2) * 64,
                 acc2.template to_vector<float>());
    aie::store_v(bfp_stream_partials + (rb_base + 3) * 64,
                 acc3.template to_vector<float>());
    return;
  }

  finish_bfp_stream_accumulator<role>(acc0, rb_base + 0, nb, c, c_panel);
  finish_bfp_stream_accumulator<role>(acc1, rb_base + 1, nb, c, c_panel);
  finish_bfp_stream_accumulator<role>(acc2, rb_base + 2, nb, c, c_panel);
  finish_bfp_stream_accumulator<role>(acc3, rb_base + 3, nb, c, c_panel);
}
#endif

template <Role role, bool forward>
static inline void run_bfp_stream(const uint8 *__restrict a,
                                  bfloat16 *__restrict c) {
  for (unsigned panel = 0; panel < C_PANEL_SLAB; ++panel) {
#if DIM_M == 32
    run_bfp_stream_m32_panel<role, forward>(a, c, panel);
#else
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      for (unsigned kb_base = 0; kb_base < kKBlocks;
           kb_base += kStreamBatchBlocks) {
        const bool first_batch = kb_base == 0;
        const bool final_batch = kb_base + kStreamBatchBlocks == kKBlocks;
        const bool last_batch = final_batch && nb + 1 == kNBlocks;
        read_bfp_batch<forward>(last_batch);
        for (unsigned rb_base = 0; rb_base < kRowBlocks; rb_base += 4)
          mac_bfp_receive_group<role>(a, rb_base, nb, kb_base, first_batch,
                                      final_batch, c, panel);
      }
    }
#endif
  }
}
#endif
#endif

#if KERNEL_STORAGE == 8
static_assert(DIM_M == 32,
              "fused Q4 expand/multicast requires four 8-row blocks");
alignas(aie::vector_decl_align) static uint8
    q4_expand_stream_scratch[Q4_TILE_BYTES];
#if DIM_N == 64
constexpr unsigned kExpandBatchBlocks = 8;
constexpr unsigned kExpandBatchBytes = kExpandBatchBlocks * 64 * 9 / 8;
alignas(aie::vector_decl_align) static bfp16ebs8
    q4_expand_bfp_batch[kExpandBatchBlocks * 8];
static_assert(sizeof(q4_expand_bfp_batch) == kExpandBatchBytes,
              "unexpected Q4 expand batch size");
#endif

static inline void read_q4_expand_stream() {
  for (unsigned offset = 0; offset < Q4_TILE_BYTES; offset += 64) {
    const auto value = aie::vector<uint8, 64>(get_ss_v64uint8());
    aie::store_v(q4_expand_stream_scratch + offset, value);
  }
}

#if DIM_N == 32
template <Role role>
static inline void run_q4_expand_stream(const uint8 *__restrict a,
                                        bfloat16 *__restrict c) {
  for (unsigned panel = 0; panel < C_PANEL_SLAB; ++panel) {
    read_q4_expand_stream();
    prepare_q4(q4_expand_stream_scratch);
    forward_bytes(reinterpret_cast<const uint8 *>(b_bfp), kBfpBytes);
    run_panel<role>(a, reinterpret_cast<const uint8 *>(b_bfp), c, panel);
  }
}
#else
static inline void prepare_q4_expand_batch(unsigned nb, unsigned kb_base) {
  const unsigned group_base = kb_base / 4;
  for (unsigned group_offset = 0; group_offset < 2; ++group_offset) {
    const unsigned group = group_base + group_offset;
    const auto scale8 =
        aie::load_v<8>(scale_vector(q4_expand_stream_scratch, group, nb));
    const auto bias8 =
        aie::load_v<8>(bias_vector(q4_expand_stream_scratch, group, nb));
    const auto scale64 = scale8.template grow_replicate<64>();
    const auto bias64 = bias8.template grow_replicate<64>();
    for (unsigned inner = 0; inner < 4; ++inner) {
      const unsigned kb = group * 4 + inner;
      const unsigned block = kb * kNBlocks + nb;
      const auto qbf16 = aie::to_float<bfloat16>(
          unpack_q_microtile(q4_expand_stream_scratch, block));
      const auto affine = aie::sub(aie::mul(qbf16, scale64), bias64);
      const auto transposed =
          aie::transpose(affine.template to_vector<bfloat16>(), 8, 8);
      aie::accum<accfloat, 64> value;
      value = transposed;
      // seek() advances a stream cursor.  Recreate the buffer stream for each
      // block so the index is absolute, exactly as in the byte-proven full-
      // tile and eight-block encoders.
      aie::block_vector_output_buffer_stream<bfp16ebs8, 64> out(
          q4_expand_bfp_batch);
      out.seek(group_offset * 4 + inner);
      out.push(value.template to_vector<bfp16ebs8>());
    }
  }
}

template <Role role>
static inline void finish_q4_expand_accumulator(
    aie::accum<accfloat, 64> local, unsigned rb, unsigned nb,
    bfloat16 *__restrict c, unsigned c_panel) {
  auto accumulator = local;
  if constexpr (role != Role::West)
    accumulator = aie::add(load_cascade(), accumulator);

  if constexpr (role != Role::East) {
    put_cascade(accumulator);
  } else {
    constexpr unsigned panel_tokens = kNBlocks / 2;
    const unsigned macro_token =
        (rb / 2) * (C_PANEL_SLAB * panel_tokens) +
        c_panel * panel_tokens + (nb / 2);
    bfloat16 *__restrict output =
        c + macro_token * 256 + (rb % 2) * 128 + (nb % 2) * 8;
    store_8x8(output, accumulator);
  }
}

template <Role role>
static inline void run_q4_expand_stream(const uint8 *__restrict a,
                                        bfloat16 *__restrict c) {
  auto *__restrict a_blocks =
      reinterpret_cast<bfp16ebs8 *>(const_cast<uint8 *>(a));
  for (unsigned panel = 0; panel < C_PANEL_SLAB; ++panel) {
    read_q4_expand_stream();
    for (unsigned nb = 0; nb < kNBlocks; ++nb) {
      auto acc0 = aie::zeros<accfloat, 64>();
      auto acc1 = aie::zeros<accfloat, 64>();
      auto acc2 = aie::zeros<accfloat, 64>();
      auto acc3 = aie::zeros<accfloat, 64>();

      for (unsigned kb_base = 0; kb_base < kKBlocks;
           kb_base += kExpandBatchBlocks) {
        const bool final_batch = kb_base + kExpandBatchBlocks == kKBlocks;
        prepare_q4_expand_batch(nb, kb_base);
        forward_bytes(reinterpret_cast<const uint8 *>(q4_expand_bfp_batch),
                      kExpandBatchBytes,
                      final_batch && nb + 1 == kNBlocks);

        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as0(a_blocks);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as1(a_blocks);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as2(a_blocks);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> as3(a_blocks);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> bs(
            q4_expand_bfp_batch);
        as0.seek(0 * kKBlocks + kb_base);
        as1.seek(1 * kKBlocks + kb_base);
        as2.seek(2 * kKBlocks + kb_base);
        as3.seek(3 * kKBlocks + kb_base);
        bs.seek(0);

        for (unsigned inner = 0; inner < kExpandBatchBlocks; ++inner)
          chess_prepare_for_pipelining chess_loop_range(8, ) {
            const auto b_value = bs.pop();
            acc0 = mac_8x8_8x8T(as0.pop(), b_value, acc0);
            acc1 = mac_8x8_8x8T(as1.pop(), b_value, acc1);
            acc2 = mac_8x8_8x8T(as2.pop(), b_value, acc2);
            acc3 = mac_8x8_8x8T(as3.pop(), b_value, acc3);
          }
      }

      finish_q4_expand_accumulator<role>(acc0, 0, nb, c, panel);
      finish_q4_expand_accumulator<role>(acc1, 1, nb, c, panel);
      finish_q4_expand_accumulator<role>(acc2, 2, nb, c, panel);
      finish_q4_expand_accumulator<role>(acc3, 3, nb, c, panel);
    }
  }
}
#endif
#endif

template <Role role, bool forward>
static inline void run_bfp(const uint8 *__restrict a, const uint8 *__restrict b,
                           bfloat16 *__restrict c,
                           unsigned c_panel = 0) {
  if constexpr (forward)
    forward_bytes(b, kBfpBytes);
  run_panel<role>(a, b, c, c_panel);
}

template <Role role, bool forward>
static inline void run_bfp_resident(const uint8 *__restrict a,
                                    const uint8 *__restrict b,
                                    bfloat16 *__restrict c) {
  if constexpr (forward)
    forward_bytes(b, kBfpBytes);
  for (unsigned panel = 0; panel < N_PANELS; ++panel)
    chess_prepare_for_pipelining {
      // This ceiling deliberately reuses one B tile. East overwrites the same
      // C tile; only the final iteration is transferred to the host.
      run_panel<role>(a, b, c);
    }
}

} // namespace

extern "C" {

#if KERNEL_STORAGE != 3 && KERNEL_STORAGE != 9 &&                           \
    (KERNEL_ROLE == 0 || KERNEL_ROLE == 4)
void q4ks_systolic_convert_a(bfloat16 *input, uint8 *output, int chunk) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  convert_a_chunk(input, output, static_cast<unsigned>(chunk));
  aie::set_rounding(saved);
}

void q4ks_systolic_copy_a(bfloat16 *input, bfloat16 *output) {
  for (unsigned offset = 0; offset < DIM_M * A_CHUNK_K; offset += 32) {
    const auto value = aie::load_v<32>(input + offset);
    aie::store_v(output + offset, value);
  }
}
#endif

#if KERNEL_STORAGE == 9 && (KERNEL_ROLE == 0 || KERNEL_ROLE == 4)
void q4ks_systolic_expand_q4_to_bfp_stream(uint8 *packed, int last_panel) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  stream_q4_as_bfp(packed, last_panel != 0);
  aie::set_rounding(saved);
}
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 1
#define DEFINE_WEST(NAME, PREP, FORWARD)                                       \
  void NAME(uint8 *a, uint8 *b) {                                              \
    const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);      \
    PREP<Role::West, FORWARD>(a, b, nullptr);                                  \
    aie::set_rounding(saved);                                                  \
  }
#else
#define DEFINE_WEST(NAME, PREP, FORWARD)
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 2
#define DEFINE_MIDDLE(NAME, PREP, FORWARD)                                     \
  void NAME(uint8 *a, uint8 *b) {                                              \
    const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);      \
    PREP<Role::Middle, FORWARD>(a, b, nullptr);                                \
    aie::set_rounding(saved);                                                  \
  }
#else
#define DEFINE_MIDDLE(NAME, PREP, FORWARD)
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 3
#define DEFINE_EAST(NAME, PREP, FORWARD)                                       \
  void NAME(uint8 *a, uint8 *b, bfloat16 *c) {                                 \
    const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);      \
    PREP<Role::East, FORWARD>(a, b, c);                                        \
    aie::set_rounding(saved);                                                  \
  }
#else
#define DEFINE_EAST(NAME, PREP, FORWARD)
#endif

#if KERNEL_STORAGE == 0
DEFINE_WEST(q4ks_systolic_west_q4, run_q4, false)
DEFINE_WEST(q4ks_systolic_west_q4_forward, run_q4, true)
DEFINE_MIDDLE(q4ks_systolic_middle_q4, run_q4, false)
DEFINE_MIDDLE(q4ks_systolic_middle_q4_forward, run_q4, true)
DEFINE_EAST(q4ks_systolic_east_q4, run_q4, false)
DEFINE_EAST(q4ks_systolic_east_q4_forward, run_q4, true)
#elif KERNEL_STORAGE == 1
DEFINE_WEST(q4ks_systolic_west_bfp, run_bfp, false)
DEFINE_WEST(q4ks_systolic_west_bfp_forward, run_bfp, true)
DEFINE_MIDDLE(q4ks_systolic_middle_bfp, run_bfp, false)
DEFINE_MIDDLE(q4ks_systolic_middle_bfp_forward, run_bfp, true)
#if C_PANEL_SLAB == 1
DEFINE_EAST(q4ks_systolic_east_bfp, run_bfp, false)
DEFINE_EAST(q4ks_systolic_east_bfp_forward, run_bfp, true)
#elif KERNEL_ROLE == 0 || KERNEL_ROLE == 3
void q4ks_systolic_east_bfp_slab(uint8 *a, uint8 *b, bfloat16 *c,
                                 int panel) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp<Role::East, false>(a, b, c, static_cast<unsigned>(panel));
  aie::set_rounding(saved);
}

void q4ks_systolic_east_bfp_forward_slab(uint8 *a, uint8 *b, bfloat16 *c,
                                         int panel) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp<Role::East, true>(a, b, c, static_cast<unsigned>(panel));
  aie::set_rounding(saved);
}
#endif
#if KERNEL_ROLE == 0 || KERNEL_ROLE == 3
void q4ks_systolic_east_bfp_stream_c(uint8 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_panel_east_stream(a, b);
  aie::set_rounding(saved);
}
#endif
#elif KERNEL_STORAGE == 2
DEFINE_WEST(q4ks_systolic_west_resident, run_bfp_resident, false)
DEFINE_WEST(q4ks_systolic_west_resident_forward, run_bfp_resident, true)
DEFINE_MIDDLE(q4ks_systolic_middle_resident, run_bfp_resident, false)
DEFINE_MIDDLE(q4ks_systolic_middle_resident_forward, run_bfp_resident, true)
DEFINE_EAST(q4ks_systolic_east_resident, run_bfp_resident, false)
DEFINE_EAST(q4ks_systolic_east_resident_forward, run_bfp_resident, true)
#elif KERNEL_STORAGE == 3
DEFINE_WEST(q4ks_systolic_west_q4expand, run_q4, false)
DEFINE_WEST(q4ks_systolic_west_q4expand_forward, run_q4, true)
DEFINE_MIDDLE(q4ks_systolic_middle_q4expand, run_q4, false)
DEFINE_MIDDLE(q4ks_systolic_middle_q4expand_forward, run_q4, true)
DEFINE_EAST(q4ks_systolic_east_q4expand, run_q4, false)
DEFINE_EAST(q4ks_systolic_east_q4expand_forward, run_q4, true)
#elif KERNEL_STORAGE == 4
#if DIRECT_C_STREAM
DEFINE_WEST(q4ks_systolic_west_q4direct, run_q4_direct, false)
DEFINE_MIDDLE(q4ks_systolic_middle_q4direct, run_q4_direct, false)
#if KERNEL_ROLE == 0 || KERNEL_ROLE == 3
void q4ks_systolic_east_q4direct_stream_c(uint8 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_direct_east_stream(a, b);
  aie::set_rounding(saved);
}
#endif
#else
DEFINE_WEST(q4ks_systolic_west_q4direct, run_q4_direct, false)
DEFINE_WEST(q4ks_systolic_west_q4direct_forward, run_q4_direct, true)
DEFINE_MIDDLE(q4ks_systolic_middle_q4direct, run_q4_direct, false)
DEFINE_MIDDLE(q4ks_systolic_middle_q4direct_forward, run_q4_direct, true)
#if C_PANEL_SLAB == 1
DEFINE_EAST(q4ks_systolic_east_q4direct, run_q4_direct, false)
DEFINE_EAST(q4ks_systolic_east_q4direct_forward, run_q4_direct, true)
#elif KERNEL_ROLE == 0 || KERNEL_ROLE == 3
void q4ks_systolic_east_q4direct_slab(uint8 *a, uint8 *b, bfloat16 *c,
                                      int panel) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_direct<Role::East, false>(a, b, c, static_cast<unsigned>(panel));
  aie::set_rounding(saved);
}

void q4ks_systolic_east_q4direct_forward_slab(uint8 *a, uint8 *b,
                                              bfloat16 *c, int panel) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_direct<Role::East, true>(a, b, c, static_cast<unsigned>(panel));
  aie::set_rounding(saved);
}
#endif
#if KERNEL_ROLE == 0 || KERNEL_ROLE == 3
void q4ks_systolic_east_q4direct_stream_c(uint8 *a, uint8 *b) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_direct_east_stream(a, b);
  aie::set_rounding(saved);
}
#endif
#endif
#elif KERNEL_STORAGE == 5
#if KERNEL_ROLE == 0 || KERNEL_ROLE == 1
void q4ks_systolic_west_q4stream(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_stream<Role::West, false>(a, nullptr);
  aie::set_rounding(saved);
}

void q4ks_systolic_west_q4stream_forward(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_stream<Role::West, true>(a, nullptr);
  aie::set_rounding(saved);
}
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 2
void q4ks_systolic_middle_q4stream(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_stream<Role::Middle, false>(a, nullptr);
  aie::set_rounding(saved);
}

void q4ks_systolic_middle_q4stream_forward(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_stream<Role::Middle, true>(a, nullptr);
  aie::set_rounding(saved);
}
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 3
void q4ks_systolic_east_q4stream_slab(uint8 *a, bfloat16 *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_stream<Role::East, false>(a, c);
  aie::set_rounding(saved);
}

void q4ks_systolic_east_q4stream_forward_slab(uint8 *a, bfloat16 *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_stream<Role::East, true>(a, c);
  aie::set_rounding(saved);
}
#endif
#elif KERNEL_STORAGE == 7
#if KERNEL_ROLE == 0 || KERNEL_ROLE == 1
void q4ks_systolic_west_bfpstream(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp_stream<Role::West, false>(a, nullptr);
  aie::set_rounding(saved);
}

void q4ks_systolic_west_bfpstream_forward(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp_stream<Role::West, true>(a, nullptr);
  aie::set_rounding(saved);
}
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 2
void q4ks_systolic_middle_bfpstream(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp_stream<Role::Middle, false>(a, nullptr);
  aie::set_rounding(saved);
}

void q4ks_systolic_middle_bfpstream_forward(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp_stream<Role::Middle, true>(a, nullptr);
  aie::set_rounding(saved);
}
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 3
#if C_PANEL_SLAB == 1
void q4ks_systolic_east_bfpstream(uint8 *a, bfloat16 *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp_stream<Role::East, false>(a, c);
  aie::set_rounding(saved);
}

void q4ks_systolic_east_bfpstream_forward(uint8 *a, bfloat16 *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp_stream<Role::East, true>(a, c);
  aie::set_rounding(saved);
}
#else
void q4ks_systolic_east_bfpstream_slab(uint8 *a, bfloat16 *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp_stream<Role::East, false>(a, c);
  aie::set_rounding(saved);
}

void q4ks_systolic_east_bfpstream_forward_slab(uint8 *a, bfloat16 *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_bfp_stream<Role::East, true>(a, c);
  aie::set_rounding(saved);
}
#endif
#endif
#elif KERNEL_STORAGE == 8
#if KERNEL_ROLE == 0 || KERNEL_ROLE == 1
void q4ks_systolic_west_q4expandstream(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_expand_stream<Role::West>(a, nullptr);
  aie::set_rounding(saved);
}

void q4ks_systolic_west_q4expandstream_forward(uint8 *a) {
  q4ks_systolic_west_q4expandstream(a);
}
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 2
void q4ks_systolic_middle_q4expandstream(uint8 *a) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_expand_stream<Role::Middle>(a, nullptr);
  aie::set_rounding(saved);
}

void q4ks_systolic_middle_q4expandstream_forward(uint8 *a) {
  q4ks_systolic_middle_q4expandstream(a);
}
#endif

#if KERNEL_ROLE == 0 || KERNEL_ROLE == 3
void q4ks_systolic_east_q4expandstream_slab(uint8 *a, bfloat16 *c) {
  const auto saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  run_q4_expand_stream<Role::East>(a, c);
  aie::set_rounding(saved);
}

void q4ks_systolic_east_q4expandstream_forward_slab(uint8 *a, bfloat16 *c) {
  q4ks_systolic_east_q4expandstream_slab(a, c);
}
#endif
#elif KERNEL_STORAGE == 6 || KERNEL_STORAGE == 9
// Helper/expander-only objects; their entry points are emitted above.
#else
#error "unsupported KERNEL_STORAGE"
#endif

#undef DEFINE_WEST
#undef DEFINE_MIDDLE
#undef DEFINE_EAST

} // extern "C"
