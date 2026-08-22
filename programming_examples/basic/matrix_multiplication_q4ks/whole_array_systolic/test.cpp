//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../common.h"
#include "systolic_common.h"

// Reuse the current Q4_K XRT harness and replace only its layout, packing, and
// reference namespace. The parent include guard prevents this macro from
// rewriting the original q4ks utilities themselves.
#define q4ks q4ks_systolic
#include "../test.cpp"
#undef q4ks
