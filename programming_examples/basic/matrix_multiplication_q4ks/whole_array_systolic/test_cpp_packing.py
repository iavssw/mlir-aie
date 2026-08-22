# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check that the local C++ and Python model-load packers are byte-identical."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from packing import Q4KSConfig, make_deterministic_native_q4_k
from systolic_packing import SystolicConfig, prepare_systolic_q4ks_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=Path, required=True)
    opts = parser.parse_args()
    host = opts.host.resolve()

    source = Q4KSConfig(
        M=256,
        K=2048,
        N=256,
        m_c=32,
        m_a=16,
        k=256,
        n=16,
        n_aie_cols=8,
        compute_type="bfp16",
        accumulation_mode="bf16",
        cache_mode="l1-weight",
    )
    _, native = make_deterministic_native_q4_k(source, seed=0x53595354)

    with tempfile.TemporaryDirectory(prefix="q4ks-systolic-packing-") as temp:
        temp = Path(temp)
        native_path = temp / "native.bin"
        native.tofile(native_path)
        for flow, n in (
            ("q4-local", 32),
            ("q4-direct", 64),
            ("bfp-prepared", 32),
        ):
            config = SystolicConfig(
                M=256,
                K=2048,
                N=256,
                n=n,
                weight_flow=flow,
            )
            expected = prepare_systolic_q4ks_weights(native, config)
            output = temp / f"{flow}.bin"
            subprocess.run(
                [
                    str(host),
                    "-M",
                    "256",
                    "-K",
                    "2048",
                    "-N",
                    "256",
                    "--tile-m-c",
                    "32",
                    "--tile-m-a",
                    "32",
                    "--tile-k",
                    "256",
                    "--tile-n",
                    str(n),
                    "--n-aie-cols",
                    "8",
                    "--compute-type",
                    "bfp16",
                    "--accumulation-mode",
                    "cascade",
                    "--cache-mode",
                    flow,
                    "--cache-k",
                    "2048",
                    "--q4-k-file",
                    str(native_path),
                    "--prepare-output",
                    str(output),
                ],
                check=True,
            )
            actual = np.fromfile(output, np.uint8)
            np.testing.assert_array_equal(actual, expected)
            print(f"{flow}: {actual.size} identical bytes")
    print("PASS!")


if __name__ == "__main__":
    main()
