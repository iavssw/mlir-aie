# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from packing import Q4KSConfig, make_deterministic_native_q4_k, prepare_q4ks_weights


def main():
    executable = Path(sys.argv[1]).resolve()
    cfg = Q4KSConfig(M=512, K=256, N=512, n_aie_cols=8)
    _, native = make_deterministic_native_q4_k(cfg)
    expected = prepare_q4ks_weights(native, cfg)
    with tempfile.TemporaryDirectory() as directory:
        native_path = Path(directory) / "native.bin"
        prepared_path = Path(directory) / "prepared.bin"
        native.tofile(native_path)
        completed = subprocess.run(
            [
                str(executable),
                "-M", "512", "-K", "256", "-N", "512",
                "--tile-m-c", "64", "--tile-m-a", "32",
                "--tile-k", "128", "--tile-n", "64",
                "--n-aie-cols", "8", "--cache-k", "256",
                "--q4-k-file", str(native_path),
                "--prepare-output", str(prepared_path),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        if "PASS!" not in completed.stdout:
            raise AssertionError(completed.stdout)
        actual = np.fromfile(prepared_path, dtype=np.uint8)
        np.testing.assert_array_equal(actual, expected)

        generated_path = Path(directory) / "generated.bin"
        generated = subprocess.run(
            [
                str(executable),
                "-M", "512", "-K", "256", "-N", "512",
                "--tile-m-c", "64", "--tile-m-a", "32",
                "--tile-k", "128", "--tile-n", "64",
                "--n-aie-cols", "8", "--cache-k", "256",
                "--prepare-output", str(generated_path), "-v", "1",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        match = re.search(
            r"A row 0 mean=([^,]+), RMS=([^;]+); .* mean=([^,]+), L2=([^\n]+)",
            generated.stdout,
        )
        if not match:
            raise AssertionError(generated.stdout)
        a_mean, a_rms, b_mean, b_l2 = map(float, match.groups())
        assert abs(a_mean) < 0.002
        assert abs(a_rms - 1.0) < 0.003
        assert abs(b_mean) < 0.004
        assert abs(b_l2 - 1.0) < 0.04
    print("PASS!")


if __name__ == "__main__":
    main()
