# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Capacity and hardware sweep for Q4_K compute/cache candidates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packing import CACHE_MODES, COMPUTE_TYPES, Q4KSConfig  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--hardware", action="store_true")
    result.add_argument("-M", type=int, default=1024)
    result.add_argument("-K", type=int, default=1024)
    result.add_argument("-N", type=int, default=2048)
    result.add_argument("--columns", type=int, default=8, choices=[1, 2, 4, 8])
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--iters", type=int, default=1)
    result.add_argument("--rounds", type=int, default=1)
    result.add_argument("--min-gflops", type=float, default=0.0)
    result.add_argument("--full-verify", action="store_true")
    result.add_argument("--compute-types", nargs="+", choices=COMPUTE_TYPES)
    result.add_argument("--cache-modes", nargs="+", choices=CACHE_MODES)
    result.add_argument("--output-prefix", type=Path)
    return result


def candidates(opts) -> list[dict]:
    symmetric = dict(m_c=64, m_a=32, k=128, n=64)
    asymmetric = dict(m_c=128, m_a=32, k=64, n=128)
    use_asymmetric = (
        opts.M % (asymmetric["m_c"] * 8) == 0
        and opts.N % (asymmetric["n"] * opts.columns) == 0
    )
    atb = asymmetric if use_asymmetric else symmetric
    result = [
        dict(name="bf16-stream-symmetric", compute_type="bf16", cache_mode="stream", **symmetric),
        dict(name="bf16-l1-symmetric", compute_type="bf16", cache_mode="l1-weight", **symmetric),
        dict(name="bfp16-l1-atb", compute_type="bfp16", cache_mode="l1-weight", **atb),
        dict(name="int8-l1-atb", compute_type="int8", cache_mode="l1-weight", **atb),
        dict(name="bf16-memtile-q4", compute_type="bf16", cache_mode="memtile-weight", **symmetric),
        dict(
            name="bfp16-memtile-q4",
            compute_type="bfp16",
            cache_mode="memtile-weight",
            **symmetric,
        ),
        dict(name="int8-memtile-q4", compute_type="int8", cache_mode="memtile-weight", **symmetric),
        dict(
            name="bfp16-preconverted-ceiling",
            compute_type="bfp16",
            cache_mode="l1-weight",
            activation_input="bfp16",
            **atb,
        ),
        dict(
            name="int8-preconverted-ceiling",
            compute_type="int8",
            cache_mode="l1-weight",
            activation_input="int8",
            **atb,
        ),
        dict(
            name="bfp16-activation-cache-model",
            compute_type="bfp16",
            cache_mode="memtile-activation",
            **symmetric,
        ),
        dict(
            name="int8-activation-cache-model",
            compute_type="int8",
            cache_mode="memtile-activation",
            **symmetric,
        ),
        dict(
            name="bfp16-joint-slab-model",
            compute_type="bfp16",
            cache_mode="joint-slab",
            **symmetric,
        ),
        dict(
            name="int8-joint-slab-model",
            compute_type="int8",
            cache_mode="joint-slab",
            **symmetric,
        ),
    ]
    if opts.compute_types:
        result = [row for row in result if row["compute_type"] in opts.compute_types]
    if opts.cache_modes:
        result = [row for row in result if row["cache_mode"] in opts.cache_modes]
    return result


def largest_joint_slab(opts, candidate) -> int | None:
    value = opts.K
    while value >= 256:
        try:
            Q4KSConfig(
                M=opts.M,
                K=opts.K,
                N=opts.N,
                n_aie_cols=opts.columns,
                activation_input="bf16",
                cache_k=value,
                **{
                    key: candidate[key]
                    for key in (
                        "m_c",
                        "m_a",
                        "k",
                        "n",
                        "compute_type",
                        "cache_mode",
                    )
                },
            )
            return value
        except ValueError:
            value //= 2
    return None


def config_for(opts, candidate) -> Q4KSConfig:
    cache_k = opts.K
    if candidate["cache_mode"] == "joint-slab":
        cache_k = largest_joint_slab(opts, candidate) or opts.K
    return Q4KSConfig(
        M=opts.M,
        K=opts.K,
        N=opts.N,
        n_aie_cols=opts.columns,
        activation_input=candidate.get("activation_input", "bf16"),
        cache_k=cache_k,
        **{key: candidate[key] for key in ("m_c", "m_a", "k", "n", "compute_type", "cache_mode")},
    )


def run_candidate(opts, candidate, cfg, result_path: Path) -> dict:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "whole_array.py"),
        "-M", str(cfg.M), "-K", str(cfg.K), "-N", str(cfg.N),
        "--m-c", str(cfg.m_c), "--m-a", str(cfg.m_a),
        "-k", str(cfg.k), "-n", str(cfg.n),
        "--n-aie-cols", str(cfg.n_aie_cols),
        "--compute-type", cfg.compute_type,
        "--cache-mode", cfg.cache_mode,
        "--activation-input", cfg.activation_input,
        "--cache-k", str(cfg.cache_k),
        "--warmup", str(opts.warmup),
        "--iters", str(opts.iters),
        "--benchmark-repeats", str(opts.rounds),
        "--verify-mode", "full" if opts.full_verify else "sampled",
        "--benchmark-json", str(result_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        return {
            "status": "failed",
            "error": (completed.stderr or completed.stdout)[-2000:],
        }
    row = json.loads(result_path.read_text())
    return {
        "status": "passed",
        "conversion_npu_us": None,
        "dma_npu_us": None,
        "total_npu_us": row["average_npu_us"],
        "throughput_gflops": row["median_gflops"],
        "max_abs": row["max_abs"],
        "max_rel": row["max_rel"],
        "nrmse": row["nrmse"],
    }


def main() -> None:
    opts = parser().parse_args()
    if opts.warmup < 0 or opts.iters < 1 or opts.rounds < 1 or opts.min_gflops < 0:
        raise SystemExit("warmup/gate must be non-negative; iters/rounds must be positive")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = (
        opts.output_prefix
        or Path(__file__).resolve().parent / "build" / f"benchmark_q4ks_{stamp}"
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for candidate in candidates(opts):
        base = dict(candidate)
        base.setdefault("activation_input", "bf16")
        try:
            cfg = config_for(opts, candidate)
            base.update(
                cache_k=cfg.cache_k,
                core_bytes=cfg.core_memory_bytes,
                memtile_bytes=sum(cfg.memtile_components().values()),
            )
        except ValueError as error:
            base.update(status="rejected", error=str(error))
            rows.append(base)
            print(f"{candidate['name']}: rejected: {error}")
            continue
        executable = (
            cfg.cache_mode in ("stream", "l1-weight", "memtile-weight")
            and cfg.activation_input == "bf16"
        )
        if not opts.hardware or not executable:
            base.update(
                status="capacity-ok" if executable else "model-only",
                conversion_npu_us=None,
                dma_npu_us=None,
                total_npu_us=None,
                throughput_gflops=None,
                max_abs=None,
                max_rel=None,
                nrmse=None,
            )
        else:
            temporary = prefix.parent / f"{prefix.name}_{candidate['name']}.json"
            base.update(run_candidate(opts, candidate, cfg, temporary))
        rows.append(base)
        print(f"{candidate['name']}: {base['status']}")

    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    qualifying = [
        row
        for row in rows
        if row["status"] == "passed"
        and row.get("throughput_gflops") is not None
        and row["throughput_gflops"] >= opts.min_gflops
    ]
    if opts.hardware:
        if not qualifying:
            raise SystemExit(
                f"no correct candidate reached {opts.min_gflops:.2f} GFLOP/s; "
                f"results: {json_path}"
            )
        fastest = max(qualifying, key=lambda row: row["throughput_gflops"])
        selected = prefix.parent / "selected_default.json"
        selected.write_text(json.dumps(fastest, indent=2) + "\n")
        print(
            f"selected {fastest['name']}: {fastest['throughput_gflops']:.2f} "
            f"GFLOP/s ({selected})"
        )
    print(f"wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
