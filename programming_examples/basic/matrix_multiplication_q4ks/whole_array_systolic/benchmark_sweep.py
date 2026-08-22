#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run reproducible Q4_K systolic candidate sweeps and merge result tables."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Candidate:
    label: str
    flow: str
    transport: str
    n: int
    eligible: bool
    note: str = ""
    m_a: int = 32
    panel_slab: int = 1
    c_panel_slab: int = 1


def candidates(preset: str) -> list[Candidate]:
    native = Candidate(
        "native-q4-column-cache-n32",
        "q4-expand-once",
        "q4-column-dequant-cache",
        32,
        True,
        "Selected native-Q4 path: dequantize each Q4 slab once per column, "
        "cache expanded BFP in MemTile, and multicast it to four rows",
        panel_slab=16,
        c_panel_slab=4,
    )
    legacy_native = Candidate(
        "native-q4-local-n32",
        "q4-local",
        "core-stream",
        32,
        True,
        "Legacy four-row local-dequant baseline",
    )
    direct32 = Candidate(
        "direct-q4-n32",
        "q4-direct",
        "core-stream",
        32,
        True,
        "Direct Q4-to-BFP operand conversion",
    )
    direct64 = Candidate(
        "direct-q4-n64",
        "q4-direct",
        "core-stream",
        64,
        True,
        "Direct Q4-to-BFP conversion with half as many N panels",
    )
    direct_m64 = Candidate(
        "direct-q4-m64-n32",
        "q4-direct",
        "core-stream",
        32,
        True,
        "64 stationary rows halve M waves and repeated weight traffic",
        m_a=64,
    )
    direct_m64_c2 = Candidate(
        "direct-q4-m64-c2-n32",
        "q4-direct",
        "core-stream",
        32,
        True,
        "64 stationary rows with two C panels per MemTile/L3 drain",
        m_a=64,
        c_panel_slab=2,
    )
    direct_c8 = Candidate(
        "direct-q4-c8-n32",
        "q4-direct",
        "core-stream",
        32,
        True,
        "Eight C panels share one 16-KiB core/MemTile/L3 transfer",
        c_panel_slab=8,
    )
    direct_p4_c8 = Candidate(
        "direct-q4-p4-c8-n32",
        "q4-direct",
        "core-stream",
        32,
        True,
        "Four-panel Q4 input slabs plus eight-panel C output slabs",
        panel_slab=4,
        c_panel_slab=8,
    )
    direct_shim_p2_c8 = Candidate(
        "direct-q4-shim-p2-c8-n32",
        "q4-direct",
        "shim-stream",
        32,
        True,
        "Direct 20-KiB shim ingress avoids the MemTile B split copy",
        panel_slab=2,
        c_panel_slab=8,
    )
    if preset == "native":
        return [native]
    if preset == "ceiling":
        return [
            Candidate(
                "resident-n32",
                "resident",
                "core-stream",
                32,
                False,
                "Ineligible stationary A/B compute ceiling",
            )
        ]
    if preset == "primitives":
        return [
            Candidate("resident-n32", "resident", "core-stream", 32, False),
            Candidate("resident-n16", "resident", "core-stream", 16, False),
            Candidate("bfp-flow-n32", "bfp-prepared", "core-stream", 32, False),
        ]
    if preset == "correctness":
        return [
            native,
            legacy_native,
            direct32,
            direct64,
            direct_m64,
            direct_m64_c2,
            direct_c8,
            direct_p4_c8,
            direct_shim_p2_c8,
            Candidate("native-q4-n16", "q4-local", "core-stream", 16, True),
            Candidate(
                "expand-once-n32",
                "q4-expand-once",
                "core-stream",
                32,
                True,
            ),
            Candidate(
                "prepared-bfp-n32",
                "bfp-prepared",
                "core-stream",
                32,
                False,
            ),
        ]
    return [
        native,
        legacy_native,
        direct32,
        direct64,
        direct_m64,
        direct_m64_c2,
        direct_c8,
        direct_p4_c8,
        direct_shim_p2_c8,
        Candidate("native-q4-n16", "q4-local", "core-stream", 16, True),
        Candidate(
            "expand-once-n32", "q4-expand-once", "core-stream", 32, True
        ),
        Candidate(
            "prepared-bfp-n32", "bfp-prepared", "core-stream", 32, False
        ),
        Candidate("resident-n32", "resident", "core-stream", 32, False),
        Candidate(
            "tile-dma-n32",
            "q4-local",
            "tile-dma",
            32,
            False,
            "Compiles, but same-offset MemTile split returned ERT abort; "
            "compute-tile distribute is rejected by the AIE verifier",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=("correctness", "native", "ceiling", "primitives", "sweep"),
        default="sweep",
    )
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--full-verify", action="store_true")
    parser.add_argument("--verify-samples", type=int, default=1000)
    parser.add_argument("--min-gflops", type=float, default=0.0)
    parser.add_argument("--power-mode", default="unknown")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--include-unsafe-tile-dma", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def device_identity() -> str:
    try:
        completed = subprocess.run(
            ["xrt-smi", "examine"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    for line in completed.stdout.splitlines():
        if "NPU Firmware Version" in line:
            return line.split(":", 1)[-1].strip()
    return "NPU2 (firmware not parsed)"


def run_candidate(
    candidate: Candidate, args: argparse.Namespace, dimensions: tuple[int, int, int]
) -> dict:
    M, K, N = dimensions
    if candidate.transport == "tile-dma" and not args.include_unsafe_tile_dma:
        return {
            "label": candidate.label,
            "variant": candidate.flow,
            "transport": candidate.transport,
            "n": candidate.n,
            "m_a": candidate.m_a,
            "panel_slab": candidate.panel_slab,
            "c_panel_slab": candidate.c_panel_slab,
            "eligible": False,
            "status": "known-runtime-abort-not-rerun",
            "note": candidate.note,
        }

    result_path = args.results_dir / f"{candidate.label}.json"
    verify_mode = (
        "none"
        if candidate.flow == "resident"
        else "full"
        if args.full_verify
        else "sampled"
        if args.preset == "correctness"
        else "none"
    )
    command = [
        sys.executable,
        str(HERE / "whole_array.py"),
        "-M",
        str(M),
        "-K",
        str(K),
        "-N",
        str(N),
        "--n-tile",
        str(candidate.n),
        "--m-tile",
        str(candidate.m_a),
        "--weight-flow",
        candidate.flow,
        "--transport",
        candidate.transport,
        "--panel-slab",
        str(candidate.panel_slab),
        "--c-panel-slab",
        str(candidate.c_panel_slab),
        "--verify-mode",
        verify_mode,
        "--verify-samples",
        str(args.verify_samples),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--benchmark-repeats",
        str(args.rounds),
        "--benchmark-json",
        str(result_path),
    ]
    if args.min_gflops:
        command.extend(["--min-gflops", str(args.min_gflops)])

    print("\n$", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if result_path.exists():
        result = json.loads(result_path.read_text())
    else:
        result = {}
    result.update(
        {
            "label": candidate.label,
            "m_a": candidate.m_a,
            "panel_slab": candidate.panel_slab,
            "c_panel_slab": candidate.c_panel_slab,
            "eligible": candidate.eligible,
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "note": candidate.note,
            "power_mode": args.power_mode,
            "conversion_npu_us": None,
            "dma_npu_us": None,
            "phase_timing_note": (
                "Only total NPU time is instrumented; resident and prepared "
                "variants bound compute and transport phases."
            ),
        }
    )
    average_us = result.get("average_npu_us")
    if average_us:
        waves = 1 if candidate.flow == "resident" else M // (4 * candidate.m_a)
        a_bytes = (
            8
            * 4
            * candidate.m_a
            * (K // 8)
            * 2
            * waves
        )
        tile_bytes = (
            (K // 8) * candidate.n * 9 // 8
            if candidate.flow in ("bfp-prepared", "resident")
            else (K // 8) * candidate.n // 2
            + (K // 8 // 32) * candidate.n * 4
        )
        panels = 1 if candidate.flow == "resident" else N // candidate.n
        b_bytes = 8 * panels * tile_bytes * waves
        c_bytes = (
            4 * candidate.m_a * candidate.n * 2
            if candidate.flow == "resident"
            else M * N * 2
        )
        transfer_bytes = a_bytes + b_bytes + c_bytes
        result["scheduled_device_bytes"] = transfer_bytes
        result["effective_device_gbytes_per_s"] = (
            transfer_bytes / (average_us * 1000.0)
        )
    return result


def run_parent_baseline(args: argparse.Namespace) -> dict:
    result_path = args.results_dir / "parent_whole_array.json"
    command = [
        sys.executable,
        str(HERE.parent / "whole_array" / "whole_array.py"),
        "-M",
        str(args.M),
        "-K",
        str(args.K),
        "-N",
        str(args.N),
        "--m-c",
        "128",
        "--m-a",
        "32",
        "-k",
        "64",
        "-n",
        "128",
        "--n-aie-cols",
        "8",
        "--compute-type",
        "bfp16",
        "--accumulation-mode",
        "bf16",
        "--cache-mode",
        "l1-weight",
        "--activation-input",
        "bf16",
        "--cache-k",
        str(args.K),
        "--verify-mode",
        "none",
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--benchmark-repeats",
        str(args.rounds),
        "--benchmark-json",
        str(result_path),
    ]
    print("\n$", " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    result.update(
        {
            "label": "parent-whole-array",
            "status": "passed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "eligible": True,
            "power_mode": args.power_mode,
        }
    )
    return result


def write_tables(args: argparse.Namespace, rows: list[dict]) -> None:
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": device_identity(),
        "power_mode": args.power_mode,
        "preset": args.preset,
        "warmup": args.warmup,
        "iters": args.iters,
        "rounds": args.rounds,
    }
    payload = {"metadata": metadata, "results": rows}
    (args.results_dir / f"{args.preset}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    fields = sorted({key for row in rows for key in row})
    with (args.results_dir / f"{args.preset}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if min(args.M, args.K, args.N, args.iters, args.rounds) <= 0:
        raise SystemExit("dimensions, iterations, and rounds must be positive")
    if args.warmup < 0 or args.min_gflops < 0:
        raise SystemExit("warmup and min-gflops must be non-negative")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dimensions = (args.M, args.K, args.N)
    if args.preset == "primitives":
        dimensions = (1024, 4096, 1024)
    rows: list[dict] = []
    if args.hardware:
        for candidate in candidates(args.preset):
            rows.append(run_candidate(candidate, args, dimensions))
        if args.include_baseline:
            rows.append(run_parent_baseline(args))
    else:
        rows = [
            {
                "label": item.label,
                "variant": item.flow,
                "transport": item.transport,
                "n": item.n,
                "m_a": item.m_a,
                "panel_slab": item.panel_slab,
                "c_panel_slab": item.c_panel_slab,
                "eligible": item.eligible,
                "status": "planned",
                "note": item.note,
            }
            for item in candidates(args.preset)
        ]
    write_tables(args, rows)

    for row in rows:
        throughput = row.get("median_gflops")
        rendered = "n/a" if throughput is None else f"{throughput:.2f}"
        print(
            f"{row['label']}: status={row['status']}, "
            f"median_gflops={rendered}, eligible={row.get('eligible')}"
        )
    if args.min_gflops:
        gate_rows = [
            row
            for row in rows
            if row.get("median_gflops") is not None
            and (args.preset == "ceiling" or row.get("eligible"))
        ]
        gate_kind = "ceiling" if args.preset == "ceiling" else "eligible"
        if not gate_rows or max(row["median_gflops"] for row in gate_rows) < args.min_gflops:
            raise SystemExit(
                f"no {gate_kind} result reached {args.min_gflops:.2f} GFLOP/s"
            )


if __name__ == "__main__":
    main()
