#!/usr/bin/env python3
"""Compare JSON outputs from benchmark_with_flags.py."""

from __future__ import annotations

import argparse
import json
from typing import Dict, Tuple


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def compare(base: float, cand: float) -> Tuple[float, float]:
    delta = cand - base
    pct = 0.0 if base == 0 else (base - cand) / base * 100.0
    return delta, pct


def print_line(label: str, base: float, cand: float, unit: str = "ms") -> None:
    delta, pct = compare(base, cand)
    print(
        f"{label:<28} baseline={base:>10.2f}{unit}  "
        f"candidate={cand:>10.2f}{unit}  delta={delta:>10.2f}{unit}  "
        f"improvement={pct:>7.2f}%"
    )


def total_estimate(results: Dict) -> float:
    metrics = results["results"]
    return (
        float(metrics["audio_encoder"]["mean"])
        + float(metrics["projector"]["mean"])
        + float(metrics["decoder_prefill"]["mean"])
        + 50.0 * float(metrics["decode_step"]["mean"])
    )


def avg_layer_time(results: Dict) -> float:
    layers = results["results"].get("layers", [])
    if not layers:
        return 0.0
    return sum(float(layer["mean"]) for layer in layers) / len(layers)


def compare_e2e(baseline: Dict, candidate: Dict) -> None:
    base_results = baseline["results"]
    cand_results = candidate["results"]

    print_line("End-to-end mean", float(base_results["mean"]), float(cand_results["mean"]))
    print_line("End-to-end std", float(base_results["std"]), float(cand_results["std"]))
    print_line(
        "Speed (ms/token)",
        float(base_results["speed_ms_per_token"]),
        float(cand_results["speed_ms_per_token"]),
    )
    if base_results.get("accuracy") is not None and cand_results.get("accuracy") is not None:
        print_line(
            "Accuracy",
            100.0 * float(base_results["accuracy"]),
            100.0 * float(cand_results["accuracy"]),
            unit="%",
        )
    print(f"{'Baseline status':<28} {base_results.get('status', 'N/A')}")
    print(f"{'Candidate status':<28} {cand_results.get('status', 'N/A')}")
    print(f"{'Baseline tokens':<28} {base_results.get('tokens', 'N/A')}")
    print(f"{'Candidate tokens':<28} {cand_results.get('tokens', 'N/A')}")


def compare_detailed(baseline: Dict, candidate: Dict) -> None:
    base_results = baseline["results"]
    cand_results = candidate["results"]

    print_line(
        "Audio encoder",
        float(base_results["audio_encoder"]["mean"]),
        float(cand_results["audio_encoder"]["mean"]),
    )
    print_line(
        "Projector",
        float(base_results["projector"]["mean"]),
        float(cand_results["projector"]["mean"]),
    )
    print_line(
        "Decoder prefill",
        float(base_results["decoder_prefill"]["mean"]),
        float(cand_results["decoder_prefill"]["mean"]),
    )
    print_line(
        "Decode step",
        float(base_results["decode_step"]["mean"]),
        float(cand_results["decode_step"]["mean"]),
    )
    print_line("Avg first 5 layers", avg_layer_time(baseline), avg_layer_time(candidate))
    print_line("Estimated total", total_estimate(baseline), total_estimate(candidate))

    layer_base = base_results.get("layers", [])
    layer_cand = cand_results.get("layers", [])
    count = min(len(layer_base), len(layer_cand))
    if count:
        print("\nPer-layer comparison:")
        for idx in range(count):
            print_line(
                f"Layer {idx}",
                float(layer_base[idx]["mean"]),
                float(layer_cand[idx]["mean"]),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark JSON outputs")
    parser.add_argument("--baseline", required=True, help="Baseline JSON path")
    parser.add_argument("--candidate", required=True, help="Candidate JSON path")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)

    baseline_mode = baseline.get("mode")
    candidate_mode = candidate.get("mode")
    if baseline_mode != candidate_mode:
        raise ValueError(
            f"Mode mismatch: baseline={baseline_mode}, candidate={candidate_mode}"
        )

    print("=" * 100)
    print("Benchmark JSON Comparison")
    print("=" * 100)
    print(f"Mode     : {baseline_mode}")
    print(f"Baseline : {baseline['folder']}")
    print(f"Candidate: {candidate['folder']}")
    print(f"Base flags: {json.dumps(baseline.get('flags', {}), sort_keys=True)}")
    print(f"Cand flags: {json.dumps(candidate.get('flags', {}), sort_keys=True)}")
    print("")

    if baseline_mode == "e2e":
        compare_e2e(baseline, candidate)
    elif baseline_mode == "detailed":
        compare_detailed(baseline, candidate)
    else:
        raise ValueError(f"Unsupported benchmark mode: {baseline_mode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
