#!/usr/bin/env python3
"""Compare detailed benchmark JSON outputs."""

import argparse
import json
from typing import Dict, Tuple


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(results: Dict, key: str) -> float:
    return float(results["results"][key]["mean"])


def total_estimate(results: Dict) -> float:
    return (
        metric(results, "audio_encoder")
        + metric(results, "projector")
        + metric(results, "decoder_prefill")
        + 50.0 * metric(results, "decode_step")
    )


def avg_layer_time(results: Dict) -> float:
    layers = results["results"].get("layers", [])
    if not layers:
        return 0.0
    return sum(float(layer["mean"]) for layer in layers) / len(layers)


def compare(base: float, cand: float) -> Tuple[float, float]:
    delta = cand - base
    pct = 0.0 if base == 0 else (base - cand) / base * 100.0
    return delta, pct


def print_line(label: str, base: float, cand: float) -> None:
    delta, pct = compare(base, cand)
    print(f"{label:<28} baseline={base:>8.2f}ms  candidate={cand:>8.2f}ms  delta={delta:>8.2f}ms  improvement={pct:>7.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze fusion benchmark outputs")
    parser.add_argument("--baseline", required=True, help="Baseline JSON path")
    parser.add_argument("--candidate", required=True, help="Candidate JSON path")
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)

    print("=" * 90)
    print("Fusion Benchmark Comparison")
    print("=" * 90)
    print(f"Baseline : {baseline['folder']}")
    print(f"Candidate: {candidate['folder']}")
    print("")

    print_line("Audio encoder", metric(baseline, "audio_encoder"), metric(candidate, "audio_encoder"))
    print_line("Projector", metric(baseline, "projector"), metric(candidate, "projector"))
    print_line("Decoder prefill", metric(baseline, "decoder_prefill"), metric(candidate, "decoder_prefill"))
    print_line("Decode step", metric(baseline, "decode_step"), metric(candidate, "decode_step"))
    print_line("Avg first 5 layers", avg_layer_time(baseline), avg_layer_time(candidate))
    print_line("Estimated total", total_estimate(baseline), total_estimate(candidate))

    print("")
    layer_base = baseline["results"].get("layers", [])
    layer_cand = candidate["results"].get("layers", [])
    count = min(len(layer_base), len(layer_cand))
    if count:
        print("Per-layer comparison:")
        for idx in range(count):
            print_line(
                f"Layer {idx}",
                float(layer_base[idx]["mean"]),
                float(layer_cand[idx]["mean"]),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
