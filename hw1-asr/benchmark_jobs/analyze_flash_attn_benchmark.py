#!/usr/bin/env python3
"""
Analyze Flash Attention Fusion benchmark results.

Parses the output logs from the correctness and detailed benchmark jobs.
Computes performance improvement statistics with confidence intervals.

Usage:
    python analyze_flash_attn_benchmark.py <correctness_log> [detailed_log]
"""

import re
import sys
import math
from collections import defaultdict


def parse_benchmark_log(filepath):
    """Parse a benchmark.sh output log for timing and accuracy results."""
    results = {}
    current_section = None

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Detect section headers
        if "Flash Attention Fusion ENABLED" in line:
            current_section = "fusion_on"
            results[current_section] = {"runs": [], "accuracy": None, "status": None}
        elif "Flash Attention Fusion DISABLED" in line:
            current_section = "fusion_off"
            results[current_section] = {"runs": [], "accuracy": None, "status": None}
        elif "Triton Example" in line:
            current_section = "baseline"
            results[current_section] = {"runs": [], "accuracy": None, "status": None}

        if current_section is None:
            continue

        # Parse run timing
        run_match = re.match(r"Run\s+\d+:\s+([\d.]+)s", line)
        if run_match:
            results[current_section]["runs"].append(float(run_match.group(1)))

        # Parse average inference time
        avg_match = re.match(r"Average inference time:\s+([\d.]+)s", line)
        if avg_match:
            results[current_section]["avg_time"] = float(avg_match.group(1))

        # Parse accuracy
        acc_match = re.match(r"Accuracy:\s+([\d.]+)%", line)
        if acc_match:
            results[current_section]["accuracy"] = float(acc_match.group(1))

        # Parse status
        if "Status: PASS" in line:
            results[current_section]["status"] = "PASS"
        elif "Status: FAIL" in line:
            results[current_section]["status"] = "FAIL"

    return results


def parse_detailed_log(filepath):
    """Parse a benchmark_detailed.sh output log for per-operator timings."""
    sections = {}
    current_section = None
    operator_data = {}

    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if "Flash Attention Fusion ENABLED" in line:
            current_section = "fusion_on"
            operator_data = {}
            sections[current_section] = operator_data
        elif "Flash Attention Fusion DISABLED" in line:
            current_section = "fusion_off"
            operator_data = {}
            sections[current_section] = operator_data
        elif "Triton Example" in line:
            current_section = "baseline"
            operator_data = {}
            sections[current_section] = operator_data

        if current_section is None:
            continue

        # Typical format: "  operator_name: XX.XXms (XX.X%)"
        op_match = re.match(r"\s*([\w_]+):\s+([\d.]+)ms", line)
        if op_match:
            op_name = op_match.group(1)
            op_time = float(op_match.group(2))
            operator_data[op_name] = op_time

    return sections


def compute_stats(values):
    """Compute mean, std, and 95% confidence interval."""
    n = len(values)
    if n == 0:
        return None, None, None, None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, mean, mean

    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    # t-value for 95% CI (approximate, conservative for small n)
    t_vals = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
              7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    t_val = t_vals.get(n, 1.96)
    margin = t_val * std / math.sqrt(n)

    return mean, std, mean - margin, mean + margin


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_flash_attn_benchmark.py <log_file> [detailed_log_file]")
        sys.exit(1)

    log_file = sys.argv[1]
    detailed_log = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 70)
    print("  Flash Attention Fusion Benchmark Analysis")
    print("=" * 70)

    # Parse correctness/timing log
    results = parse_benchmark_log(log_file)

    for name, data in results.items():
        label = {
            "fusion_on": "Fusion ENABLED",
            "fusion_off": "Fusion DISABLED",
            "baseline": "Reference Baseline"
        }.get(name, name)

        print(f"\n--- {label} ---")
        print(f"  Accuracy:  {data.get('accuracy', 'N/A')}%")
        print(f"  Status:    {data.get('status', 'N/A')}")

        runs = data.get("runs", [])
        if runs:
            mean, std, ci_lo, ci_hi = compute_stats(runs)
            print(f"  Runs:      {len(runs)}")
            print(f"  Mean time: {mean:.4f}s ± {std:.4f}s")
            print(f"  95% CI:    [{ci_lo:.4f}s, {ci_hi:.4f}s]")

    # Compare performance
    if "fusion_on" in results and "fusion_off" in results:
        on_runs = results["fusion_on"].get("runs", [])
        off_runs = results["fusion_off"].get("runs", [])

        if on_runs and off_runs:
            on_mean, _, _, _ = compute_stats(on_runs)
            off_mean, _, _, _ = compute_stats(off_runs)
            speedup = off_mean / on_mean if on_mean > 0 else float("inf")
            pct_change = (off_mean - on_mean) / off_mean * 100

            print(f"\n{'=' * 70}")
            print(f"  COMPARISON: Fusion ON vs Fusion OFF")
            print(f"{'=' * 70}")
            print(f"  Fusion ON mean:   {on_mean:.4f}s")
            print(f"  Fusion OFF mean:  {off_mean:.4f}s")
            print(f"  Speedup:          {speedup:.3f}x")
            print(f"  Improvement:      {pct_change:+.1f}%")

    if "fusion_on" in results and "baseline" in results:
        on_runs = results["fusion_on"].get("runs", [])
        base_runs = results["baseline"].get("runs", [])

        if on_runs and base_runs:
            on_mean, _, _, _ = compute_stats(on_runs)
            base_mean, _, _, _ = compute_stats(base_runs)
            speedup = base_mean / on_mean if on_mean > 0 else float("inf")
            pct_change = (base_mean - on_mean) / base_mean * 100

            print(f"\n{'=' * 70}")
            print(f"  COMPARISON: Fusion ON vs Reference Baseline")
            print(f"{'=' * 70}")
            print(f"  Fusion ON mean:   {on_mean:.4f}s")
            print(f"  Baseline mean:    {base_mean:.4f}s")
            print(f"  Speedup:          {speedup:.3f}x")
            print(f"  Improvement:      {pct_change:+.1f}%")

    # Parse detailed log if available
    if detailed_log:
        print(f"\n{'=' * 70}")
        print(f"  PER-OPERATOR TIMINGS")
        print(f"{'=' * 70}")

        detailed = parse_detailed_log(detailed_log)
        for name, ops in detailed.items():
            label = {
                "fusion_on": "Fusion ENABLED",
                "fusion_off": "Fusion DISABLED",
                "baseline": "Reference Baseline"
            }.get(name, name)
            print(f"\n--- {label} ---")
            for op, time in sorted(ops.items(), key=lambda x: -x[1]):
                print(f"  {op:30s}: {time:8.2f}ms")


if __name__ == "__main__":
    main()
