#!/usr/bin/env python3
"""
Micro-benchmark for Flash Attention Fusion vs Unfused Attention.

Measures the isolated attention kernel performance across the model's
actual operating points (audio encoder and text decoder dimensions).
"""

import torch
import time
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm_asr_triton_template'))

from attention import (
    _scaled_dot_product_attention_fused,
    _scaled_dot_product_attention_unfused,
)


def benchmark_attention(fn, q, k, v, mask, is_causal, warmup=50, runs=200):
    """Benchmark a single attention function call."""
    # Warmup
    for _ in range(warmup):
        fn(q, k, v, mask, is_causal)
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    for i in range(runs):
        start_events[i].record()
        fn(q, k, v, mask, is_causal)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times


def compute_stats(times):
    """Mean, std, 95% CI (t-distribution)."""
    n = len(times)
    mean = sum(times) / n
    if n <= 1:
        return mean, 0.0, mean, mean

    var = sum((t - mean) ** 2 for t in times) / (n - 1)
    std = math.sqrt(var)
    # Use t-distribution approximation for 95% CI
    t_val = 1.96  # close enough for n > 30
    margin = t_val * std / math.sqrt(n)
    return mean, std, mean - margin, mean + margin


def main():
    device = torch.device("cuda")

    # Model operating points
    configs = [
        # (name, batch, heads, seq_q, seq_k, head_dim, is_causal)
        ("Audio Encoder (prefill)", 1, 20, 187, 187, 64, False),
        ("Audio Encoder (single head)", 1, 1, 187, 187, 64, False),
        ("Text Decoder (prefill)", 1, 16, 59, 59, 128, True),
        ("Text Decoder (1-token decode, cache=20)", 1, 16, 1, 20, 128, False),
        ("Text Decoder (1-token decode, cache=50)", 1, 16, 1, 50, 128, False),
        ("Text Decoder (1-token decode, cache=100)", 1, 16, 1, 100, 128, False),
        ("Text Decoder (1-token decode, cache=200)", 1, 16, 1, 200, 128, False),
        ("Small test (seq=16, dim=64)", 2, 4, 16, 16, 64, False),
        ("Causal small test (seq=16, dim=64)", 2, 4, 16, 16, 64, True),
    ]

    results_rows = []

    print("=" * 90)
    print("  Flash Attention Fusion Micro-Benchmark")
    print("=" * 90)
    print()

    for name, batch, heads, seq_q, seq_k, head_dim, is_causal in configs:
        q = torch.randn(batch, heads, seq_q, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch, heads, seq_k, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch, heads, seq_k, head_dim, device=device, dtype=torch.float32)

        print(f"--- {name} ---")
        print(f"    Shape: batch={batch}, heads={heads}, seq_q={seq_q}, seq_k={seq_k}, dim={head_dim}, causal={is_causal}")

        # Verify correctness first
        out_fused = _scaled_dot_product_attention_fused(q, k, v, None, is_causal)
        out_unfused = _scaled_dot_product_attention_unfused(q, k, v, None, is_causal)
        max_diff = (out_fused - out_unfused).abs().max().item()
        print(f"    Max diff (fused vs unfused): {max_diff:.2e}")
        assert max_diff < 1e-3, f"Outputs differ too much: {max_diff}"

        # Benchmark
        times_fused = benchmark_attention(
            _scaled_dot_product_attention_fused, q, k, v, None, is_causal
        )
        times_unfused = benchmark_attention(
            _scaled_dot_product_attention_unfused, q, k, v, None, is_causal
        )

        mean_f, std_f, ci_lo_f, ci_hi_f = compute_stats(times_fused)
        mean_u, std_u, ci_lo_u, ci_hi_u = compute_stats(times_unfused)

        speedup = mean_u / mean_f if mean_f > 0 else float("inf")
        pct_improvement = (mean_u - mean_f) / mean_u * 100

        print(f"    Fused:   {mean_f:.4f}ms ± {std_f:.4f}ms  [95% CI: {ci_lo_f:.4f} - {ci_hi_f:.4f}]")
        print(f"    Unfused: {mean_u:.4f}ms ± {std_u:.4f}ms  [95% CI: {ci_lo_u:.4f} - {ci_hi_u:.4f}]")
        print(f"    Speedup: {speedup:.3f}x  ({pct_improvement:+.1f}%)")
        print()

        results_rows.append({
            "name": name,
            "mean_fused": mean_f,
            "mean_unfused": mean_u,
            "speedup": speedup,
            "pct_improvement": pct_improvement,
            "max_diff": max_diff,
        })

    # Summary table
    print("=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Config':<50} {'Fused':>10} {'Unfused':>10} {'Speedup':>10} {'Improv':>10}")
    print("-" * 90)
    for r in results_rows:
        print(f"{r['name']:<50} {r['mean_fused']:>9.4f}ms {r['mean_unfused']:>9.4f}ms {r['speedup']:>9.3f}x {r['pct_improvement']:>+9.1f}%")


if __name__ == "__main__":
    main()
