#!/usr/bin/env python3
"""
Micro-benchmark for DecoderRMSNormQKV fusion.

Measures:
  1. Correctness: max absolute error between fused and unfused paths.
  2. Performance: latency of fused vs unfused under realistic workloads
     matching GLM-ASR Text Decoder dimensions.

Text Decoder dimensions (from GlmAsrConfig):
  hidden_size = 3584  (but Nano-2512 actually uses 2048)
  num_heads = 28  (Nano: 16)
  num_kv_heads = 4
  head_dim = 128
  Q output dim = num_heads * head_dim = 28 * 128 = 3584  (Nano: 16 * 128 = 2048)
  K/V output dim = num_kv_heads * head_dim = 4 * 128 = 512

We benchmark with the Nano-2512 dimensions since that's the model used.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Insert the template folder so we can import its layers module directly
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "glm_asr_triton_template")
if TEMPLATE_DIR not in sys.path:
    sys.path.insert(0, TEMPLATE_DIR)


def clear_modules():
    for mod_name in list(sys.modules.keys()):
        if mod_name in ["layers", "attention", "rope", "conv", "model", "weight_loader"]:
            del sys.modules[mod_name]


def create_test_modules(hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, device: torch.device):
    """Create RMSNorm, Q/K/V Linear modules with random weights."""
    clear_modules()
    from layers import RMSNorm, Linear, DecoderRMSNormQKV

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim

    rmsnorm = RMSNorm(hidden_size)
    rmsnorm.weight = torch.randn(hidden_size, dtype=torch.float32, device=device)

    q_proj = Linear(hidden_size, q_dim, bias=False)
    q_proj.weight = torch.randn(q_dim, hidden_size, dtype=torch.float32, device=device)
    q_proj._weight_t_padded = None

    k_proj = Linear(hidden_size, kv_dim, bias=False)
    k_proj.weight = torch.randn(kv_dim, hidden_size, dtype=torch.float32, device=device)
    k_proj._weight_t_padded = None

    v_proj = Linear(hidden_size, kv_dim, bias=False)
    v_proj.weight = torch.randn(kv_dim, hidden_size, dtype=torch.float32, device=device)
    v_proj._weight_t_padded = None

    fused = DecoderRMSNormQKV(rmsnorm, q_proj, k_proj, v_proj)

    return rmsnorm, q_proj, k_proj, v_proj, fused


def run_unfused(rmsnorm, q_proj, k_proj, v_proj, x):
    """Unfused path: RMSNorm -> separate Q, K, V linear."""
    normed = rmsnorm(x)
    q = q_proj(normed)
    k = k_proj(normed)
    v = v_proj(normed)
    return q, k, v


def correctness_check(hidden_size, num_heads, num_kv_heads, head_dim, seq_lens, device):
    """Check numerical correctness of fused vs unfused."""
    print("\n" + "=" * 70)
    print("CORRECTNESS CHECK")
    print("=" * 70)

    rmsnorm, q_proj, k_proj, v_proj, fused = create_test_modules(
        hidden_size, num_heads, num_kv_heads, head_dim, device
    )

    results = []
    for seq_len in seq_lens:
        x = torch.randn(1, seq_len, hidden_size, dtype=torch.float32, device=device)

        # Unfused
        q_ref, k_ref, v_ref = run_unfused(rmsnorm, q_proj, k_proj, v_proj, x)

        # Fused
        old_fused_flag = fused.FUSED
        fused.__class__.FUSED = True  # ensure class-level flag is True
        q_fused, k_fused, v_fused = fused(x)
        fused.__class__.FUSED = old_fused_flag

        err_q = (q_ref - q_fused).abs().max().item()
        err_k = (k_ref - k_fused).abs().max().item()
        err_v = (v_ref - v_fused).abs().max().item()

        status = "✅" if max(err_q, err_k, err_v) < 1e-3 else "❌"
        print(f"  seq_len={seq_len:>4}: max_err Q={err_q:.2e}, K={err_k:.2e}, V={err_v:.2e}  {status}")
        results.append({
            "seq_len": seq_len,
            "err_q": err_q,
            "err_k": err_k,
            "err_v": err_v,
            "pass": max(err_q, err_k, err_v) < 1e-3,
        })
    return results


def benchmark_one(fn, warmup_iters, bench_iters, device):
    """Benchmark a callable, returning list of latencies in ms."""
    # Warmup
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize(device)

    latencies = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize(device)
        latencies.append(start.elapsed_time(end))
    return latencies


def performance_benchmark(
    hidden_size, num_heads, num_kv_heads, head_dim,
    configs, warmup_iters, bench_iters, device
):
    """Benchmark fused vs unfused across multiple configs."""
    print("\n" + "=" * 70)
    print(f"PERFORMANCE BENCHMARK ({bench_iters} runs, {warmup_iters} warmup)")
    print("=" * 70)

    rmsnorm, q_proj, k_proj, v_proj, fused = create_test_modules(
        hidden_size, num_heads, num_kv_heads, head_dim, device
    )

    results = []
    for label, batch, seq_len in configs:
        x = torch.randn(batch, seq_len, hidden_size, dtype=torch.float32, device=device)

        # Unfused
        unfused_lats = benchmark_one(
            lambda: run_unfused(rmsnorm, q_proj, k_proj, v_proj, x),
            warmup_iters, bench_iters, device
        )

        # Fused
        old_fused = fused.__class__.FUSED
        fused.__class__.FUSED = True
        fused_lats = benchmark_one(
            lambda: fused(x),
            warmup_iters, bench_iters, device
        )
        fused.__class__.FUSED = old_fused

        uf_mean = np.mean(unfused_lats)
        uf_std = np.std(unfused_lats)
        f_mean = np.mean(fused_lats)
        f_std = np.std(fused_lats)
        speedup = uf_mean / f_mean if f_mean > 0 else float("inf")
        pct = (uf_mean - f_mean) / uf_mean * 100

        print(f"\n  {label} (b={batch}, s={seq_len}):")
        print(f"    Unfused : {uf_mean:.4f} ± {uf_std:.4f} ms")
        print(f"    Fused   : {f_mean:.4f} ± {f_std:.4f} ms")
        print(f"    Speedup : {speedup:.3f}×  ({pct:+.1f}%)")

        results.append({
            "label": label,
            "batch": batch,
            "seq_len": seq_len,
            "unfused_mean_ms": uf_mean,
            "unfused_std_ms": uf_std,
            "fused_mean_ms": f_mean,
            "fused_std_ms": f_std,
            "speedup": speedup,
            "improvement_pct": pct,
            "unfused_raw": unfused_lats,
            "fused_raw": fused_lats,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="DecoderRMSNormQKV micro-benchmark")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--json-out", type=str, help="Save results as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # GLM-ASR Nano-2512 dimensions
    hidden_size = 2048
    num_heads = 16
    num_kv_heads = 4
    head_dim = 128

    print(f"\nModel dimensions:")
    print(f"  hidden_size = {hidden_size}")
    print(f"  num_heads = {num_heads}, num_kv_heads = {num_kv_heads}, head_dim = {head_dim}")
    print(f"  Q output = {num_heads * head_dim}")
    print(f"  K/V output = {num_kv_heads * head_dim}")

    # --- Correctness ---
    correctness_results = correctness_check(
        hidden_size, num_heads, num_kv_heads, head_dim,
        seq_lens=[1, 16, 59, 128, 256],
        device=device,
    )

    # --- Performance ---
    configs = [
        ("Prefill (typical ASR)", 1, 59),
        ("Decode 1-token",        1, 1),
        ("Short prefill",         1, 16),
        ("Long prefill",          1, 128),
        ("Max prefill",           1, 256),
        ("Batch 4 decode",        4, 1),
        ("Batch 8 decode",        8, 1),
    ]

    perf_results = performance_benchmark(
        hidden_size, num_heads, num_kv_heads, head_dim,
        configs, args.warmup, args.runs, device,
    )

    # --- Save results ---
    payload = {
        "device": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu",
        "model_dims": {
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "warmup_iters": args.warmup,
        "bench_iters": args.runs,
        "correctness": correctness_results,
        "performance": [
            {k: v for k, v in r.items() if k not in ("unfused_raw", "fused_raw")}
            for r in perf_results
        ],
    }

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
        print(f"\nResults saved to: {args.json_out}")

    # Also print JSON to stdout for log parsing
    print("\nMICROBENCH_JSON=" + json.dumps(payload, default=lambda o: float(o) if isinstance(o, np.floating) else o))


if __name__ == "__main__":
    main()
