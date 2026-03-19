"""Quick correctness test for refactored DecoderRMSNormQKV (Plan C)."""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

import torch

# Bypass __init__.py defaults — test both fused and unfused
def test_correctness():
    from glm_asr_triton_template.layers import (
        RMSNorm, Linear, DecoderRMSNormQKV
    )

    torch.manual_seed(42)
    device = torch.device("cuda")

    # Model dimensions from GLM-ASR Nano-2512
    hidden_size = 2048
    n_q = 2048    # 16 heads * 128 head_dim
    n_k = 512     # 4 kv_heads * 128 head_dim
    n_v = 512

    rmsnorm = RMSNorm(hidden_size)
    q_proj = Linear(hidden_size, n_q, bias=False)
    k_proj = Linear(hidden_size, n_k, bias=False)
    v_proj = Linear(hidden_size, n_v, bias=False)

    # Initialize with random weights
    rmsnorm.weight = torch.randn(hidden_size, dtype=torch.float32, device=device)
    q_proj.weight = torch.randn(n_q, hidden_size, dtype=torch.float32, device=device)
    k_proj.weight = torch.randn(n_k, hidden_size, dtype=torch.float32, device=device)
    v_proj.weight = torch.randn(n_v, hidden_size, dtype=torch.float32, device=device)

    fused = DecoderRMSNormQKV(rmsnorm, q_proj, k_proj, v_proj)

    test_configs = [
        ("decode b=1 s=1",   (1, 1, hidden_size)),
        ("prefill b=1 s=16", (1, 16, hidden_size)),
        ("prefill b=1 s=59", (1, 59, hidden_size)),
        ("prefill b=1 s=128",(1, 128, hidden_size)),
        ("prefill b=1 s=256",(1, 256, hidden_size)),
        ("decode b=4 s=1",   (4, 1, hidden_size)),
        ("decode b=8 s=1",   (8, 1, hidden_size)),
    ]

    print("=" * 60)
    print("DecoderRMSNormQKV Refactor (Plan C) — Correctness Test")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Dimensions: K={hidden_size}, N_q={n_q}, N_k={n_k}, N_v={n_v}")
    print()

    all_pass = True
    for name, shape in test_configs:
        x = torch.randn(shape, dtype=torch.float32, device=device)

        # Reference: unfused path (RMSNorm + separate matmuls)
        x_norm_ref = rmsnorm(x)
        M = x_norm_ref.reshape(-1, hidden_size).shape[0]
        x_norm_2d = x_norm_ref.reshape(M, hidden_size)
        q_ref = (x_norm_2d @ q_proj.weight.t()).reshape(x.shape[:-1] + (n_q,))
        k_ref = (x_norm_2d @ k_proj.weight.t()).reshape(x.shape[:-1] + (n_k,))
        v_ref = (x_norm_2d @ v_proj.weight.t()).reshape(x.shape[:-1] + (n_v,))

        # Fused path (Plan C)
        q_fused, k_fused, v_fused = fused(x)

        dq = (q_fused - q_ref).abs().max().item()
        dk = (k_fused - k_ref).abs().max().item()
        dv = (v_fused - v_ref).abs().max().item()

        # Since both paths use the same cuBLAS GEMM now, errors should be zero or near-zero
        status = "PASS" if max(dq, dk, dv) < 1e-3 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {name:25s}  max|dQ|={dq:.2e}  max|dK|={dk:.2e}  max|dV|={dv:.2e}  [{status}]")

    print()

    # Benchmark: measure latency
    print("=" * 60)
    print("Performance Comparison (200 runs, 50 warmup)")
    print("=" * 60)

    for name, shape in test_configs:
        x = torch.randn(shape, dtype=torch.float32, device=device)

        # Warmup
        for _ in range(50):
            _ = fused(x)
        torch.cuda.synchronize()

        # Fused timing
        start = time.perf_counter()
        for _ in range(200):
            _ = fused(x)
        torch.cuda.synchronize()
        fused_ms = (time.perf_counter() - start) / 200 * 1000

        # Unfused timing (RMSNorm + 3 separate cuBLAS GEMMs)
        for _ in range(50):
            xn = rmsnorm(x)
            _ = q_proj(xn)
            _ = k_proj(xn)
            _ = v_proj(xn)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(200):
            xn = rmsnorm(x)
            _ = q_proj(xn)
            _ = k_proj(xn)
            _ = v_proj(xn)
        torch.cuda.synchronize()
        unfused_ms = (time.perf_counter() - start) / 200 * 1000

        speedup = unfused_ms / fused_ms if fused_ms > 0 else float('inf')
        change = (1 - fused_ms / unfused_ms) * 100 if unfused_ms > 0 else 0
        print(f"  {name:25s}  unfused={unfused_ms:.3f}ms  fused={fused_ms:.3f}ms  speedup={speedup:.3f}x  ({change:+.1f}%)")

    print()
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

if __name__ == "__main__":
    test_correctness()
