#!/usr/bin/env python3
"""Debug script to find correctness issues in DecoderRMSNormQKV fusion."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../glm_asr_triton_template"))
import torch
import numpy as np

torch.manual_seed(42)
device = torch.device("cuda")

# Small dimensions for easy debugging
hidden_size = 2048
num_heads = 16
num_kv_heads = 4
head_dim = 128

from layers import RMSNorm, Linear, DecoderRMSNormQKV

# Create modules
rmsnorm = RMSNorm(hidden_size)
rmsnorm.weight = torch.randn(hidden_size, dtype=torch.float32, device=device)

q_proj = Linear(hidden_size, num_heads * head_dim, bias=False)
q_proj.weight = torch.randn(num_heads * head_dim, hidden_size, dtype=torch.float32, device=device)

k_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
k_proj.weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.float32, device=device)

v_proj = Linear(hidden_size, num_kv_heads * head_dim, bias=False)
v_proj.weight = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=torch.float32, device=device)

fused = DecoderRMSNormQKV(rmsnorm, q_proj, k_proj, v_proj)

# Create input
x = torch.randn(1, 4, hidden_size, dtype=torch.float32, device=device)

# Step 1: Compute reference RMSNorm
x_2d = x.reshape(-1, hidden_size)
variance = torch.mean(x_2d * x_2d, dim=-1, keepdim=True)
inv_rms = torch.rsqrt(variance + rmsnorm.eps)
x_normed_ref = x_2d * inv_rms * rmsnorm.weight[None, :]

print("Step 1: RMSNorm reference computed")
print(f"  x_2d shape: {x_2d.shape}")
print(f"  inv_rms: {inv_rms.squeeze()[:4]}")
print(f"  x_normed_ref[0,:4]: {x_normed_ref[0,:4]}")

# Step 2: Compute reference Q via cuBLAS
Linear.BACKEND = "cublas"
q_ref = q_proj(x_normed_ref.reshape(1, 4, hidden_size))
k_ref = k_proj(x_normed_ref.reshape(1, 4, hidden_size))
v_ref = v_proj(x_normed_ref.reshape(1, 4, hidden_size))

print(f"\nStep 2: cuBLAS Q/K/V reference")
print(f"  q_ref[0,0,:4]: {q_ref[0,0,:4]}")
print(f"  k_ref[0,0,:4]: {k_ref[0,0,:4]}")

# Step 3: Compute fused
fused.__class__.FUSED = True
q_fused, k_fused, v_fused = fused(x)

print(f"\nStep 3: Fused Q/K/V")
print(f"  q_fused[0,0,:4]: {q_fused[0,0,:4]}")
print(f"  k_fused[0,0,:4]: {k_fused[0,0,:4]}")

# Step 4: Check RMSNorm via triton
q_via_rmsnorm = rmsnorm(x)
print(f"\nStep 4: RMSNorm via triton kernel")
print(f"  normed[0,0,:4]: {q_via_rmsnorm[0,0,:4]}")
print(f"  normed_ref[0,:4]: {x_normed_ref[0,:4]}")
print(f"  RMSNorm diff: {(q_via_rmsnorm.reshape(-1, hidden_size) - x_normed_ref).abs().max():.2e}")

# Step 5: Error analysis
err_q = (q_ref - q_fused).abs()
err_k = (k_ref - k_fused).abs()
print(f"\nStep 5: Error analysis")
print(f"  Q max err: {err_q.max():.4e}, mean: {err_q.mean():.4e}")
print(f"  K max err: {err_k.max():.4e}, mean: {err_k.mean():.4e}")

# Step 6: Check if the weight is the issue
# The fused kernel uses _weight_t_padded. Let's check it matches weight.t()
q_proj._ensure_weight_prepared()
print(f"\nStep 6: Weight check")
print(f"  q_proj.weight shape: {q_proj.weight.shape}")
print(f"  q_proj._weight_t_padded shape: {q_proj._weight_t_padded.shape}")
print(f"  weight.t() == _weight_t_padded: {torch.allclose(q_proj.weight.t(), q_proj._weight_t_padded[:hidden_size, :num_heads*head_dim])}")

# Step 7: Manual matmul with the same weight
q_manual = x_normed_ref @ q_proj._weight_t_padded[:hidden_size, :num_heads*head_dim]
print(f"\nStep 7: Manual matmul with transposed padded weight")
print(f"  q_manual[0,:4]: {q_manual[0,:4]}")
print(f"  q_ref[0,0,:4]: {q_ref[0,0,:4]}")
print(f"  Manual vs cuBLAS Q max err: {(q_manual - q_ref.reshape(-1, num_heads*head_dim)).abs().max():.2e}")
print(f"  Manual vs fused Q max err: {(q_manual - q_fused.reshape(-1, num_heads*head_dim)).abs().max():.2e}")

# Step 8: Check if rmsnorm hidden_size vs weight dim is correct
print(f"\nStep 8: Dimension checks")
print(f"  rmsnorm.hidden_size: {rmsnorm.hidden_size}")
print(f"  rmsnorm.use_triton: {rmsnorm.use_triton}")  
print(f"  hidden_size is power of 2: {hidden_size > 0 and (hidden_size & (hidden_size - 1)) == 0}")
