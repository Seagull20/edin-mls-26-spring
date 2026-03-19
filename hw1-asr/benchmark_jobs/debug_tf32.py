#!/usr/bin/env python3
"""Debug: isolate TF32 vs fp32 issue in fused matmul."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../glm_asr_triton_template"))
import torch
import triton
import triton.language as tl

torch.manual_seed(42)
device = torch.device("cuda")

hidden_size = 2048
out_features = 2048

# Create data
x = torch.randn(4, hidden_size, dtype=torch.float32, device=device)
w = torch.randn(hidden_size, out_features, dtype=torch.float32, device=device)  # transposed weight

# Reference: torch matmul (uses TF32 by default)
torch.backends.cuda.matmul.allow_tf32 = True
ref_tf32 = x @ w
torch.backends.cuda.matmul.allow_tf32 = False
ref_fp32 = x @ w
torch.backends.cuda.matmul.allow_tf32 = True

print(f"torch TF32 vs FP32 max diff: {(ref_tf32 - ref_fp32).abs().max():.4e}")

# Now test: does the fused kernel's tl.dot error match the TF32 error magnitude?
from layers import RMSNorm, Linear, DecoderRMSNormQKV

rmsnorm = RMSNorm(hidden_size)
rmsnorm.weight = torch.ones(hidden_size, dtype=torch.float32, device=device)  # weight=1 => norm * 1 = norm

q_proj = Linear(hidden_size, out_features, bias=False)
q_proj.weight = w.t().contiguous()  # store as (out, in) since Linear stores it that way

k_proj = Linear(hidden_size, 512, bias=False)
k_proj.weight = torch.randn(512, hidden_size, dtype=torch.float32, device=device)

v_proj = Linear(hidden_size, 512, bias=False)
v_proj.weight = torch.randn(512, hidden_size, dtype=torch.float32, device=device)

fused = DecoderRMSNormQKV(rmsnorm, q_proj, k_proj, v_proj)

# Compute normed x manually
x_3d = x.unsqueeze(0)  # (1, 4, 2048)
variance = torch.mean(x * x, dim=-1, keepdim=True)
inv_rms = torch.rsqrt(variance + 1e-6)
x_normed = x * inv_rms  # rmsnorm.weight = 1, so this is the normed value

# Manual matmul (FP32 - without TF32)
torch.backends.cuda.matmul.allow_tf32 = False
q_manual_fp32 = x_normed @ w
torch.backends.cuda.matmul.allow_tf32 = True
q_manual_tf32 = x_normed @ w

# Fused
fused.__class__.FUSED = True
q_fused, _, _ = fused(x_3d)
q_fused = q_fused.squeeze(0)

print(f"\nWith RMSNorm weight=1 (identity):")
print(f"  fused vs manual_fp32 max err: {(q_fused - q_manual_fp32).abs().max():.4e}")
print(f"  fused vs manual_tf32 max err: {(q_fused - q_manual_tf32).abs().max():.4e}")
print(f"  manual_tf32 vs manual_fp32 max err: {(q_manual_tf32 - q_manual_fp32).abs().max():.4e}")

# Now check: what if we disable TF32 in tl.dot by using allow_tf32=False?
# We can't directly - but let's check if the error pattern is consistent with TF32 rounding
print(f"\nError pattern: is fused closer to TF32 or FP32?")
err_vs_tf32 = (q_fused - q_manual_tf32).abs().mean()
err_vs_fp32 = (q_fused - q_manual_fp32).abs().mean()
print(f"  Mean err vs TF32: {err_vs_tf32:.4e}")
print(f"  Mean err vs FP32: {err_vs_fp32:.4e}")
if err_vs_tf32 < err_vs_fp32:
    print("  => Fused is closer to TF32 (expected for tl.dot)")
else:
    print("  => Fused is closer to FP32")
