#!/usr/bin/env python3
"""Correctness checks for FlashAttention fused vs unfused paths."""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from benchmark_utils import clear_folder_modules, folder_path


def load_attention_module():
    path = folder_path("glm_asr_triton_template")
    if path not in sys.path:
        sys.path.insert(0, path)
    clear_folder_modules()
    import attention

    return attention


def build_mask(
    batch: int,
    seq_q: int,
    seq_k: int,
    kind: Optional[str],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if kind is None:
        return None
    if kind == "broadcast_tail_mask":
        mask = torch.zeros((batch, 1, seq_q, seq_k), dtype=torch.float32, device=device)
        mask[..., seq_k // 2 :] = -1e9
        return mask
    raise ValueError(f"Unknown mask kind: {kind}")


def run_case(attention, case: Dict[str, object]) -> bool:
    device = torch.device("cuda")
    q = torch.randn(
        case["batch"],
        case["heads"],
        case["seq_q"],
        case["head_dim"],
        device=device,
        dtype=torch.float32,
    )
    k = torch.randn(
        case["batch"],
        case["heads"],
        case["seq_k"],
        case["head_dim"],
        device=device,
        dtype=torch.float32,
    )
    v = torch.randn(
        case["batch"],
        case["heads"],
        case["seq_k"],
        case["head_dim"],
        device=device,
        dtype=torch.float32,
    )
    mask = build_mask(
        case["batch"], case["seq_q"], case["seq_k"], case["mask_kind"], device
    )

    out_fused = attention._scaled_dot_product_attention_fused(
        q, k, v, mask, case["is_causal"]
    )
    out_unfused = attention._scaled_dot_product_attention_unfused(
        q, k, v, mask, case["is_causal"]
    )

    diff = (out_fused - out_unfused).abs()
    max_diff = float(diff.max().item())
    ref_scale = max(float(out_unfused.abs().max().item()), 1.0)
    rel = max_diff / ref_scale
    passed = rel < 1e-3
    status = "PASS" if passed else "FAIL"

    print(
        f"{case['name']:30s} max|diff|={max_diff:9.3e}  rel={rel:9.3e}  [{status}]"
    )
    return passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FlashAttention correctness testing")

    torch.manual_seed(42)
    attention = load_attention_module()

    cases = [
        {
            "name": "encoder_nonpow2",
            "batch": 1,
            "heads": 20,
            "seq_q": 187,
            "seq_k": 187,
            "head_dim": 64,
            "is_causal": False,
            "mask_kind": None,
        },
        {
            "name": "decoder_prefill_causal",
            "batch": 1,
            "heads": 16,
            "seq_q": 59,
            "seq_k": 59,
            "head_dim": 128,
            "is_causal": True,
            "mask_kind": None,
        },
        {
            "name": "decoder_broadcast_mask",
            "batch": 1,
            "heads": 16,
            "seq_q": 1,
            "seq_k": 50,
            "head_dim": 128,
            "is_causal": False,
            "mask_kind": "broadcast_tail_mask",
        },
        {
            "name": "large_seq_fallback",
            "batch": 1,
            "heads": 4,
            "seq_q": 32,
            "seq_k": 300,
            "head_dim": 128,
            "is_causal": False,
            "mask_kind": None,
        },
        {
            "name": "large_dim_fallback",
            "batch": 1,
            "heads": 4,
            "seq_q": 16,
            "seq_k": 16,
            "head_dim": 320,
            "is_causal": False,
            "mask_kind": None,
        },
    ]

    print("=" * 90)
    print("FlashAttention Correctness Test")
    print("=" * 90)

    all_pass = True
    for case in cases:
        all_pass = run_case(attention, case) and all_pass

    print("\nOverall:", "ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
