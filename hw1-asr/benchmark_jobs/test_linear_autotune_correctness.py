#!/usr/bin/env python3
"""Correctness checks for Triton Linear fixed-tile vs autotuned paths."""

from __future__ import annotations

import math
import os
import sys
from typing import List, Tuple

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from benchmark_utils import clear_folder_modules, folder_path


def make_linear(in_features: int, out_features: int):
    if folder_path("glm_asr_triton_template") not in sys.path:
        sys.path.insert(0, folder_path("glm_asr_triton_template"))
    clear_folder_modules()
    from layers import Linear

    linear = Linear(in_features, out_features, bias=True)
    scale = 1.0 / math.sqrt(in_features)
    linear.weight = (
        torch.randn(out_features, in_features, device="cuda", dtype=torch.float32) * scale
    )
    linear.bias_param = torch.randn(out_features, device="cuda", dtype=torch.float32) * 0.01
    return Linear, linear


def compare_outputs(name: str, shape: Tuple[int, ...], out_features: int) -> bool:
    in_features = shape[-1]
    Linear, linear = make_linear(in_features, out_features)
    x = torch.randn(shape, device="cuda", dtype=torch.float32) * 0.1

    torch.cuda.synchronize()
    Linear.BACKEND = "triton"
    Linear.USE_AUTOTUNE = False
    out_fixed = linear(x)

    torch.cuda.synchronize()
    Linear.BACKEND = "triton"
    Linear.USE_AUTOTUNE = True
    out_auto = linear(x)

    torch.cuda.synchronize()
    Linear.BACKEND = "cublas"
    Linear.USE_AUTOTUNE = False
    out_ref = linear(x)

    diff_auto_fixed = (out_auto - out_fixed).abs()
    diff_auto_ref = (out_auto - out_ref).abs()
    ref_scale = max(float(out_ref.abs().max().item()), 1.0)

    max_auto_fixed = float(diff_auto_fixed.max().item())
    max_auto_ref = float(diff_auto_ref.max().item())
    rel_auto_fixed = max_auto_fixed / ref_scale
    rel_auto_ref = max_auto_ref / ref_scale
    passed = rel_auto_fixed < 5e-2 and rel_auto_ref < 5e-2

    status = "PASS" if passed else "FAIL"
    print(
        f"{name:26s} shape={shape!s:18s} out={out_features:5d}  "
        f"max|auto-fixed|={max_auto_fixed:8.4f}  rel={rel_auto_fixed:7.4f}  "
        f"max|auto-ref|={max_auto_ref:8.4f}  rel={rel_auto_ref:7.4f}  [{status}]"
    )
    return passed


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for linear autotune correctness testing")

    torch.manual_seed(42)
    cases: List[Tuple[str, Tuple[int, ...], int]] = [
        ("decode_mlp", (1, 1, 3584), 18944),
        ("prefill_mlp", (1, 59, 3584), 18944),
        ("audio_encoder_fc1", (1, 187, 1280), 5120),
        ("audio_attention_proj", (2, 17, 1280), 1280),
    ]

    print("=" * 110)
    print("Linear Autotune Correctness Test")
    print("=" * 110)

    all_pass = True
    for name, shape, out_features in cases:
        all_pass = compare_outputs(name, shape, out_features) and all_pass

    print("\nOverall:", "ALL PASS" if all_pass else "SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
