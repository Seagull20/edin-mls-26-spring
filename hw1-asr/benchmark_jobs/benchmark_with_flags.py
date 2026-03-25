#!/usr/bin/env python3
"""Flag-driven Triton benchmark runner for end-to-end and detailed profiling."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark_detailed import detailed_profile_torch, print_summary
from benchmark_jobs.benchmark_utils import (
    build_torch_inputs,
    get_generate_fn,
    load_audio,
    load_triton_model,
    normalize_flags,
    to_builtin,
    warmup_detailed_model,
)


def benchmark_e2e(
    folder: str,
    flags: Dict[str, Any],
    audio_path: str | None,
    warmup: int,
    runs: int,
) -> Dict[str, Any]:
    """Run end-to-end transcription benchmarking with the requested flag set."""
    import torch
    from benchmark_student import check_transcription, decode_output

    audio_array, expected_text, duration, resolved_audio_path = load_audio(audio_path)
    _, model, processor = load_triton_model(folder, flags)
    input_features, input_ids, input_features_mask = build_torch_inputs(
        processor, audio_array
    )
    generate_fn = get_generate_fn(model)

    print("=" * 80)
    print("GLM-ASR Flag-Driven Benchmark")
    print("=" * 80)
    print(f"Folder: {folder}")
    print(f"Audio: {resolved_audio_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Flags: {json.dumps(flags, sort_keys=True)}")
    print(f"Generate function: {generate_fn.__name__}")

    print(f"\nWarmup ({warmup} runs)...")
    for _ in range(warmup):
        with torch.no_grad():
            try:
                _ = generate_fn(
                    input_features,
                    input_ids=input_ids,
                    input_features_mask=input_features_mask,
                    max_new_tokens=100,
                    temperature=1.0,
                    top_k=1,
                )
            except TypeError:
                _ = generate_fn(
                    input_features,
                    input_ids=input_ids,
                    max_new_tokens=100,
                    temperature=1.0,
                    top_k=1,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print(f"Benchmarking ({runs} runs)...")
    times_ms = []
    tokens = 0
    output = None
    for idx in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            try:
                output = generate_fn(
                    input_features,
                    input_ids=input_ids,
                    input_features_mask=input_features_mask,
                    max_new_tokens=100,
                    temperature=1.0,
                    top_k=1,
                )
            except TypeError:
                output = generate_fn(
                    input_features,
                    input_ids=input_ids,
                    max_new_tokens=100,
                    temperature=1.0,
                    top_k=1,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        tokens = int(output.shape[1] - input_ids.shape[1])
        print(f"  Run {idx + 1}: {elapsed_ms:.1f}ms ({tokens} tokens)")

    if output is None:
        raise RuntimeError("Benchmark did not produce any output tensor")

    transcription = decode_output(output.detach().cpu().numpy(), processor)
    passed = True
    accuracy = None
    if expected_text != "[synthetic]":
        passed, accuracy = check_transcription(transcription, expected_text)

    results = {
        "mean": float(np.mean(times_ms)),
        "std": float(np.std(times_ms)),
        "min": float(np.min(times_ms)),
        "max": float(np.max(times_ms)),
        "tokens": tokens,
        "speed_ms_per_token": float(np.mean(times_ms) / max(tokens, 1)),
        "transcription": transcription,
        "expected_text": expected_text,
        "accuracy": None if accuracy is None else float(accuracy),
        "status": "PASS" if passed else "FAIL",
        "times_ms": [float(x) for x in times_ms],
    }

    print("\nRESULTS")
    print("=" * 80)
    print(f"Time: {results['mean']:.1f}ms (+/- {results['std']:.1f}ms)")
    print(f"Tokens: {results['tokens']}")
    print(f"Speed: {results['speed_ms_per_token']:.2f}ms/token")
    print(f"Transcription: {results['transcription']}")
    if accuracy is not None:
        print(f"Accuracy: {results['accuracy'] * 100:.1f}%")
    print(f"Status: {results['status']}")

    payload = {
        "folder": folder,
        "mode": "e2e",
        "warmup": warmup,
        "runs": runs,
        "audio_path": resolved_audio_path,
        "flags": flags,
        "results": results,
    }
    return payload


def benchmark_detailed(
    folder: str,
    flags: Dict[str, Any],
    audio_path: str | None,
    warmup: int,
    runs: int,
) -> Dict[str, Any]:
    """Run detailed operator profiling with the requested flag set."""
    import torch

    audio_array, _, duration, resolved_audio_path = load_audio(audio_path)
    _, model, processor = load_triton_model(folder, flags)
    input_features, input_ids, input_features_mask = build_torch_inputs(
        processor, audio_array
    )

    print("=" * 80)
    print("GLM-ASR Detailed Flag-Driven Benchmark")
    print("=" * 80)
    print(f"Folder: {folder}")
    print(f"Audio: {resolved_audio_path}")
    print(f"Duration: {duration:.2f}s")
    print(f"Flags: {json.dumps(flags, sort_keys=True)}")
    print(f"Warmup runs: {warmup}")
    print(f"Profile runs: {runs}")

    with torch.inference_mode():
        print("\nWarming up model...")
        for _ in range(warmup):
            warmup_detailed_model(model, input_features, input_ids)
        results = detailed_profile_torch(
            model,
            input_features,
            input_ids,
            input_features_mask,
            num_runs=runs,
        )

    print_summary(results)

    payload = {
        "folder": folder,
        "mode": "detailed",
        "warmup": warmup,
        "runs": runs,
        "audio_path": resolved_audio_path,
        "flags": flags,
        "results": to_builtin(results),
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flag-driven GLM-ASR benchmark runner")
    parser.add_argument("folder", help="Triton folder to benchmark")
    parser.add_argument(
        "--mode",
        choices=("e2e", "detailed"),
        default="e2e",
        help="Benchmark mode",
    )
    parser.add_argument("--audio", help="Optional audio path")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs")
    parser.add_argument(
        "--linear-backend",
        default="cublas",
        help="Linear backend: cublas, triton, or torch",
    )
    parser.add_argument(
        "--linear-autotune",
        default="0",
        help="Enable Triton linear autotune (0/1)",
    )
    parser.add_argument(
        "--flash-attn-fusion",
        default="0",
        help="Enable FlashAttention fusion (0/1)",
    )
    parser.add_argument(
        "--decoder-qkv-fusion",
        default="0",
        help="Enable DecoderRMSNormQKV fusion (0/1)",
    )
    parser.add_argument("--json-out", help="Optional path to save JSON results")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    flags = normalize_flags(
        args.linear_backend,
        args.linear_autotune,
        args.flash_attn_fusion,
        args.decoder_qkv_fusion,
    )

    if args.mode == "e2e":
        payload = benchmark_e2e(args.folder, flags, args.audio, args.warmup, args.runs)
    else:
        payload = benchmark_detailed(
            args.folder, flags, args.audio, args.warmup, args.runs
        )

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    print("\nBENCHMARK_JSON=" + json.dumps(to_builtin(payload)))
    return 0 if payload["results"].get("status", "PASS") == "PASS" else 1


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
