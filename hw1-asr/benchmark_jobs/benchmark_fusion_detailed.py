#!/usr/bin/env python3
"""Legacy wrapper for DecoderRMSNormQKV detailed benchmarking."""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from benchmark_with_flags import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Detailed fusion benchmark wrapper")
    parser.add_argument("folder", type=str, help="Folder name to benchmark")
    parser.add_argument("--audio", type=str, help="Path to test audio file")
    parser.add_argument("--runs", type=int, default=3, help="Number of profiling runs")
    parser.add_argument("--json-out", type=str, help="Optional path to save JSON results")
    args = parser.parse_args()

    legacy_argv = [
        args.folder,
        "--mode",
        "detailed",
        "--warmup",
        "1",
        "--runs",
        str(args.runs),
        "--linear-backend",
        "cublas",
        "--linear-autotune",
        "0",
        "--flash-attn-fusion",
        "0",
        "--decoder-qkv-fusion",
        "1" if args.folder == "glm_asr_triton_template" else "0",
    ]
    if args.audio:
        legacy_argv.extend(["--audio", args.audio])
    if args.json_out:
        legacy_argv.extend(["--json-out", args.json_out])
    return run(legacy_argv)


if __name__ == "__main__":
    raise SystemExit(main())
