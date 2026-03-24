#!/usr/bin/env python3
"""JSON-based wrapper for FlashAttention benchmark comparison."""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from compare_benchmark_json import main as compare_main


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare FlashAttention benchmark JSON outputs"
    )
    parser.add_argument("--baseline", required=True, help="Baseline JSON path")
    parser.add_argument("--candidate", required=True, help="Candidate JSON path")
    args = parser.parse_args()
    return compare_main()


if __name__ == "__main__":
    raise SystemExit(main())
