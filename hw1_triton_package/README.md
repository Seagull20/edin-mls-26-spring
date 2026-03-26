# Reproducibility Package: GPU Kernel Optimization for GLM-ASR Inference

This package contains the minimal set of files needed to reproduce the benchmark results presented in the accompanying report. It allows reviewers to independently verify end-to-end latency, component-level profiling, and transcription correctness for both the baseline and optimized systems.

## Prerequisites

- NVIDIA GPU with CUDA support (results in the report were collected on RTX 5090)
- Python 3.10+ with PyTorch, Triton, and `transformers` installed
- Model weights are downloaded automatically from `zai-org/GLM-ASR-Nano-2512` on first run (requires internet access)

If using the provided conda environment:

```bash
conda activate mls
```

## Package Structure

```
hw1_triton_package/
  benchmark.sh              # End-to-end benchmark (wall-clock + correctness)
  benchmark_detailed.sh     # Component-level operator profiling
  benchmark_student.py      # Python driver for benchmark.sh
  benchmark_detailed.py     # Python driver for benchmark_detailed.sh
  test_audio.wav            # Reference audio input (coursework-provided)
  test_audio.txt            # Expected transcription
  glm_asr_triton_example/   # Provided baseline implementation
  glm_asr_triton_template/  # Optimized implementation (this work)
```

### Key Source Files

| File | Role |
|---|---|
| `glm_asr_triton_template/attention.py` | FlashAttention fused kernel (`flash_attention_fused_kernel`) |
| `glm_asr_triton_template/layers.py` | All custom Triton kernels, `DecoderRMSNormQKV` fusion class, autotune configurations |
| `glm_asr_triton_template/model.py` | Model integration with optimization flags |
| `glm_asr_triton_example/` | Unmodified baseline; same kernels without fusion or autotuning |

## Running Benchmarks

### 1. End-to-End Benchmark (Report Table 4)

Measures total inference latency and checks transcription correctness against the reference output.

```bash
# Baseline system
./benchmark.sh glm_asr_triton_example

# Optimized system (all optimizations enabled by default)
./benchmark.sh glm_asr_triton_template
```

**What to look for:** Wall-clock time (ms), standard deviation, and PASS/FAIL correctness status. The report claims 11.1% combined speedup with 100% transcription accuracy.

### 2. Component-Level Profiling (Report Sections 3 and 4)

Breaks down latency into pipeline stages: audio encoder, multi-modal projector, decoder prefill, and per-step autoregressive decoding.

```bash
# Baseline profiling
./benchmark_detailed.sh glm_asr_triton_example

# Optimized profiling
./benchmark_detailed.sh glm_asr_triton_template
```

**What to look for:** The output prints timing for each pipeline phase and a summary table showing the percentage contribution of each component. The report uses these breakdowns to identify the decoder autoregressive loop as the dominant bottleneck and to justify the memory-bound classification via arithmetic intensity analysis.

Optional flags:

```bash
./benchmark_detailed.sh glm_asr_triton_template --runs 5    # More runs for tighter statistics
./benchmark_detailed.sh glm_asr_triton_template --nsys       # Nsight Systems trace (requires nsys)
```

## Mapping Benchmarks to Report Claims

| Report Claim | How to Verify |
|---|---|
| Baseline latency ~600 ms (Table 4) | `./benchmark.sh glm_asr_triton_example` |
| Combined optimized latency ~534 ms, +11.1% speedup (Table 4) | `./benchmark.sh glm_asr_triton_template` |
| All configurations pass correctness (Table 4) | Check PASS/FAIL output from both commands above |
| Decoder decode dominates wall-clock time (Section 3) | Compare component times from `./benchmark_detailed.sh` on either variant |
| Decode-phase kernels are strongly memory-bound, AI ~0.5 (Section 4) | Component profiling shows decode-step times; the low FLOPs/byte ratio follows from the M=1 working set detailed in the report |

## Optimization Switches

The three optimizations in `glm_asr_triton_template` are controlled via environment variables, read at import time in `glm_asr_triton_template/__init__.py`:

| Environment Variable | Default | Effect |
|---|---|---|
| `FLASH_ATTN_FUSION` | `0` (off) | Enables FlashAttention fused kernel, replacing the 3-kernel attention path |
| `DECODER_QKV_FUSION` | `0` (off) | Enables RMSNorm+QKV fusion, replacing separate RMSNorm + 3 Q/K/V projections |
| `LINEAR_AUTOTUNE` | `0` (off) | Enables `@triton.autotune` tile-size selection for linear kernels |
| `LINEAR_BACKEND` | `cublas` | Linear layer backend: `cublas` (PyTorch/cuBLAS matmul) or `triton` (custom Triton kernel) |

### Ablation Examples (Report Table 4)

```bash
# FlashAttention only
FLASH_ATTN_FUSION=1 ./benchmark.sh glm_asr_triton_template

# RMSNorm+QKV fusion only
DECODER_QKV_FUSION=1 ./benchmark.sh glm_asr_triton_template

# Both optimizations
FLASH_ATTN_FUSION=1 DECODER_QKV_FUSION=1 ./benchmark.sh glm_asr_triton_template

# All optimizations disabled (default, should match baseline performance)
./benchmark.sh glm_asr_triton_template
```

The same environment variables work with `benchmark_detailed.sh`.

## Notes

- The baseline (`glm_asr_triton_example`) uses cuBLAS for linear layers by default, matching the report's comparison protocol.
- Absolute latency numbers depend on GPU model, driver version, and thermal state. Relative speedup percentages are the meaningful comparison metric.
- Test audio is a short (~3 s) coursework-provided sample; results reflect this specific input length.
