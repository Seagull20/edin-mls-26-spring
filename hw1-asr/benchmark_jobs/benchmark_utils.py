#!/usr/bin/env python3
"""Shared helpers for flag-driven Triton benchmarks."""

from __future__ import annotations

import importlib
import os
import struct
import sys
import wave
from typing import Any, Dict, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODULE_NAMES = ["weight_loader", "model", "layers", "attention", "rope", "conv"]
MODEL_NAME = "zai-org/GLM-ASR-Nano-2512"


def parse_bool(value: Any) -> bool:
    """Convert common truthy/falsey values into bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "on"}:
        return True
    if value_str in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def normalize_flags(
    linear_backend: str,
    linear_autotune: Any,
    flash_attn_fusion: Any,
    decoder_qkv_fusion: Any,
) -> Dict[str, Any]:
    """Normalize benchmark flags into a JSON-friendly dict."""
    backend = str(linear_backend).strip().lower()
    if backend not in {"torch", "cublas", "triton"}:
        raise ValueError(f"Unsupported linear backend: {linear_backend}")
    return {
        "linear_backend": backend,
        "linear_autotune": parse_bool(linear_autotune),
        "flash_attn_fusion": parse_bool(flash_attn_fusion),
        "decoder_qkv_fusion": parse_bool(decoder_qkv_fusion),
        "mlp_fused": False,
        "encoder_mlp_fused": False,
    }


def set_env_flags(flags: Dict[str, Any]) -> None:
    """Mirror the active flag set into env vars for consistency/debugging."""
    os.environ["LINEAR_BACKEND"] = flags["linear_backend"]
    os.environ["LINEAR_AUTOTUNE"] = "1" if flags["linear_autotune"] else "0"
    os.environ["FLASH_ATTN_FUSION"] = "1" if flags["flash_attn_fusion"] else "0"
    os.environ["DECODER_QKV_FUSION"] = "1" if flags["decoder_qkv_fusion"] else "0"


def clear_folder_modules() -> None:
    """Drop per-folder top-level modules so a new config can be imported cleanly."""
    for mod_name in list(sys.modules.keys()):
        if mod_name in MODULE_NAMES:
            del sys.modules[mod_name]


def folder_path(folder: str) -> str:
    path = os.path.join(PROJECT_ROOT, folder)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Folder not found: {path}")
    return path


def configure_folder_modules(folder: str, flags: Dict[str, Any]):
    """Import the target folder's modules and apply the requested flags."""
    path = folder_path(folder)
    if path not in sys.path:
        sys.path.insert(0, path)

    clear_folder_modules()
    set_env_flags(flags)
    importlib.invalidate_caches()

    layers = importlib.import_module("layers")
    attention = importlib.import_module("attention")

    layers.Linear.BACKEND = flags["linear_backend"]
    layers.Linear.USE_AUTOTUNE = flags["linear_autotune"]

    if hasattr(layers, "MLP"):
        layers.MLP.FUSED = False
    if hasattr(layers, "EncoderMLP"):
        layers.EncoderMLP.FUSED = False
    if hasattr(layers, "DecoderRMSNormQKV"):
        layers.DecoderRMSNormQKV.FUSED = flags["decoder_qkv_fusion"]
    if hasattr(attention, "USE_FLASH_ATTENTION_FUSION"):
        attention.USE_FLASH_ATTENTION_FUSION = flags["flash_attn_fusion"]

    return path, layers, attention


def load_triton_model(folder: str, flags: Dict[str, Any]):
    """Load a Triton model after applying the requested runtime flags."""
    path, _, _ = configure_folder_modules(folder, flags)
    from weight_loader import load_model_from_hf

    model, processor = load_model_from_hf(MODEL_NAME)
    return path, model, processor


def load_audio(audio_path: Optional[str]) -> Tuple[np.ndarray, str, float, str]:
    """Load the benchmark audio and return array, expected text, duration, and path."""
    from benchmark_student import load_test_audio

    resolved_path = audio_path
    if resolved_path is None:
        resolved_path = os.path.join(PROJECT_ROOT, "test_audio.wav")
    audio_array, expected_text, duration = load_test_audio(resolved_path)
    return audio_array, expected_text, duration, resolved_path


def build_torch_inputs(processor, audio_array):
    """Prepare torch inputs for Triton model benchmarking."""
    import torch
    from benchmark_student import prepare_inputs_torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return prepare_inputs_torch(audio_array, processor, device)


def get_generate_fn(model):
    """Select the best available generate entrypoint for the loaded model."""
    generate_fn = model.generate
    if hasattr(model, "generate_v8b"):
        generate_fn = model.generate_v8b
    elif hasattr(model, "generate_v8"):
        generate_fn = model.generate_v8
    elif hasattr(model, "generate_v6"):
        generate_fn = model.generate_v6
    return generate_fn


def warmup_detailed_model(model, input_features, input_ids) -> None:
    """Prime the detailed-profile path to avoid measuring one-time setup."""
    import torch

    audio_features = model.audio_encoder(input_features)
    projected = model.multi_modal_projector(audio_features)

    embed_tokens = model.text_decoder.embed_tokens
    text_embeds = embed_tokens(input_ids)
    audio_token_id = 59260
    audio_mask = input_ids == audio_token_id
    combined_embeds = text_embeds.clone()

    if torch.any(audio_mask):
        audio_positions = torch.where(audio_mask[0])[0]
        num_audio_tokens = int(audio_positions.numel())
        if num_audio_tokens <= projected.shape[1]:
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[
                0, :num_audio_tokens
            ]

    hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    next_embed = embed_tokens(next_token)
    _ = model.text_decoder(inputs_embeds=next_embed)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def to_builtin(value: Any) -> Any:
    """Convert numpy scalars/containers into plain JSON-compatible types."""
    if isinstance(value, dict):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_wav_seconds(audio_path: str) -> float:
    """Read wav duration for logging when a direct path is used."""
    with wave.open(audio_path, "rb") as wav:
        sr = wav.getframerate()
        n_frames = wav.getnframes()
    return n_frames / float(sr)


def load_audio_array_from_wav(audio_path: str) -> np.ndarray:
    """Load a wav file using stdlib, matching existing benchmark conventions."""
    with wave.open(audio_path, "rb") as wav:
        sr = wav.getframerate()
        n_frames = wav.getnframes()
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        raw_data = wav.readframes(n_frames)

    if sample_width == 2:
        audio_array = np.array(
            struct.unpack(f"<{n_frames * n_channels}h", raw_data),
            dtype=np.float32,
        )
        audio_array = audio_array / 32768.0
    elif sample_width == 4:
        audio_array = np.array(
            struct.unpack(f"<{n_frames * n_channels}i", raw_data),
            dtype=np.float32,
        )
        audio_array = audio_array / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if n_channels > 1:
        audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)

    if sr != 16000:
        raise ValueError(f"Expected 16kHz wav for direct loader, got {sr}Hz")
    return audio_array
