#!/usr/bin/env python3
"""
Detailed benchmark launcher with explicit operator toggles.

This keeps the comparison focused on decoder RMSNorm+QKV fusion by forcing
the relevant class flags before the model is imported from a target folder.
"""

import argparse
import json
import os
import struct
import sys
import wave
from typing import Any

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark_detailed import detailed_profile_torch, print_summary


MODULE_NAMES = ["weight_loader", "model", "layers", "attention", "rope", "conv"]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_audio(audio_path: str) -> np.ndarray:
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
        else:
            audio_array = np.zeros(n_frames, dtype=np.float32)

        if n_channels > 1:
            audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)

    print(f"Audio: {len(audio_array) / sr:.2f}s @ {sr}Hz")
    return audio_array


def clear_folder_modules() -> None:
    for mod_name in list(sys.modules.keys()):
        if mod_name in MODULE_NAMES:
            del sys.modules[mod_name]


def configure_layers(folder_path: str, decoder_qkv_fused: bool) -> None:
    sys.path.insert(0, folder_path)
    clear_folder_modules()

    import layers

    layers.Linear.BACKEND = "cublas"
    if hasattr(layers, "MLP"):
        layers.MLP.FUSED = False
    if hasattr(layers, "EncoderMLP"):
        layers.EncoderMLP.FUSED = False
    if hasattr(layers, "DecoderRMSNormQKV"):
        layers.DecoderRMSNormQKV.FUSED = decoder_qkv_fused


def load_model(folder: str, folder_path: str):
    print(f"\nLoading model from {folder}...")
    print("Benchmark toggles:")
    print("  Linear.BACKEND = cublas")
    print("  MLP.FUSED = False")
    print("  EncoderMLP.FUSED = False")
    print(f"  DecoderRMSNormQKV.FUSED = {folder == 'glm_asr_triton_template'}")

    configure_layers(folder_path, decoder_qkv_fused=(folder == "glm_asr_triton_template"))

    from weight_loader import load_model_from_hf

    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")
    return model, processor


def build_inputs(processor, audio_array):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(processor, "apply_transcription_request"):
        inputs = processor.apply_transcription_request(audio_array)
        input_features = inputs.input_features.to(device=device, dtype=torch.float32)
        input_ids = inputs.input_ids.to(device=device, dtype=torch.int64)
        input_features_mask = None
        if hasattr(inputs, "input_features_mask") and inputs.input_features_mask is not None:
            input_features_mask = inputs.input_features_mask.to(
                device=device, dtype=torch.float32
            )
    else:
        features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
        )
        input_features = features["input_features"].to(device=device, dtype=torch.float32)
        input_ids = torch.tensor(
            [[59253, 10, 59261] + [59260] * 100 + [59262, 59253, 10, 9249, 70891, 419, 7122, 1119, 1467, 59254, 10]],
            dtype=torch.int64,
            device=device,
        )
        input_features_mask = None

    print(f"Input features shape: {input_features.shape}")
    print(f"Input IDs shape: {input_ids.shape}")
    return input_features, input_ids, input_features_mask


def warmup_model(model, input_features, input_ids) -> None:
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
            combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

    hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
    logits = model.lm_head(hidden_states[:, -1:, :])
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    next_embed = embed_tokens(next_token)
    _ = model.text_decoder(inputs_embeds=next_embed)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description="Detailed fusion benchmark")
    parser.add_argument("folder", type=str, help="Folder name to benchmark")
    parser.add_argument("--audio", type=str, help="Path to test audio file")
    parser.add_argument("--runs", type=int, default=3, help="Number of profiling runs")
    parser.add_argument("--json-out", type=str, help="Optional path to save JSON results")
    args = parser.parse_args()

    print("=" * 70)
    print("GLM-ASR Fusion Benchmark")
    print("=" * 70)

    audio_path = args.audio or os.path.join(PROJECT_ROOT, "test_audio.wav")

    print("\nLoading test audio...")
    audio_array = load_audio(audio_path)

    folder_path = os.path.join(PROJECT_ROOT, args.folder)
    model, processor = load_model(args.folder, folder_path)
    input_features, input_ids, input_features_mask = build_inputs(processor, audio_array)

    import torch

    with torch.inference_mode():
        print("\nWarming up model...")
        warmup_model(model, input_features, input_ids)
        results = detailed_profile_torch(
            model,
            input_features,
            input_ids,
            input_features_mask,
            num_runs=args.runs,
        )

    print_summary(results)

    payload = {
        "folder": args.folder,
        "runs": args.runs,
        "audio_path": audio_path,
        "flags": {
            "linear_backend": "cublas",
            "mlp_fused": False,
            "encoder_mlp_fused": False,
            "decoder_qkv_fused": args.folder == "glm_asr_triton_template",
        },
        "results": _to_builtin(results),
    }

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    print("\nBENCHMARK_JSON=" + json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
