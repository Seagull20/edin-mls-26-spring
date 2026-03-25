"""
Template Baseline - Triton Student Assignment
Performance: TBD (Torch baseline with Triton kernels available)

Key Characteristics:
- Pure Torch tensor operations
- Triton kernels for core ops (student TODOs)
- Flash Attention-style fusion (toggle via USE_FLASH_ATTENTION_FUSION)
"""

import os
import sys

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from . import layers

_linear_backend_env = os.environ.get("LINEAR_BACKEND", "cublas")
layers.Linear.BACKEND = _linear_backend_env

_linear_autotune_env = os.environ.get("LINEAR_AUTOTUNE", "0")
layers.Linear.USE_AUTOTUNE = _linear_autotune_env in ("1", "true", "True", "yes")

layers.MLP.FUSED = False
layers.EncoderMLP.FUSED = False
_decoder_qkv_fusion_env = os.environ.get("DECODER_QKV_FUSION", "1")
layers.DecoderRMSNormQKV.FUSED = _decoder_qkv_fusion_env in (
    "1",
    "true",
    "True",
    "yes",
)

from . import attention

# --- Flash Attention Fusion Switch ---
# Set to True to fuse score computation, softmax, and output into a single
# Triton kernel (FlashAttention-style). Set to False for the baseline 3-kernel
# approach. Can be overridden at runtime or by benchmark scripts via:
#   import glm_asr_triton_template as m; m.attention.USE_FLASH_ATTENTION_FUSION = True/False
_flash_attn_env = os.environ.get("FLASH_ATTN_FUSION", "1")
attention.USE_FLASH_ATTENTION_FUSION = _flash_attn_env in ("1", "true", "True", "yes")

from . import model
from . import rope
from . import conv
from . import weight_loader
