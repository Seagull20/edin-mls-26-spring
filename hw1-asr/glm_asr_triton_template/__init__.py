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

layers.Linear.BACKEND = "cublas"
layers.MLP.FUSED = False
layers.EncoderMLP.FUSED = False
layers.DecoderRMSNormQKV.FUSED = False

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
