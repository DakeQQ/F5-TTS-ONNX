from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import sys
import math
from pathlib import Path
from typing import Any
import f5_tts
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from omegaconf import OmegaConf
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding
from vocos.feature_extractors import EncodecFeatures, FeatureExtractor
from Rewrite_F5_ONNX import rewrite_mish_subgraphs
from STFT_Process import STFT_Process
from f5_tts.model import CFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="F5-TTS model export",
        description="Export F5-TTS v0 or v1 to ONNX.",
        epilog="Authors: DakeQQ and contributors",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=logging.getLevelNamesMapping().keys(),
        help="Set the logging level. Default: INFO.",
    )
    parser.add_argument("--model_series", "--model-series", choices=("v0", "v1"), default="v1")
    parser.add_argument("--vocab_path", "--vocab-path", type=Path, help="Path to vocab.txt; defaults to the file beside the checkpoint.")
    parser.add_argument("--f5safetensor_path", "--f5-safetensor-path", type=Path, help="Path to the F5 safetensors checkpoint.")
    parser.add_argument("--omegacfg_path", "--omegacfg-path", type=Path, help="Path to a compatible F5 YAML config; auto-discovered by default.")
    parser.add_argument("--vocosmodel_dir", "--vocos-model-dir", type=Path, help="Directory containing the Vocos config and weights.")
    parser.add_argument("--preprocessmodel_path", "--preprocess-model-path", type=Path, help="Path for F5_Preprocess.onnx.")
    parser.add_argument("--transformermodel_path", "--transformer-model-path", type=Path, help="Path for F5_Transformer.onnx.")
    parser.add_argument("--decodermodel_path", "--decoder-model-path", type=Path, help="Path for F5_Decode.onnx.")
    parser.add_argument("--metadatamodel_path", "--metadata-model-path", type=Path, help="Path for F5_Metadata.onnx.")
    parser.add_argument("--decoder_output_type", "--decoder-output-type", choices=("f16", "f32", "i16"), default="f32")
    parser.add_argument("--prepro_input_type", "--prepro-input-type", choices=("f16", "f32", "i16"), default="f32")
    parser.add_argument("--onnx_opset_version", "--onnx-opset-version", type=int, default=20)
    parser.add_argument(
        "--onnx_add_metadata",
        "--onnx-add-metadata",
        action="store_true",
        help="Compatibility flag; metadata is always emitted by the refactored exporter.",
    )
    parser.add_argument("--fp16transformer", action="store_true", help="Export the Transformer in float16.")
    parser.add_argument("--testlang", default="zh", help="Language used by the post-export inference smoke test.")
    parser.add_argument("--seed", type=int, default=9527, help="ONNX Runtime random seed used by post-export inference.")
    parser.add_argument("--target_rms", "--target-rms", type=float, default=0.1)
    parser.add_argument("--nfe_step", "--nfe-step", type=int, default=32)
    parser.add_argument("--check_config", "--check-config", action="store_true", help="Validate the selected series, config, vocabulary, and checkpoint without exporting ONNX.")
    parser.add_argument("--skip_inference", "--skip-inference", action="store_true", help="Skip the post-export ONNX Runtime smoke test.")
    return parser.parse_args()


args = parse_args()
logging.basicConfig(level=args.loglevel.upper())

script_dir = Path(__file__).resolve().parent
onnx_folder = script_dir / "F5_ONNX"
downloads_folder = Path.home() / "Downloads"
F5_MODEL_SERIES = args.model_series
F5_MODEL_PROFILES = {
    "v0": (
        downloads_folder / "F5TTS_v0_Base" / "model_1200000.safetensors",
    ),
    "v1": (
        downloads_folder / "F5TTS_v1_Base" / "model_1250000.safetensors",
    ),
}


def resolve_default_checkpoint(model_series: str) -> Path:
    candidates = tuple(path.expanduser().resolve() for path in F5_MODEL_PROFILES[model_series])
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"No default {model_series} checkpoint was found; checked {list(candidates)}. "
            "Pass --f5safetensor_path explicitly."
        )
    return matches[0]


F5_checkpoint_path = (
    args.f5safetensor_path.expanduser().resolve()
    if args.f5safetensor_path
    else resolve_default_checkpoint(F5_MODEL_SERIES)
)
_OFFICIAL_CHECKPOINT_SERIES = {
    "model_1200000.safetensors": "v0",
    "model_1250000.safetensors": "v1",
}
checkpoint_series_hint = _OFFICIAL_CHECKPOINT_SERIES.get(F5_checkpoint_path.name.casefold())
if checkpoint_series_hint is not None and checkpoint_series_hint != F5_MODEL_SERIES:
    raise ValueError(
        f"Checkpoint {F5_checkpoint_path.name} is the official {checkpoint_series_hint} Base model, "
        f"but --model_series requested {F5_MODEL_SERIES}."
    )
vocos_model_path = str((args.vocosmodel_dir or downloads_folder / "vocos-mel-24khz").expanduser().resolve())

preprocess_path = (args.preprocessmodel_path or onnx_folder / "F5_Preprocess.onnx").expanduser().resolve()
transformer_path = (args.transformermodel_path or onnx_folder / "F5_Transformer.onnx").expanduser().resolve()
decode_path = (args.decodermodel_path or onnx_folder / "F5_Decode.onnx").expanduser().resolve()
metadata_path = (args.metadatamodel_path or preprocess_path.parent / "F5_Metadata.onnx").expanduser().resolve()
transformer_raw_path = transformer_path.with_name(f".{transformer_path.stem}.raw{transformer_path.suffix}")

onnx_model_Preprocess = str(preprocess_path)
onnx_model_Transformer = str(transformer_path)
onnx_model_Transformer_Raw = str(transformer_raw_path)
onnx_model_Decode = str(decode_path)
onnx_model_Metadata = str(metadata_path)

# Model Parameters
DYNAMIC_AXES = True                     # Default dynamic_axes is input audio length. Note, some providers only work for static axes.
NFE_STEP = args.nfe_step                # Fixed at export time and recorded in package metadata.
IN_SAMPLE_RATE = 24000                  # Public prompt-audio ONNX input rate.
OUT_SAMPLE_RATE = 24000                 # Public generated-waveform ONNX output rate.
_CLI_AUDIO_DTYPES = {"f16": "F16", "f32": "F32", "i16": "INT16"}
IN_AUDIO_DTYPE = _CLI_AUDIO_DTYPES[args.prepro_input_type]
OUT_AUDIO_DTYPE = _CLI_AUDIO_DTYPES[args.decoder_output_type]
MODEL_SAMPLE_RATE = 24000               # Native F5/Vocos sample rate; do not edit.
CFG_STRENGTH = 2.0                      # F5-TTS model setting
SWAY_COEFFICIENT = -1.0                 # F5-TTS model setting
TARGET_RMS = args.target_rms            # Root-mean-square target for quiet reference audio.
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT. It affects the generated audio length and speech speed.
if NFE_STEP < 1:
    raise ValueError("NFE_STEP must be >= 1.")
if TARGET_RMS <= 0:
    raise ValueError("target_rms must be positive.")

# STFT/ISTFT Settings
N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 1024                             # Number of FFT components for the STFT process
WINDOW_LENGTH = 1024                    # Length of windowing, edit it carefully.
WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT
MAX_SIGNAL_LENGTH = 4096                # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.

OPSET = args.onnx_opset_version
if OPSET < 17:
    raise ValueError("onnx_opset_version must be >= 17.")

_AUDIO_DTYPES = {"F16": torch.float16, "F32": torch.float32, "INT16": torch.int16}
use_fp16_transformer = args.fp16transformer
if IN_SAMPLE_RATE < 1 or OUT_SAMPLE_RATE < 1:
    raise ValueError("IN_SAMPLE_RATE and OUT_SAMPLE_RATE must be positive.")
if IN_AUDIO_DTYPE.upper() not in _AUDIO_DTYPES:
    raise ValueError(f"Unsupported IN_AUDIO_DTYPE={IN_AUDIO_DTYPE!r}; expected one of {tuple(_AUDIO_DTYPES)}.")
if OUT_AUDIO_DTYPE.upper() not in _AUDIO_DTYPES:
    raise ValueError(f"Unsupported OUT_AUDIO_DTYPE={OUT_AUDIO_DTYPE!r}; expected one of {tuple(_AUDIO_DTYPES)}.")


f5_package_paths = tuple(Path(path).resolve() for path in f5_tts.__path__)


def find_vocab_file(checkpoint_path: Path) -> Path:
    checkpoint_folder = checkpoint_path.parent
    candidates = sorted(
        path for path in checkpoint_folder.iterdir()
        if path.is_file() and "vocab" in path.name.casefold()
    )
    if not candidates:
        candidates = sorted(path for path in checkpoint_folder.glob("*.txt") if path.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one vocab file beside {checkpoint_path.name}, found {len(candidates)}: "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0].resolve()


def get_checkpoint_architecture(checkpoint_path: Path) -> tuple[int, int]:
    if checkpoint_path.suffix.casefold() != ".safetensors":
        raise ValueError(f"Automatic config discovery requires a .safetensors checkpoint: {checkpoint_path}")

    from safetensors import safe_open

    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        keys = checkpoint.keys()
        projection_keys = [key for key in keys if key.endswith("transformer.proj_out.weight")]
        block_indices = {
            int(key.split("transformer.transformer_blocks.", 1)[1].split(".", 1)[0])
            for key in keys
            if "transformer.transformer_blocks." in key
        }
        if len(projection_keys) != 1 or not block_indices:
            raise ValueError(f"Could not infer the F5 architecture from checkpoint keys in {checkpoint_path}.")
        hidden_size = checkpoint.get_slice(projection_keys[0]).get_shape()[1]
    return hidden_size, max(block_indices) + 1


def is_compatible_model_config(
    config_path: Path,
    hidden_size: int,
    depth: int,
    model_series: str,
) -> bool:
    config = OmegaConf.load(config_path)
    model = config.get("model")
    if model is None or model.get("backbone") != "DiT":
        return False
    arch = model.get("arch")
    if arch is None or arch.get("dim") != hidden_size or arch.get("depth") != depth:
        return False
    text_mask_padding = arch.get("text_mask_padding", True)
    pe_attn_head = arch.get("pe_attn_head")
    return (
        (model_series == "v0" and text_mask_padding is False and pe_attn_head == 1)
        or (model_series == "v1" and text_mask_padding is True and pe_attn_head is None)
    )


def find_model_config(checkpoint_path: Path, model_series: str) -> Path:
    local_configs = sorted(path.resolve() for path in checkpoint_path.parent.glob("*.yaml") if path.is_file())
    if local_configs:
        candidates = local_configs
    else:
        candidates = sorted({
            path.resolve()
            for package_path in f5_package_paths
            for path in (package_path / "configs").glob("*.yaml")
            if path.is_file()
        })

    hidden_size, depth = get_checkpoint_architecture(checkpoint_path)
    matches = [
        config_path
        for config_path in candidates
        if is_compatible_model_config(config_path, hidden_size, depth, model_series)
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one compatible {model_series} YAML config for {checkpoint_path.name}, "
            f"found {len(matches)} among {[path.name for path in candidates]}: "
            f"{[path.name for path in matches]}"
        )
    return matches[0]


if not F5_checkpoint_path.is_file():
    raise FileNotFoundError(f"F5 checkpoint was not found: {F5_checkpoint_path}")
if args.omegacfg_path is None:
    F5_config_path = find_model_config(F5_checkpoint_path, F5_MODEL_SERIES)
else:
    F5_config_path = args.omegacfg_path.expanduser().resolve()
    if not F5_config_path.is_file():
        raise FileNotFoundError(f"F5 config was not found: {F5_config_path}")
    hidden_size, depth = get_checkpoint_architecture(F5_checkpoint_path)
    if not is_compatible_model_config(F5_config_path, hidden_size, depth, F5_MODEL_SERIES):
        raise ValueError(
            f"Config {F5_config_path} does not match checkpoint architecture "
            f"(dim={hidden_size}, depth={depth}) and requested series {F5_MODEL_SERIES}."
        )
F5_safetensors_path  = str(F5_checkpoint_path)
vocab_file = (args.vocab_path.expanduser().resolve() if args.vocab_path else find_vocab_file(F5_checkpoint_path))
if not vocab_file.is_file():
    raise FileNotFoundError(f"F5 vocabulary was not found: {vocab_file}")
vocab_path = str(vocab_file)
print(
    f"[F5-TTS] series={F5_MODEL_SERIES}, checkpoint={F5_checkpoint_path.name}, "
    f"config={F5_config_path.name}, vocab={vocab_file.name}"
)


# Load the vocab.txt
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)

# Export-specific local replacements for the F5 transformer and Vocos decoder.
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        out = x.transpose(1, 2)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x / (torch.sqrt(variance + self.eps))
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight
        return x


class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None, modulation=None):
        if modulation is None:
            modulation = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.split(modulation, self.dim, dim=-1)
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNorm_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None, modulation=None):
        if modulation is None:
            modulation = self.linear(self.silu(emb))
        scale, shift = torch.split(modulation, self.dim, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class Attention(nn.Module):
    def __init__(
        self,
        processor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: int | None = None,
        context_pre_only: bool = False,
        qk_norm: str | None = None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only
        rope_permutation = torch.arange(dim_head, dtype=torch.int32).view(-1, 2).flip(-1).reshape(-1)
        self.register_buffer("rope_permutation", rope_permutation, persistent=False)
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)])

    def forward(self, x, c=None, mask=None, rope=None, rope_cos=None, rope_sin=None, c_rope=None):
        if c is not None:
            raise NotImplementedError("The standalone F5 export path only supports DiT self-attention.")
        return self.processor(self, x, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)

    def fuse_qkv(self, scale=1.0):
        """Collapse the separate to_q/to_k/to_v projections into one GEMM (to_qkv),
        folding the attention scale (head_dim ** -0.25) into the q & k weight rows.
        The weight-side scale fold is numerically equivalent within the export tolerance."""
        fused = nn.Linear(self.dim, 3 * self.inner_dim)
        fused.weight.data = torch.cat((self.to_q.weight.data * scale, self.to_k.weight.data * scale, self.to_v.weight.data), dim=0)
        fused.bias.data = torch.cat((self.to_q.bias.data * scale, self.to_k.bias.data * scale, self.to_v.bias.data), dim=0)
        self.to_qkv = fused
        del self.to_q, self.to_k, self.to_v


def rotate_half(x, permutation):
    return torch.index_select(x, -1, permutation)


def apply_rotary(x, rope_cos, rope_sin, permutation):
    return x * rope_cos + rotate_half(x, permutation) * rope_sin


class AttnProcessor:
    def __init__(self, head_dim, hidden_size, heads, pe_attn_head=None):
        if pe_attn_head is not None and not 1 <= pe_attn_head <= heads:
            raise ValueError(f"pe_attn_head must be between 1 and {heads}, got {pe_attn_head}.")
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.heads = heads
        self.heads_2 = heads + heads
        self.pe_attn_head = pe_attn_head

    def __call__(self, attn: Attention, x, mask=None, rope_cos=None, rope_sin=None):
        # One fused GEMM, then a single reshape splits the packed q/k stack from v across the head axis.
        qkv = attn.to_qkv(x).view(2, -1, 3 * self.heads, self.head_dim).transpose(1, 2)
        qk, value = torch.split(qkv, [self.heads_2, self.heads], dim=1)
        if self.pe_attn_head is None:
            qk = apply_rotary(qk, rope_cos, rope_sin, attn.rope_permutation)
            query, key = torch.split(qk, self.heads, dim=1)
        else:
            query, key = torch.split(qk, self.heads, dim=1)
            query = torch.cat((
                apply_rotary(query[:, :self.pe_attn_head], rope_cos, rope_sin, attn.rope_permutation),
                query[:, self.pe_attn_head:],
            ), dim=1)
            key = torch.cat((
                apply_rotary(key[:, :self.pe_attn_head], rope_cos, rope_sin, attn.rope_permutation),
                key[:, self.pe_attn_head:],
            ), dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2))
        if use_fp16_transformer:
            weights = torch.softmax(scores.float() * 100.0, dim=-1, dtype=torch.float32).half()
        else:
            weights = torch.softmax(scores, dim=-1)
        x = torch.matmul(weights, value).transpose(1, 2).reshape(2, -1, self.hidden_size)
        return attn.to_out[0](x)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1, qk_norm=None, pe_attn_head=None, attn_backend="torch", attn_mask_enabled=True):
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(dim_head, dim, heads, pe_attn_head=pe_attn_head),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope_cos=None, rope_sin=None, modulation=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t, modulation=modulation)
        attn_output = self.attn(x=norm, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + gate_msa * attn_output
        norm = self.ff_norm(x) * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm)
        x = x + gate_mlp * ff_output
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        self.mask_padding = mask_padding
        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = MAX_SIGNAL_LENGTH
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos).unsqueeze(0), persistent=False)
            self.text_blocks = nn.Sequential(*[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)])
        else:
            self.extra_modeling = False

    def forward(self, text, max_duration):
        text_mask = (text == 0).unsqueeze(-1) if self.mask_padding else None
        text = self.text_embed(torch.cat((text, torch.zeros_like(text)), dim=0))
        if self.extra_modeling:
            pos_idx = self.freqs_cis[:, :max_duration]
            text = text + pos_idx
            if text_mask is None:
                text = self.text_blocks(text)
            else:
                text = torch.where(text_mask, 0.0, text)
                for block in self.text_blocks:
                    text = block(text)
                    text = torch.where(text_mask, 0.0, text)
        text, text_drop = torch.split(text, [1, 1], dim=0)
        return text, text_drop


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, drop_audio_cond=False):
        x = self.proj(torch.cat((x, cond), dim=-1))
        return self.conv_pos_embed(x) + x


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        text_mask_padding=True,
        qk_norm=None,
        conv_layers=0,
        pe_attn_head=None,
        attn_backend="torch",
        attn_mask_enabled=False,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers)
        self.text_cond, self.text_uncond = None, None
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    pe_attn_head=pe_attn_head,
                    attn_backend=attn_backend,
                    attn_mask_enabled=attn_mask_enabled,
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNorm_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.time_activation = nn.SiLU()
        self.checkpoint_activations = checkpoint_activations
        self.initialize_weights()

    def initialize_weights(self):
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def fuse_time_projections(self):
        if hasattr(self, "time_modulation"):
            raise RuntimeError("Time modulation projections are already fused.")
        block_width = self.dim * 6
        final_width = self.dim * 2
        reference = self.transformer_blocks[0].attn_norm.linear.weight
        fused = nn.Linear(
            self.dim,
            block_width * self.depth + final_width,
            device=reference.device,
            dtype=reference.dtype,
        )
        offset = 0
        with torch.no_grad():
            for block in self.transformer_blocks:
                linear = block.attn_norm.linear
                next_offset = offset + block_width
                fused.weight[offset:next_offset].copy_(linear.weight)
                fused.bias[offset:next_offset].copy_(linear.bias)
                offset = next_offset
            linear = self.norm_out.linear
            fused.weight[offset:].copy_(linear.weight)
            fused.bias[offset:].copy_(linear.bias)
        self.time_modulation = fused
        self.time_modulation_split_sizes = [block_width] * self.depth + [final_width]
        for block in self.transformer_blocks:
            del block.attn_norm.linear
            del block.attn_norm.silu
        del self.norm_out.linear
        del self.norm_out.silu

    def forward(self, x, cond, cond_drop, time, rope_cos, rope_sin, mask=None):
        x = torch.cat((x, x), dim=0)
        cond = torch.cat((cond, cond_drop), dim=0)
        x = self.input_embed(x, cond)
        if hasattr(self, "time_modulation"):
            modulations = torch.split(
                self.time_modulation(self.time_activation(time)),
                self.time_modulation_split_sizes,
                dim=-1,
            )
            block_modulations = modulations[:-1]
            final_modulation = modulations[-1]
        else:
            block_modulations = [None] * self.depth
            final_modulation = None
        for block, modulation in zip(self.transformer_blocks, block_modulations):
            x = block(x, time, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin, modulation=modulation)
        return self.proj_out(self.norm_out(x, time, modulation=final_modulation))


class VocosConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, layer_scale_init_value: float, adanorm_num_embeddings: int | None = None):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = VocosAdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6) if adanorm_num_embeddings else nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        if self.adanorm:
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)
        return residual + x


class VocosAdaLayerNorm(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = embedding_dim
        self.scale = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.shift = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.nn.init.ones_(self.scale.weight)
        torch.nn.init.zeros_(self.shift.weight)

    def forward(self, x: torch.Tensor, cond_embedding_id: torch.Tensor) -> torch.Tensor:
        scale = self.scale(cond_embedding_id)
        shift = self.shift(cond_embedding_id)
        x = nn.functional.layer_norm(x, (self.dim,), eps=self.eps)
        return x * scale + shift


class VocosBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        layer_scale_init_value: float | None = None,
        adanorm_num_embeddings: int | None = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=7, padding=3)
        self.adanorm = adanorm_num_embeddings is not None
        self.norm = VocosAdaLayerNorm(adanorm_num_embeddings, dim, eps=1e-6) if adanorm_num_embeddings else nn.LayerNorm(dim, eps=1e-6)
        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                VocosConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=adanorm_num_embeddings,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get('bandwidth_id', None)
        x = self.embed(x)
        if self.adanorm:
            x = self.norm(x.transpose(1, 2), cond_embedding_id=bandwidth_id)
        else:
            x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x, cond_embedding_id=bandwidth_id)
        return self.final_layer_norm(x.transpose(1, 2))


class UnusedISTFTPlaceholder(nn.Module):
    """State-dict-compatible placeholder; actual ISTFT is handled by STFT_Process."""

    def __init__(self, n_fft: int):
        super().__init__()
        self.register_buffer("window", torch.empty(n_fft, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        raise RuntimeError("Vocos ISTFT is unused in this export path; use the custom STFT_Process ISTFT.")


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2
        self.num_bins = out_dim // 2
        self.out = nn.Linear(dim, out_dim)
        self.istft = UnusedISTFTPlaceholder(n_fft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(x).transpose(1, 2)
        mag, p = torch.split(x, self.num_bins, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)
        return mag, p


EXPORT_CLASS_OVERRIDES = {
    "vocos.models.VocosBackbone": VocosBackbone,
    "vocos.heads.ISTFTHead": ISTFTHead,
}


def instantiate_export_class(args: Any | tuple[Any, ...], init: dict[str, Any]) -> Any:
    kwargs = init.get("init_args", {})
    if not isinstance(args, tuple):
        args = (args,)
    class_path = init["class_path"]
    args_class = EXPORT_CLASS_OVERRIDES.get(class_path)
    if args_class is None:
        class_module, class_name = class_path.rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        args_class = getattr(module, class_name)
    return args_class(*args, **kwargs)


class Vocos(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_hparams(cls, config_path: str) -> Vocos:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        feature_extractor = instantiate_export_class(args=(), init=config["feature_extractor"])
        backbone = instantiate_export_class(args=(), init=config["backbone"])
        head = instantiate_export_class(args=(), init=config["head"])
        return cls(feature_extractor=feature_extractor, backbone=backbone, head=head)

    @classmethod
    def from_pretrained(cls, repo_id: str, revision: str | None = None) -> Vocos:
        config_path = repo_id + "/config.yaml"
        model_path = repo_id + "/pytorch_model.bin"
        model = cls.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in model.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @torch.inference_mode()
    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        features = self.feature_extractor(audio_input, **kwargs)
        return self.decode(features, **kwargs)

    @torch.inference_mode()
    def decode(self, features_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = self.backbone(features_input, **kwargs)
        return self.head(x)


class F5Preprocess(torch.nn.Module):
    def __init__(self, f5_model, custom_stft, nfft, n_mels, sample_rate, num_head, head_dim, target_rms, use_fp16):
        super(F5Preprocess, self).__init__()
        self.f5_text_embed = f5_model.transformer.text_embed
        self.custom_stft = custom_stft
        self.num_channels = n_mels
        self.base_rescale_factor = 1.0      # Official setting
        self.interpolation_factor = 1.0     # Official setting
        self.target_rms = target_rms
        base = 10000.0 * self.base_rescale_factor ** (head_dim / (head_dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(MAX_SIGNAL_LENGTH, dtype=torch.float32), inv_freq) / self.interpolation_factor
        freqs_cos = torch.stack((freqs.cos(), freqs.cos()), dim=-1).flatten(-2)
        freqs_sin = torch.stack((-freqs.sin(), freqs.sin()), dim=-1).flatten(-2)
        # head axis kept at 1 so a single table broadcasts across the packed q/k head stack (2 * num_head).
        rope_dtype = torch.float16 if use_fp16 else torch.float32
        self.register_buffer("rope_cos", freqs_cos.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1).half().to(rope_dtype), persistent=False)
        self.register_buffer("rope_sin", freqs_sin.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1).half().to(rope_dtype), persistent=False)
        self.register_buffer("mel_padding", torch.zeros((1, MAX_SIGNAL_LENGTH, n_mels), dtype=torch.float32), persistent=False)
        self.register_buffer("text_padding", torch.zeros((1, MAX_SIGNAL_LENGTH), dtype=torch.int32), persistent=False)
        fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0)
        self.register_buffer("fbank", fbank, persistent=False)
        self.inv_int16 = float(1.0 / 32768.0)
        self.input_resample_scale = float(sample_rate / IN_SAMPLE_RATE)
        self.use_fp16 = use_fp16

    def forward(self,
                audio: torch.ShortTensor,
                text_ids: torch.IntTensor,
                max_duration: torch.LongTensor,
                ):
        audio = audio.float()
        if "int" in IN_AUDIO_DTYPE.lower():
            audio = audio * self.inv_int16
        if self.input_resample_scale != 1.0:
            audio = F.interpolate(
                audio,
                scale_factor=self.input_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        audio_rms = torch.sqrt(torch.mean(audio * audio))
        rms_scale = torch.clamp(audio_rms / self.target_rms, max=1.0).reshape(1)
        audio_gain = torch.clamp(self.target_rms / audio_rms.clamp(min=1e-6), min=1.0)
        audio = audio * audio_gain
        mel_signal_real, mel_signal_imag = self.custom_stft(audio)
        mel_signal = torch.matmul(self.fbank, torch.sqrt(mel_signal_real * mel_signal_real + mel_signal_imag * mel_signal_imag)).transpose(1, 2).clamp(min=1e-5).log()
        ref_signal_len = (torch.ones_like(mel_signal[:, :, 0]).sum(dtype=torch.float32) - 1.0).to(torch.int64)
        ref_mel_tail = mel_signal[:, -1:]
        zeros = self.mel_padding[:, :max_duration]
        mel_signal = torch.cat((mel_signal, self.mel_padding), dim=1)[:, :max_duration]
        text_ids = torch.cat((text_ids + 1, self.text_padding), dim=-1)[:, :max_duration]
        noise = torch.randn_like(zeros)
        rope_cos = self.rope_cos[:, :, :max_duration]
        rope_sin = self.rope_sin[:, :, :max_duration]
        text, text_drop = self.f5_text_embed(text_ids, max_duration)
        cat_mel_text = torch.cat((mel_signal, text), dim=-1)
        cat_mel_text_drop = torch.cat((zeros, text_drop), dim=-1)
        if self.use_fp16:
            return noise.half(), rope_cos, rope_sin, cat_mel_text.half(), cat_mel_text_drop.half(), ref_signal_len, rms_scale, ref_mel_tail.half()
        return noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, ref_signal_len, rms_scale, ref_mel_tail


def get_epss_timesteps(n, dtype=torch.float32):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, dtype=dtype)
    return dt * torch.tensor(t, dtype=dtype)


class F5Transformer(torch.nn.Module):
    def __init__(self, f5_model, cfg, steps, sway_coef, dtype):
        super(F5Transformer, self).__init__()
        self.f5_transformer = f5_model.transformer
        self.time_mlp = f5_model.transformer.time_embed.time_mlp
        self.freq_embed_dim = 256
        self.time_mlp_dim = 1024
        self.cfg_strength = cfg
        self.cond_scale = 1.0 + cfg
        self.uncond_scale = -cfg
        self.sway_sampling_coef = sway_coef
        t = get_epss_timesteps(steps, dtype=torch.float32)
        time_step = t + self.sway_sampling_coef * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
        delta_t = torch.diff(time_step).to(dtype).view(-1, 1, 1)
        time_expand = torch.zeros((1, len(time_step), self.time_mlp_dim), dtype=torch.float32)
        half_dim = self.freq_embed_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
        for i in range(len(time_step)):
            emb = time_step[i] * emb_factor
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            time_expand[:, [i]] = self.time_mlp(emb)
        self.register_buffer("delta_t", delta_t, persistent=False)
        self.register_buffer("time_expand", time_expand.to(dtype), persistent=False)
        self.num_steps = steps

    def denoise_step(self, noise, cat_mel_text, cat_mel_text_drop, rope_cos, rope_sin, time_embed, delta_t):
        pred = self.f5_transformer(x=noise, cond=cat_mel_text, cond_drop=cat_mel_text_drop, time=time_embed, rope_cos=rope_cos, rope_sin=rope_sin)
        pred, pred_drop = torch.split(pred, [1, 1], dim=0)
        return noise + (pred * self.cond_scale + pred_drop * self.uncond_scale) * delta_t

    def forward(self,
                noise: torch.FloatTensor,
                rope_cos: torch.FloatTensor,
                rope_sin: torch.FloatTensor,
                cat_mel_text: torch.FloatTensor,
                cat_mel_text_drop: torch.FloatTensor,
                time_step: torch.IntTensor
                ):
        # The NFE loop lives in the inference driver; each call runs one denoise step, gathering this
        # step's time embedding / delta_t from the precomputed tables via the time_step index.
        return self.denoise_step(noise, cat_mel_text, cat_mel_text_drop, rope_cos, rope_sin, self.time_expand[:, time_step], self.delta_t[time_step])


class F5Decode(torch.nn.Module):
    def __init__(self, vocos, custom_istft, target_rms, use_fp16):
        super(F5Decode, self).__init__()
        self.vocos = vocos
        self.custom_istft = custom_istft
        self.target_rms = float(target_rms)
        self.output_resample_scale = float(OUT_SAMPLE_RATE / MODEL_SAMPLE_RATE)
        self.use_fp16 = use_fp16

    def forward(self,
                denoised: torch.FloatTensor,
                ref_signal_len: torch.LongTensor,
                rms_scale: torch.FloatTensor,
                ref_mel_tail: torch.FloatTensor,
                ):
        denoised = denoised[:, ref_signal_len:]
        if self.use_fp16:
            denoised = denoised.float()
            ref_mel_tail = ref_mel_tail.float()
        denoised = torch.cat((ref_mel_tail, denoised[:, 1:]), dim=1)
        denoised = self.vocos.decode(denoised.transpose(1, 2))
        generated_signal = self.custom_istft(*denoised)
        generated_signal = generated_signal * rms_scale
        generated_signal = generated_signal.clamp(min=-1.0, max=1.0)
        if self.output_resample_scale != 1.0:
            generated_signal = F.interpolate(
                generated_signal,
                scale_factor=self.output_resample_scale,
                mode="linear",
                align_corners=False,
                recompute_scale_factor=False,
            )
        if "int" in OUT_AUDIO_DTYPE.lower():
            return (generated_signal * 32767.0).clamp(-32768.0, 32767.0).to(torch.int16)
        if "32" in OUT_AUDIO_DTYPE:
            return generated_signal.float()
        return generated_signal.half()


def load_model(ckpt_path):
    model_cfg = OmegaConf.load(F5_config_path)
    model_cls = globals()[model_cfg.model.backbone]
    model = CFM(
        transformer=model_cls(**model_cfg.model.arch, text_num_embeds=vocab_size, mel_dim=N_MELS),
        mel_spec_kwargs=dict(  # Not important here. Use the custom STFT/ISTFT instead.
            target_sample_rate=MODEL_SAMPLE_RATE,
            n_mel_channels=N_MELS,
            hop_length=HOP_LENGTH,
        ),
        odeint_kwargs=dict(
            method='euler',     # Only the Euler method is implemented for ONNX here.
        ),
        vocab_char_map=vocab_char_map,
    ).to('cpu')
    return load_checkpoint(model, ckpt_path, 'cpu', use_ema=True), model_cfg.model.arch.heads, model_cfg.model.arch.dim


def load_checkpoint(model, ckpt_path, device: str, dtype=None, use_ema=True):
    if dtype is None:
        dtype = torch.float32
    model = model.to(dtype)

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path, device=device)
    else:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"])

    del checkpoint
    torch.cuda.empty_cache()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers — export the pipeline geometry once so inference never has to
# hand-duplicate the fixed-at-export constants (mirrors the ASR repo).
# ─────────────────────────────────────────────────────────────────────────────
def build_model_metadata(*sections):
    """Flatten metadata sections into a ``{str: str}`` dict for ``metadata_props``."""
    metadata = {}
    for section in sections:
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                metadata[str(key)] = "1" if value else "0"
            elif isinstance(value, (list, tuple)):
                metadata[str(key)] = ",".join(str(item) for item in value)
            else:
                metadata[str(key)] = str(value)
    return metadata


def write_onnx_metadata(onnx_path, metadata):
    """Add/overwrite ``metadata_props`` in place, leaving any external-weight sidecar untouched."""
    import onnx

    onnx_model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in onnx_model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            onnx_model.metadata_props.add(key=key, value=value)
    onnx.save(onnx_model, onnx_path)


class METADATA_CARRIER(torch.nn.Module):
    """Tiny identity graph that carries the static package contract."""

    def forward(self, marker):
        return marker


# Dummy input shapes used only while tracing the export graphs.
DUMMY_AUDIO_LENGTH = 160000
DUMMY_TEXT_IDS_LENGTH = 60
DUMMY_MAX_GENERATED_LENGTH = 600
DUMMY_TEXT_EMBED_LENGTH = 512 + N_MELS
DUMMY_MODEL_AUDIO_LENGTH = int(DUMMY_AUDIO_LENGTH * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
DUMMY_REFERENCE_SIGNAL_LENGTH = DUMMY_MODEL_AUDIO_LENGTH // HOP_LENGTH + 1
DUMMY_MAX_DURATION = min(DUMMY_REFERENCE_SIGNAL_LENGTH + DUMMY_MAX_GENERATED_LENGTH, MAX_SIGNAL_LENGTH)


if args.check_config:
    checked_model, checked_heads, checked_hidden_size = load_model(F5_safetensors_path)
    checked_arch = OmegaConf.load(F5_config_path).model.arch
    expected_pe_attn_head = 1 if F5_MODEL_SERIES == "v0" else None
    expected_text_mask_padding = F5_MODEL_SERIES == "v1"
    if checked_arch.get("pe_attn_head") != expected_pe_attn_head:
        raise RuntimeError(f"Unexpected pe_attn_head for {F5_MODEL_SERIES}: {checked_arch.get('pe_attn_head')}")
    if checked_arch.get("text_mask_padding", True) is not expected_text_mask_padding:
        raise RuntimeError(
            f"Unexpected text_mask_padding for {F5_MODEL_SERIES}: "
            f"{checked_arch.get('text_mask_padding', True)}"
        )
    print(
        f"[F5-TTS] Configuration check passed: series={F5_MODEL_SERIES}, "
        f"hidden_size={checked_hidden_size}, heads={checked_heads}, "
        f"depth={len(checked_model.transformer.transformer_blocks)}, vocab_size={vocab_size}."
    )
    del checked_model
    gc.collect()
    raise SystemExit(0)


for output_path in (preprocess_path, transformer_path, decode_path, metadata_path, transformer_raw_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)


print("\n\nStart to Export the F5-TTS Preprocess Part.")
with torch.inference_mode():
    # Dummy for Export the F5_Preprocess part
    audio = torch.ones((1, 1, DUMMY_AUDIO_LENGTH), dtype=_AUDIO_DTYPES[IN_AUDIO_DTYPE.upper()])
    text_ids = torch.ones((1, DUMMY_TEXT_IDS_LENGTH), dtype=torch.int32)
    max_duration = torch.tensor([DUMMY_MAX_DURATION], dtype=torch.long)
    f5_model, NUM_HEAD, HIDDEN_SIZE = load_model(F5_safetensors_path)
    HEAD_DIM = HIDDEN_SIZE // NUM_HEAD
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE, pad_mode='reflect').eval()
    f5_preprocess = F5Preprocess(f5_model, custom_stft, nfft=NFFT, n_mels=N_MELS, sample_rate=MODEL_SAMPLE_RATE, num_head=NUM_HEAD, head_dim=HEAD_DIM, target_rms=TARGET_RMS, use_fp16=use_fp16_transformer)
    torch.onnx.export(
        f5_preprocess,
        (audio, text_ids, max_duration),
        onnx_model_Preprocess,
        input_names=['audio', 'text_ids', 'max_duration'],
        output_names=['noise', 'rope_cos', 'rope_sin', 'cat_mel_text', 'cat_mel_text_drop', 'ref_signal_len', 'rms_scale', 'ref_mel_tail'],
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'text_ids': {1: 'text_ids_len'},
            'noise': {1: 'max_duration'},
            'rope_cos': {2: 'max_duration'},
            'rope_sin': {2: 'max_duration'},
            'cat_mel_text': {1: 'max_duration'},
            'cat_mel_text_drop': {1: 'max_duration'}
        } if DYNAMIC_AXES else None,
        dynamo=False,
        opset_version=OPSET)
    del custom_stft
    del f5_preprocess
    del audio
    del text_ids
    del max_duration
    gc.collect()
print("\nExport Done.")


print("\n\nStart to Export the F5-TTS Transformer Part.")
with torch.inference_mode():
    scale_factor = math.pow(HEAD_DIM, -0.25)
    if use_fp16_transformer:
        print("\nNote: Exporting F5_Transformer.onnx in float16 format will take a long time.")
        scale_factor *= 0.1  # To avoid overflow in float16 format.
        dtype = torch.float16
    else:
        dtype = torch.float32
    # Fuse q/k/v into one GEMM per block, folding the head_dim**-0.25 attention scale into q & k.
    for i in range(len(f5_model.transformer.transformer_blocks)):
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.fuse_qkv(scale_factor)
    f5_model.transformer.fuse_time_projections()

    noise = torch.ones((1, DUMMY_MAX_DURATION, N_MELS), dtype=dtype)
    rope_cos = torch.ones((2, 1, DUMMY_MAX_DURATION, HEAD_DIM), dtype=dtype)
    rope_sin = torch.ones((2, 1, DUMMY_MAX_DURATION, HEAD_DIM), dtype=dtype)
    cat_mel_text = torch.ones((1, DUMMY_MAX_DURATION, DUMMY_TEXT_EMBED_LENGTH), dtype=dtype)
    cat_mel_text_drop = torch.ones((1, DUMMY_MAX_DURATION, DUMMY_TEXT_EMBED_LENGTH), dtype=dtype)
    time_step = torch.tensor([0], dtype=torch.int32)
    print('\nNote: Exporting the Transformer as a single denoise step; the inference driver runs the NFE loop.')
    f5_transformer = F5Transformer(f5_model, cfg=CFG_STRENGTH, steps=NFE_STEP, sway_coef=SWAY_COEFFICIENT, dtype=dtype)
    if use_fp16_transformer:
        f5_transformer = f5_transformer.half()
    transformer_inputs = (noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, time_step)
    transformer_input_names = ['noise', 'rope_cos', 'rope_sin', 'cat_mel_text', 'cat_mel_text_drop', 'time_step']
    transformer_output_names = ['denoised']
    torch.onnx.export(
        f5_transformer,
        transformer_inputs,
        onnx_model_Transformer_Raw,
        input_names=transformer_input_names,
        output_names=transformer_output_names,
        dynamic_axes={
            'noise': {1: 'max_duration'},
            'rope_cos': {2: 'max_duration'},
            'rope_sin': {2: 'max_duration'},
            'cat_mel_text': {1: 'max_duration'},
            'cat_mel_text_drop': {1: 'max_duration'},
            'denoised': {1: 'max_duration'}
        } if DYNAMIC_AXES else None,
        dynamo=False,
        opset_version=OPSET)
    del f5_transformer
    del noise
    del rope_cos
    del rope_sin
    del cat_mel_text
    del cat_mel_text_drop
    gc.collect()
    print("\nExport Done.")


print("\n\nStart to Export the F5-TTS Decode Part.")
with torch.inference_mode():
    # Dummy for Export the F5_Decode part
    denoised = torch.ones((1, DUMMY_MAX_DURATION, N_MELS), dtype=dtype)
    ref_signal_len = torch.tensor(DUMMY_REFERENCE_SIGNAL_LENGTH, dtype=torch.long)
    rms_scale = torch.ones((1,), dtype=torch.float32)
    ref_mel_tail = torch.ones((1, 1, N_MELS), dtype=dtype)
    custom_istft = STFT_Process(model_type='istft_A', n_fft=NFFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    # Vocos model preprocess
    vocos = Vocos.from_pretrained(vocos_model_path)
    f5_decode = F5Decode(vocos, custom_istft, target_rms=TARGET_RMS, use_fp16=use_fp16_transformer)
    torch.onnx.export(
        f5_decode,
        (denoised, ref_signal_len, rms_scale, ref_mel_tail),
        onnx_model_Decode,
        input_names=['denoised', 'ref_signal_len', 'rms_scale', 'ref_mel_tail'],
        output_names=['output_audio'],
        dynamic_axes={
            'denoised': {1: 'max_duration'},
            'output_audio': {2: 'generated_len'},
        } if DYNAMIC_AXES else None,
        dynamo=False,
        opset_version=OPSET)
    del f5_decode
    del denoised
    del ref_signal_len
    del rms_scale
    del ref_mel_tail
    del vocos
    del custom_istft
    gc.collect()
    print("\nExport Done.")

# ── Metadata carrier + stamp the metadata onto every exported graph ──
onnx_metadata = build_model_metadata(
    {
        "model_series": F5_MODEL_SERIES,
        "sample_rate": MODEL_SAMPLE_RATE,
        "in_sample_rate": IN_SAMPLE_RATE,
        "out_sample_rate": OUT_SAMPLE_RATE,
        "nfe_step": NFE_STEP,
        "max_signal_length": MAX_SIGNAL_LENGTH,
        "hop_length": HOP_LENGTH,
        "model_file_name_preprocess": Path(os.path.relpath(preprocess_path, metadata_path.parent)).as_posix(),
        "model_file_name_transformer": Path(os.path.relpath(transformer_path, metadata_path.parent)).as_posix(),
        "model_file_name_decode": Path(os.path.relpath(decode_path, metadata_path.parent)).as_posix(),
    },
)
metadata_marker = torch.zeros((1,), dtype=torch.int64)
torch.onnx.export(
    METADATA_CARRIER(),
    (metadata_marker,),
    onnx_model_Metadata,
    input_names=['metadata_marker'],
    output_names=['metadata_marker_out'],
    dynamic_axes=None,
    opset_version=OPSET,
    dynamo=False
)
del metadata_marker
_metadata_targets = [
    onnx_model_Preprocess,
    onnx_model_Transformer_Raw,
    onnx_model_Decode,
    onnx_model_Metadata,
]
for _target in _metadata_targets:
    write_onnx_metadata(_target, onnx_metadata)
if OPSET >= 18:
    _mish_report = rewrite_mish_subgraphs(onnx_model_Transformer_Raw, onnx_model_Transformer)
else:
    transformer_path.unlink(missing_ok=True)
    os.replace(transformer_raw_path, transformer_path)
    _mish_report = None
print(
    f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {len(_metadata_targets)} raw/source graph(s); "
    "the final Transformer preserves them."
)
if _mish_report is None:
    print("[Rewrite] Opset 17 selected; retained the exporter's compatible Mish decomposition.")
else:
    transformer_raw_path.unlink(missing_ok=True)
    print(
        f"[Rewrite] Replaced {_mish_report['matched_subgraphs']} Mish decomposition(s); "
        f"net node reduction: {_mish_report['net_node_reduction']}."
    )
print(f"[Cleanup] Removed temporary Transformer graph: {transformer_raw_path}")

print("\nExport done!")
if not args.skip_inference:
    print("\nStart running inference via Inference_F5_TTS_ONNX.py ...")
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "Inference_F5_TTS_ONNX.py"),
            "--metadata_path",
            onnx_model_Metadata,
            "--preprocessmodel_path",
            onnx_model_Preprocess,
            "--transformermodel_path",
            onnx_model_Transformer,
            "--decodermodel_path",
            onnx_model_Decode,
            "--vocab_path",
            vocab_path,
            "--prepro_input_type",
            args.prepro_input_type,
            "--testlang",
            args.testlang,
            "--seed",
            str(args.seed),
        ],
        check=True,
    )