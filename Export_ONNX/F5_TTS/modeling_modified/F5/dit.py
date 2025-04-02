"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNorm_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)


# Text embedding


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, mask_padding=True, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int['b nt'], seq_len):
        text_mask = text == 0
        text_drop = self.text_embed(torch.zeros_like(text))
        text = self.text_embed(text)  # b n -> b n d
        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1))
        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            pos_idx = get_pos_embed_indices(0, seq_len, max_pos=self.precompute_max_pos)
            pos_idx = self.freqs_cis[pos_idx]

            text = text + pos_idx
            # convnextv2 blocks
            text = text.masked_fill(text_mask, 0.0)
            for block in self.text_blocks:
                text = block(text)
                text = text.masked_fill(text_mask, 0.0)

            text_drop = text_drop + pos_idx
            # convnextv2 blocks
            text_drop = text_drop.masked_fill(text_mask, 0.0)
            for block in self.text_blocks:
                text_drop = block(text_drop)
                text_drop = text_drop.masked_fill(text_mask, 0.0)
        return text, text_drop


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float['b n d'], cond: float['b n d'], drop_audio_cond=False):
        x = self.proj(torch.cat((x, cond), dim=-1))
        return self.conv_pos_embed(x) + x


# Transformer backbone using DiT blocks


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
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(
            text_num_embeds, text_dim, mask_padding=text_mask_padding, conv_layers=conv_layers
        )
        self.text_cond, self.text_uncond = None, None  # text cache
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
                )
                for _ in range(depth)
            ]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNorm_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out AdaLN layers in DiT blocks:
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm.linear.weight, 0)
            nn.init.constant_(block.attn_norm.linear.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None

    def forward(
            self,
            x: float['b n d'],  # nosied input audio
            cond: float['b n d'],  # masked cond audio
            cond_drop,
            time: float['b'] | float[''],  # time step
            rope_cos_q,
            rope_sin_q,
            rope_cos_k,
            rope_sin_k,
            mask: bool['b n'] | None = None
    ):
        x = torch.cat((self.input_embed(x, cond), self.input_embed(x, cond_drop)), dim=0)
        for block in self.transformer_blocks:
            x = block(x, time, mask=mask, rope_cos_q=rope_cos_q, rope_sin_q=rope_sin_q, rope_cos_k=rope_cos_k, rope_sin_k=rope_sin_k)
        return self.proj_out(self.norm_out(x, time))
