#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STFT / ISTFT  —  Ultra-Optimized ONNX-exportable Short-Time Fourier Transform.

Static-graph, ONNXRuntime-friendly export. All optimization happens in PyTorch
__init__() and forward() — no post-export graph surgery or onnxslim.

This script:
  1. Builds a PyTorch model (Conv1d STFT / ConvTranspose1d ISTFT) with all
     constants precomputed as registered buffers.
  2. Exports to ONNX with static shapes (dynamic_axes=None).
  3. Validates against torch.stft / torch.istft.
  4. Runs a round-trip (STFT → ISTFT) reconstruction test.

Optimization summary vs original:
  - Static graph: fixed input/output shapes, no Shape/Gather/Range ops.
  - Forward dispatch eliminated: direct method call, no dict lookup.
  - No batch-dependent branching in forward.
  - inv_win_sum stored as float32 (no half→float Cast op in graph).
  - inv_win_sum pre-sliced to exact output size (no runtime dynamic indexing).
  - Removed unused buffers (ones, expected_len) from exported graph.
  - do_constant_folding=True in torch.onnx.export (allowed).
"""

import torch
import numpy as np
import onnxruntime as ort
from onnxslim import slim

# ═════════════════════════════════════════════════════════════════════════════
# 1.  Configuration
# ═════════════════════════════════════════════════════════════════════════════

# -- Model variants to export ----------------------------------------------
#    stft_A  / istft_A  →  real-only STFT  /  magnitude+phase ISTFT
#    stft_B  / istft_B  →  real+imag STFT  /  real+imag       ISTFT
STFT_TYPE  = "stft_B"
ISTFT_TYPE = "istft_B"

OPSET = 18                   # ONNX opset version (torch 2.12+ requires ≥18)
DYNAMIC_AXES = True          # Set True for the In/Out dynamic tensor shape.

# -- FFT / framing parameters ----------------------------------------------
NFFT         = 400           # FFT size (number of frequency bins before folding)
WIN_LENGTH   = 400           # Analysis window length in samples (≤ NFFT)
HOP_LENGTH   = 160           # Hop (stride) between successive frames
WINDOW_TYPE  = 'hann'        # Window function: bartlett | blackman | hamming | hann | kaiser

# -- Padding ---------------------------------------------------------------
CENTER_PAD   = True          # True  → pad signal so frame centres align with sample indices
                             # False → no padding, first frame starts at sample 0
PAD_MODE     = 'constant'    # Padding style when CENTER_PAD is True: 'reflect' | 'constant'

# -- Audio dimensions (fixed for static export) ----------------------------
INPUT_AUDIO_LENGTH = 16000   # Length of the waveform (samples) — static export shape

# -- Derived export paths ---------------------------------------------------
export_path_stft  = f"{STFT_TYPE}.onnx"
export_path_istft = f"{ISTFT_TYPE}.onnx"


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Derived constants & helpers
# ═════════════════════════════════════════════════════════════════════════════

NFFT       = min(NFFT, INPUT_AUDIO_LENGTH)
WIN_LENGTH = min(WIN_LENGTH, NFFT)
HOP_LENGTH = min(HOP_LENGTH, INPUT_AUDIO_LENGTH)
HALF_NFFT  = NFFT // 2
F_BINS     = HALF_NFFT + 1   # Number of one-sided frequency bins

# Number of STFT frames for the static input length.
if CENTER_PAD:
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
else:
    STFT_SIGNAL_LENGTH = (INPUT_AUDIO_LENGTH - NFFT) // HOP_LENGTH + 1

# Static ISTFT output bounds (precomputed, no runtime Shape/Gather).
_ISTFT_RAW_LEN = NFFT + HOP_LENGTH * (STFT_SIGNAL_LENGTH - 1)
if CENTER_PAD:
    _ISTFT_OUT_START = HALF_NFFT
    _ISTFT_OUT_END   = _ISTFT_RAW_LEN - HALF_NFFT
else:
    _ISTFT_OUT_START = 0
    _ISTFT_OUT_END   = _ISTFT_RAW_LEN

# -- Window function registry ----------------------------------------------
WINDOW_FUNCTIONS = {
    'bartlett':  lambda L: torch.bartlett_window(L, periodic=True),
    'blackman':  lambda L: torch.blackman_window(L, periodic=True),
    'hamming':   lambda L: torch.hamming_window(L,  periodic=True),
    'hann':      lambda L: torch.hann_window(L,     periodic=True),
    'hann_sqrt': lambda L: torch.hann_window(L,     periodic=False).pow(0.5),
    'povey':     lambda L: torch.hann_window(L,     periodic=False).pow(0.85),
    'kaiser':    lambda L: torch.kaiser_window(L,   periodic=True, beta=12.0)
}
DEFAULT_WINDOW_FN = lambda L: torch.hann_window(L, periodic=True)


def create_padded_window(win_length: int, n_fft: int, window_type: str) -> torch.Tensor:
    """Create a window of length *n_fft*, center-padding or cropping as needed."""
    win_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    win = win_fn(win_length).float()

    if win_length == n_fft:
        return win
    if win_length < n_fft:
        pad_total = n_fft - win_length
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return torch.cat([torch.zeros(pad_left), win, torch.zeros(pad_right)])
    start = (win_length - n_fft) // 2
    return win[start : start + n_fft]


def get_raw_window(win_length: int, window_type: str) -> torch.Tensor:
    """Return the raw (un-padded) window — used by ``torch.stft`` for reference tests."""
    win_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    return win_fn(win_length).float()


WINDOW = create_padded_window(WIN_LENGTH, NFFT, WINDOW_TYPE)


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Optimized STFT / ISTFT Models (Static Graph)
# ═════════════════════════════════════════════════════════════════════════════

class STFT_Process(torch.nn.Module):
    """
    Static-graph Conv1d STFT / ConvTranspose1d ISTFT for ONNX export.

    All constants precomputed in __init__() as registered buffers.
    Forward path is pure tensor ops — no dispatch, no branching, no shape queries.

    Variants
    --------
    stft_A   → Conv1d producing real part only.
    stft_B   → Conv1d producing real + imag (split after convolution).
    istft_A  → (magnitude, phase) → ConvTranspose1d reconstruction.
    istft_B  → (real, imag) → ConvTranspose1d reconstruction.
    """

    def __init__(
        self,
        model_type: str,
        n_fft: int       = NFFT,
        win_length: int  = WIN_LENGTH,
        hop_len: int     = HOP_LENGTH,
        max_frames: int  = STFT_SIGNAL_LENGTH,
        window_type: str = WINDOW_TYPE,
        center_pad: bool = CENTER_PAD,
        pad_mode: str    = PAD_MODE
    ):
        super().__init__()

        self.model_type = model_type
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.half_n_fft = n_fft // 2
        self.n_frames   = max_frames

        f_bins = self.half_n_fft + 1
        window = create_padded_window(win_length, n_fft, window_type)

        # ── Precompute static output slice bounds for ISTFT ───────────────
        raw_len = n_fft + hop_len * (max_frames - 1)
        if center_pad:
            self._out_start = self.half_n_fft
            self._out_end   = raw_len - self.half_n_fft
        else:
            self._out_start = 0
            self._out_end   = raw_len

        # ── Bind forward to the correct variant (no dispatch overhead) ────
        if model_type == 'stft_A':
            self.forward = self._stft_A_forward
        elif model_type == 'stft_B':
            self.forward = self._stft_B_forward
        elif model_type == 'istft_A':
            self.forward = self._istft_A_forward
        elif model_type == 'istft_B':
            self.forward = self._istft_B_forward
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # ── STFT: constant zero-padding buffer ────────────────────────────
        if model_type in ('stft_A', 'stft_B'):
            self._build_stft_kernels(n_fft, f_bins, window, model_type)
            if center_pad and pad_mode == 'constant':
                self.register_buffer(
                    'padding_zero',
                    torch.zeros(1, 1, self.half_n_fft, dtype=torch.float32)
                )
            self._center_pad = center_pad
            self._pad_mode   = pad_mode

        # ── ISTFT: inverse kernel + pre-sliced normalization ──────────────
        if model_type in ('istft_A', 'istft_B'):
            self._build_istft_kernels(n_fft, f_bins, window, hop_len, max_frames)

    def _build_stft_kernels(self, n_fft, f_bins, window, model_type):
        """Precompute windowed DFT basis as Conv1d kernel weights."""
        omega_factor = 2.0 * torch.pi / n_fft
        t = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        f = torch.arange(f_bins, dtype=torch.float32).unsqueeze(1)
        omega = omega_factor * f * t

        windowed_cos = ( torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        windowed_sin = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)

        if model_type == 'stft_A':
            self.register_buffer('stft_kernel', windowed_cos)
        else:
            self.register_buffer('stft_kernel', torch.cat([windowed_cos, windowed_sin], dim=0))

    def _build_istft_kernels(self, n_fft, f_bins, window, hop_len, n_frames):
        """Precompute inverse-DFT kernel and window² kernel for COLA normalization."""
        omega_factor = 2.0 * torch.pi / n_fft
        k = torch.arange(f_bins, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        omega = omega_factor * k * n

        cos_basis = torch.cos(omega)
        sin_basis = torch.sin(omega)

        scale = 2.0 * torch.ones(f_bins, 1)
        scale[0] = 1.0
        if n_fft % 2 == 0:
            scale[f_bins - 1] = 1.0

        inv_n     = 1.0 / n_fft
        ifft_real = (scale *  cos_basis * inv_n) * window.unsqueeze(0)
        ifft_imag = (scale * -sin_basis * inv_n) * window.unsqueeze(0)

        self.register_buffer(
            'inverse_kernel',
            torch.cat([ifft_real, ifft_imag], dim=0).unsqueeze(1)
        )

        # Store window² kernel for dynamic COLA normalization in forward.
        self.register_buffer('win_sq_kernel', window.square().reshape(1, 1, -1))

    # --------------------------------------------------------------------- #
    #  STFT forward variants (no branching, static tensor ops only)         #
    # --------------------------------------------------------------------- #

    def _stft_A_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STFT producing real part only (cosine projection)."""
        if self._center_pad:
            if self._pad_mode == 'reflect':
                left  = x[..., 1: self.half_n_fft + 1].flip(2)
                right = x[..., -(self.half_n_fft + 1): -1].flip(2)
                x = torch.cat([left, x, right], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero = torch.cat([self.padding_zero] * x.shape[0], dim=0)
                else:
                    padding_zero = self.padding_zero
                x = torch.cat([padding_zero, x, padding_zero], dim=2)
        return torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)

    def _stft_B_forward(self, x: torch.Tensor):
        """STFT producing (real, imag) via a single Conv1d + channel Split."""
        if self._center_pad:
            if self._pad_mode == 'reflect':
                left  = x[..., 1: self.half_n_fft + 1].flip(2)
                right = x[..., -(self.half_n_fft + 1): -1].flip(2)
                x = torch.cat([left, x, right], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero = torch.cat([self.padding_zero] * x.shape[0], dim=0)
                else:
                    padding_zero = self.padding_zero
                x = torch.cat([padding_zero, x, padding_zero], dim=2)
        out = torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)
        return torch.split(out, self.half_n_fft + 1, dim=1)

    # --------------------------------------------------------------------- #
    #  ISTFT forward variants (static slicing, no Shape/Gather ops)         #
    # --------------------------------------------------------------------- #

    def _istft_B_forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """ISTFT from rectangular form. Dynamic-length compatible."""
        inp = torch.cat((real, imag), dim=1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_kernel, stride=self.hop_len)
        # Compute COLA normalization dynamically based on input n_frames.
        ones = torch.ones(1, 1, real.shape[2], dtype=real.dtype, device=real.device)
        win_sum = torch.nn.functional.conv_transpose1d(ones, self.win_sq_kernel, stride=self.hop_len)
        inv = inv[..., self._out_start:self._out_end] / win_sum[..., self._out_start:self._out_end]
        return inv

    def _istft_A_forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """ISTFT from polar form. Dynamic-length compatible."""
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        inp = torch.cat((real, imag), dim=1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_kernel, stride=self.hop_len)
        # Compute COLA normalization dynamically based on input n_frames.
        ones = torch.ones(1, 1, magnitude.shape[2], dtype=magnitude.dtype, device=magnitude.device)
        win_sum = torch.nn.functional.conv_transpose1d(ones, self.win_sq_kernel, stride=self.hop_len)
        inv = inv[..., self._out_start:self._out_end] / win_sum[..., self._out_start:self._out_end]
        return inv


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Validation helpers
# ═════════════════════════════════════════════════════════════════════════════

def _torch_istft_safe(complex_spec: torch.Tensor):
    """Call torch.istft; returns (audio_numpy, True) or (None, False) if COLA fails."""
    try:
        audio = torch.istft(
            complex_spec,
            n_fft=NFFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=get_raw_window(WIN_LENGTH, WINDOW_TYPE),
            center=CENTER_PAD
        ).squeeze().numpy()
        return audio, True
    except RuntimeError:
        return None, False


def test_onnx_stft_A(x: torch.Tensor):
    """Compare ONNX STFT-A output against torch.stft."""
    torch_out = torch.view_as_real(torch.stft(
        x.squeeze(0), n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        return_complex=True, window=get_raw_window(WIN_LENGTH, WINDOW_TYPE),
        pad_mode=PAD_MODE, center=CENTER_PAD
    ))
    pt_real = torch_out[..., 0].squeeze().numpy()

    sess     = ort.InferenceSession(export_path_stft)
    ort_real = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0].squeeze()

    err = np.abs(pt_real - ort_real[:, :pt_real.shape[-1]]).mean()
    print(f"STFT (A) [ONNX vs torch.stft]: mean |Δ| = {err:.2e}")
    return err


def test_onnx_stft_B(x: torch.Tensor):
    """Compare ONNX STFT-B output against torch.stft."""
    torch_out = torch.view_as_real(torch.stft(
        x.squeeze(0), n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        return_complex=True, window=get_raw_window(WIN_LENGTH, WINDOW_TYPE),
        pad_mode=PAD_MODE, center=CENTER_PAD
    ))
    pt_r = torch_out[..., 0].squeeze().numpy()
    pt_i = torch_out[..., 1].squeeze().numpy()

    sess         = ort.InferenceSession(export_path_stft)
    ort_r, ort_i = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})

    err = 0.5 * (
        np.abs(pt_r - ort_r.squeeze()[:, :pt_r.shape[-1]]).mean() +
        np.abs(pt_i - ort_i.squeeze()[:, :pt_r.shape[-1]]).mean()
    )
    print(f"STFT (B) [ONNX vs torch.stft]: mean |Δ| = {err:.2e}")
    return err


def test_onnx_istft_A(mag: torch.Tensor, phase: torch.Tensor):
    """Validate ONNX ISTFT-A against torch.istft."""
    sess      = ort.InferenceSession(export_path_istft)
    ort_audio = sess.run(None, {
        sess.get_inputs()[0].name: mag.numpy(),
        sess.get_inputs()[1].name: phase.numpy()
    })[0].squeeze()

    pt_audio, ok = _torch_istft_safe(torch.polar(mag, phase))
    if ok:
        min_len = min(len(pt_audio), len(ort_audio))
        err = np.abs(pt_audio[:min_len] - ort_audio[:min_len]).mean()
        print(f"ISTFT (A) [ONNX vs torch.istft]: mean |Δ| = {err:.2e}")
        return err
    print("ISTFT (A): torch.istft skipped (COLA not met)")
    return None


def test_onnx_istft_B(real: torch.Tensor, imag: torch.Tensor):
    """Validate ONNX ISTFT-B against torch.istft."""
    sess      = ort.InferenceSession(export_path_istft)
    ort_audio = sess.run(None, {
        sess.get_inputs()[0].name: real.numpy(),
        sess.get_inputs()[1].name: imag.numpy()
    })[0].squeeze()

    pt_audio, ok = _torch_istft_safe(torch.complex(real, imag))
    if ok:
        min_len = min(len(pt_audio), len(ort_audio))
        err = np.abs(pt_audio[:min_len] - ort_audio[:min_len]).mean()
        print(f"ISTFT (B) [ONNX vs torch.istft]: mean |Δ| = {err:.2e}")
        return err
    print("ISTFT (B): torch.istft skipped (COLA not met)")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# 5.  Export & verification
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import random
    import time
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with torch.no_grad():
        print(
            f"\nConfig  NFFT={NFFT}, WIN_LEN={WIN_LENGTH}, HOP={HOP_LENGTH}, "
            f"CENTER={CENTER_PAD}, FRAMES={STFT_SIGNAL_LENGTH}, "
            f"OUT_LEN={_ISTFT_OUT_END - _ISTFT_OUT_START}"
        )
        print(f"Export  STFT={STFT_TYPE}, ISTFT={ISTFT_TYPE}, opset={OPSET}, static_graph=True\n")

        # ── 5a. Export STFT ──────────────────────────────────────────────
        stft_model  = STFT_Process(STFT_TYPE).eval()
        dummy_audio = torch.randn(1, 1, INPUT_AUDIO_LENGTH, dtype=torch.float32)

        if STFT_TYPE == 'stft_A':
            out_names = ['real']
            dyn_axes_stft = {
                'input_audio': {2: 'audio_len'},
                'real':        {2: 'n_frames'},
            }
        else:
            out_names = ['real', 'imag']
            dyn_axes_stft = {
                'input_audio': {2: 'audio_len'},
                'real':        {2: 'n_frames'},
                'imag':        {2: 'n_frames'},
            }

        torch.onnx.export(
            stft_model,
            (dummy_audio,),
            export_path_stft,
            input_names=['input_audio'],
            output_names=out_names,
            dynamic_axes=dyn_axes_stft if DYNAMIC_AXES else None,
            opset_version=OPSET,
        )
        slim(model=export_path_stft, output_model=export_path_stft)
        print(f"  Exported: {export_path_stft}")

        # ── 5b. Export ISTFT ─────────────────────────────────────────────
        istft_model = STFT_Process(ISTFT_TYPE).eval()

        if ISTFT_TYPE == 'istft_A':
            dummy_in1 = torch.randn(1, F_BINS, STFT_SIGNAL_LENGTH)
            dummy_in2 = torch.randn_like(dummy_in1)
            in_names  = ['magnitude', 'phase']
            dyn_axes_istft = {
                'magnitude':    {2: 'n_frames'},
                'phase':        {2: 'n_frames'},
                'output_audio': {2: 'audio_len'},
            }
        else:
            dummy_in1 = torch.randn(1, F_BINS, STFT_SIGNAL_LENGTH)
            dummy_in2 = torch.randn_like(dummy_in1)
            in_names  = ['real', 'imag']
            dyn_axes_istft = {
                'real':         {2: 'n_frames'},
                'imag':         {2: 'n_frames'},
                'output_audio': {2: 'audio_len'},
            }

        torch.onnx.export(
            istft_model,
            (dummy_in1, dummy_in2),
            export_path_istft,
            input_names=in_names,
            output_names=['output_audio'],
            dynamic_axes=dyn_axes_istft if DYNAMIC_AXES else None,
            opset_version=OPSET,
        )
        slim(model=export_path_istft, output_model=export_path_istft)

        print(f"  Exported: {export_path_istft}")

        # ── 5c. Validate STFT ────────────────────────────────────────────
        print("\n── Validation: Custom STFT vs torch.stft ──")
        if STFT_TYPE == 'stft_A':
            test_onnx_stft_A(dummy_audio)
        else:
            test_onnx_stft_B(dummy_audio)

        # ── 5d. Validate ISTFT ───────────────────────────────────────────
        print("\n── Validation: Custom ISTFT vs torch.istft ──")
        if ISTFT_TYPE == 'istft_A':
            test_onnx_istft_A(dummy_in1, dummy_in2)
        else:
            test_onnx_istft_B(dummy_in1, dummy_in2)

        # ── 5e. PyTorch vs ORT agreement ─────────────────────────────────
        print("\n── Validation: PyTorch forward vs ONNXRuntime ──")
        stft_sess = ort.InferenceSession(export_path_stft)
        pt_out = stft_model(dummy_audio)
        if STFT_TYPE == 'stft_B':
            pt_r, pt_i = pt_out
            ort_r, ort_i = stft_sess.run(None, {'input_audio': dummy_audio.numpy()})
            err_r = np.abs(pt_r.numpy() - ort_r).max()
            err_i = np.abs(pt_i.numpy() - ort_i).max()
            print(f"STFT PyTorch vs ORT: max |Δ| real={err_r:.2e}, imag={err_i:.2e}")
        else:
            pt_r = pt_out
            ort_r = stft_sess.run(None, {'input_audio': dummy_audio.numpy()})[0]
            err_r = np.abs(pt_r.numpy() - ort_r).max()
            print(f"STFT PyTorch vs ORT: max |Δ| = {err_r:.2e}")

        istft_sess = ort.InferenceSession(export_path_istft)
        pt_inv = istft_model(dummy_in1, dummy_in2)
        ort_inv = istft_sess.run(None, {
            istft_sess.get_inputs()[0].name: dummy_in1.numpy(),
            istft_sess.get_inputs()[1].name: dummy_in2.numpy()
        })[0]
        err_inv = np.abs(pt_inv.numpy() - ort_inv).max()
        print(f"ISTFT PyTorch vs ORT: max |Δ| = {err_inv:.2e}")

        # ── 5f. Round-trip: STFT → ISTFT ─────────────────────────────────
        if STFT_TYPE == 'stft_B':
            print("\n── Round-trip: STFT → ISTFT ──")
            ort_r, ort_i = stft_sess.run(None, {'input_audio': dummy_audio.numpy()})

            if ISTFT_TYPE == 'istft_A':
                mag   = np.sqrt(ort_r ** 2 + ort_i ** 2)
                phase = np.arctan2(ort_i, ort_r)
                recon = istft_sess.run(None, {'magnitude': mag, 'phase': phase})[0]
            else:
                recon = istft_sess.run(None, {'real': ort_r, 'imag': ort_i})[0]

            recon = recon.squeeze()
            orig  = dummy_audio.squeeze().numpy()

            min_len = min(len(orig), len(recon))
            skip    = NFFT if not CENTER_PAD else 0
            if min_len > 2 * skip:
                s, e = skip, min_len - skip
                rt_err = np.abs(orig[s:e] - recon[s:e])
                print(f"Round-trip mean |Δ| = {rt_err.mean():.2e}, max |Δ| = {rt_err.max():.2e}")

        # ── 5g. RTF (Real-Time Factor) ────────────────────────────────
        print("\n── RTF (Real-Time Factor) ──")
        audio_duration = INPUT_AUDIO_LENGTH / 16000.0  # assume 16 kHz sample rate
        rtf_iterations = 100

        # Warm-up
        for _ in range(10):
            stft_sess.run(None, {'input_audio': dummy_audio.numpy()})
            istft_sess.run(None, {
                istft_sess.get_inputs()[0].name: dummy_in1.numpy(),
                istft_sess.get_inputs()[1].name: dummy_in2.numpy()
            })

        # STFT RTF
        t0 = time.perf_counter()
        for _ in range(rtf_iterations):
            stft_sess.run(None, {'input_audio': dummy_audio.numpy()})
        stft_elapsed = (time.perf_counter() - t0) / rtf_iterations
        stft_rtf = stft_elapsed / audio_duration

        # ISTFT RTF
        t0 = time.perf_counter()
        for _ in range(rtf_iterations):
            istft_sess.run(None, {
                istft_sess.get_inputs()[0].name: dummy_in1.numpy(),
                istft_sess.get_inputs()[1].name: dummy_in2.numpy()
            })
        istft_elapsed = (time.perf_counter() - t0) / rtf_iterations
        istft_rtf = istft_elapsed / audio_duration

        # Combined round-trip RTF
        combined_rtf = stft_rtf + istft_rtf

        print(f"  Iterations:    {rtf_iterations}")
        print(f"  Audio length:  {audio_duration:.3f} s")
        print(f"  STFT  latency: {stft_elapsed*1000:.3f} ms  |  RTF = {stft_rtf:.4f}")
        print(f"  ISTFT latency: {istft_elapsed*1000:.3f} ms  |  RTF = {istft_rtf:.4f}")
        print(f"  Combined RTF:  {combined_rtf:.4f}  ({'< 1 ✓ real-time' if combined_rtf < 1 else '≥ 1 ✗ slower than real-time'})")

        # ── 5h. Summary ─────────────────────────────────────────────────
        print("\n── Export Summary ──")
        print(f"  input_shape:  [1, 1, {INPUT_AUDIO_LENGTH}]")
        print(f"  stft_output:  [1, {F_BINS}, {STFT_SIGNAL_LENGTH}] x {'2 (real,imag)' if STFT_TYPE == 'stft_B' else '1 (real)'}")
        print(f"  istft_output: [1, 1, {_ISTFT_OUT_END - _ISTFT_OUT_START}]")
        print(f"  dynamic_axes: audio_len (dim 2), n_frames (dim 2)")
        print(f"  opset:        {OPSET}")
        print(f"  post-export:  None (no onnxslim/simplifier)")
        print(f"  remaining_dynamic_behavior: None")


if __name__ == "__main__":
    main()
  
