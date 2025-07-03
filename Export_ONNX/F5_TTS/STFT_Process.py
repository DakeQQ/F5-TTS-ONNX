#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import onnxruntime as ort
import torch

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────
DYNAMIC_AXES = True                  # Default dynamic axes is input audio (signal) length.
NFFT = 512                           # Number of FFT components for the STFT process
WIN_LENGTH = 400                     # Length of the window function (can be different from NFFT)
HOP_LENGTH = 160                     # Number of samples between successive frames in the STFT
INPUT_AUDIO_LENGTH  = 16000          # dummy length for export / test
MAX_SIGNAL_LENGTH   = 2048           # Maximum number of frames for the audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE         = 'hann'         # bartlett | blackman | hamming | hann | kaiser
PAD_MODE            = 'constant'     # reflect | constant

STFT_TYPE  = "stft_B"                # # stft_A: output real_part only;  stft_B: outputs real_part & imag_part
ISTFT_TYPE = "istft_B"               # istft_A: Inputs = [magnitude, phase];  istft_B: Inputs = [real_part, imag_part], The dtype of imag_part is float format.

export_path_stft  = f"{STFT_TYPE}.onnx"
export_path_istft = f"{ISTFT_TYPE}.onnx"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Pre-computations / helpers
# ─────────────────────────────────────────────────────────────────────────────
HALF_NFFT          = NFFT // 2
STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1

# clip parameters to sensible ranges
NFFT       = min(NFFT, INPUT_AUDIO_LENGTH)
WIN_LENGTH = min(WIN_LENGTH, NFFT)
HOP_LENGTH = min(HOP_LENGTH, INPUT_AUDIO_LENGTH)

WINDOW_FUNCTIONS = {
    'bartlett': torch.bartlett_window,
    'blackman': torch.blackman_window,
    'hamming' : torch.hamming_window,
    'hann'    : torch.hann_window,
    'kaiser'  : lambda L: torch.kaiser_window(L, periodic=True, beta=12.0)
}
DEFAULT_WINDOW_FN = torch.hann_window


def create_padded_window(win_length, n_fft, window_type):
    """Return length-n_fft window (centre-padded / cropped if needed)."""
    win_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    win    = win_fn(win_length).float()
    if win_length == n_fft:
        return win
    if win_length < n_fft:                                   # pad
        pl = (n_fft - win_length) // 2
        pr = n_fft - win_length - pl
        return torch.nn.functional.pad(win, (pl, pr))
    # truncate (shouldn’t occur given sanity checks)
    start = (win_length - n_fft) // 2
    return win[start:start + n_fft]


WINDOW = create_padded_window(WIN_LENGTH, NFFT, WINDOW_TYPE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────────────────────────────────────
class STFT_Process(torch.nn.Module):
    def __init__(self,
                 model_type,
                 n_fft=NFFT,
                 win_length=WIN_LENGTH,
                 hop_len=HOP_LENGTH,
                 max_frames=MAX_SIGNAL_LENGTH,
                 window_type=WINDOW_TYPE):
        super().__init__()
        self.model_type  = model_type
        self.n_fft       = n_fft
        self.hop_len     = hop_len
        self.half_n_fft  = n_fft // 2

        window = create_padded_window(win_length, n_fft, window_type)

        # constant-pad buffer (for 'constant' pad mode)
        self.register_buffer('padding_zero', torch.zeros(1, 1, self.half_n_fft, dtype=torch.float32))

        # ─── kernels for STFT_A / STFT_B ───────────────────────────────────
        if model_type in ('stft_A', 'stft_B'):
            t  = torch.arange(n_fft).float().unsqueeze(0)
            f  = torch.arange(self.half_n_fft + 1).float().unsqueeze(1)
            omega = 2 * torch.pi * f * t / n_fft
            self.register_buffer(
                'cos_kernel',
                (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
            )
            self.register_buffer(
                'sin_kernel',
                (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
            )

        # ─── kernels for ISTFT_A / ISTFT_B ─────────────────────────────────
        if model_type in ('istft_A', 'istft_B'):
            fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
            fourier_basis = torch.vstack([
                torch.real(fourier_basis[:self.half_n_fft + 1]),
                torch.imag(fourier_basis[:self.half_n_fft + 1])
            ]).float()

            forward_basis = window * fourier_basis.unsqueeze(1)
            inverse_basis = window * torch.linalg.pinv(
                (fourier_basis * n_fft) / hop_len
            ).T.unsqueeze(1)

            # overlap-add weighting
            n          = n_fft + hop_len * (max_frames - 1)
            window_sum = torch.zeros(n, dtype=torch.float32)

            orig_win = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)(win_length).float()
            wn = orig_win / orig_win.abs().max()

            if win_length < n_fft:
                pl = (n_fft - win_length) // 2
                pr = n_fft - win_length - pl
                win_sq = torch.nn.functional.pad(wn ** 2, (pl, pr))
            else:
                win_sq = wn ** 2

            for i in range(max_frames):
                s = i * hop_len
                window_sum[s:s + n_fft] += win_sq[:max(0, min(n_fft, n - s))]

            self.register_buffer('forward_basis', forward_basis)
            self.register_buffer('inverse_basis', inverse_basis)
            self.register_buffer('window_sum_inv', n_fft / (window_sum * hop_len + 1e-7))

    # ───── dispatcher ──────────────────────────────────────────────────────
    def forward(self, *args):
        if self.model_type == 'stft_A':  return self.stft_A_forward(*args)
        if self.model_type == 'stft_B':  return self.stft_B_forward(*args)
        if self.model_type == 'istft_A': return self.istft_A_forward(*args)
        if self.model_type == 'istft_B': return self.istft_B_forward(*args)
        raise ValueError(self.model_type)

    # ───── STFT (A & B) ────────────────────────────────────────────────────
    def _pad_input(self, x, mode):
        if mode == 'reflect':
            return torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode='reflect')
        return torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)

    def stft_A_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        x_padded = self._pad_input(x, pad_mode)
        return torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)

    def stft_B_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        x_padded = self._pad_input(x, pad_mode)
        real = torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)
        imag = torch.nn.functional.conv1d(x_padded, self.sin_kernel, stride=self.hop_len)
        return real, imag

    # ───── ISTFT_A (magnitude, phase) ──────────────────────────────────────
    def istft_A_forward(self, magnitude, phase):
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        inp   = torch.cat((magnitude * cos_p, magnitude * sin_p), dim=1)
        inv   = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
        s, e  = self.half_n_fft, inv.size(-1) - self.half_n_fft
        return inv[:, :, s:e] * self.window_sum_inv[s:e]

    # ───── ISTFT_B (real, imag) — updated as requested ────────────────────
    def istft_B_forward(self, real, imag):
        inp = torch.cat((real, imag), dim=1)  # == cat(real, imag)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
        s, e = self.half_n_fft, inv.size(-1) - self.half_n_fft
        return inv[:, :, s:e] * self.window_sum_inv[s:e]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Test helpers  (A & B variants)
# ─────────────────────────────────────────────────────────────────────────────
def test_onnx_stft_A(x):
    torch_out = torch.view_as_real(torch.stft(
        x.squeeze(0),
        n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        return_complex=True,
        window=WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(WIN_LENGTH),
        pad_mode=PAD_MODE, center=True
    ))
    pt_real = torch_out[..., 0].squeeze().numpy()

    sess = ort.InferenceSession(export_path_stft)
    ort_real = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0].squeeze()
    print("\nSTFT Result (A): mean |Δ| =", np.abs(pt_real - ort_real).mean())


def test_onnx_stft_B(x):
    torch_out = torch.view_as_real(torch.stft(
        x.squeeze(0),
        n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        return_complex=True,
        window=WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(WIN_LENGTH),
        pad_mode=PAD_MODE, center=True
    ))
    pt_r = torch_out[..., 0].squeeze().numpy()
    pt_i = torch_out[..., 1].squeeze().numpy()

    sess = ort.InferenceSession(export_path_stft)
    ort_r, ort_i = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})
    diff = 0.5 * (np.abs(pt_r - ort_r.squeeze()).mean() +
                  np.abs(pt_i - ort_i.squeeze()).mean())
    print("\nSTFT Result (B): mean |Δ| =", diff)


def test_onnx_istft_A(mag, phase):
    complex_spec = torch.polar(mag, phase)
    pt_audio = torch.istft(
        complex_spec,
        n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(WIN_LENGTH)
    ).squeeze().numpy()

    sess = ort.InferenceSession(export_path_istft)
    ort_audio = sess.run(None, {
        sess.get_inputs()[0].name: mag.numpy(),
        sess.get_inputs()[1].name: phase.numpy()
    })[0].squeeze()
    print("\nISTFT Result (A): mean |Δ| =", np.abs(pt_audio - ort_audio).mean())


def test_onnx_istft_B(real, imag):
    pt_audio = torch.istft(
        torch.complex(real, imag),
        n_fft=NFFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        window=WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(WIN_LENGTH)
    ).squeeze().numpy()

    sess = ort.InferenceSession(export_path_istft)
    ort_audio = sess.run(None, {
        sess.get_inputs()[0].name: real.numpy(),
        sess.get_inputs()[1].name: imag.numpy()
    })[0].squeeze()
    print("\nISTFT Result (B): mean |Δ| =", np.abs(pt_audio - ort_audio).mean())


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Export & quick verification
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with torch.inference_mode():
        print(f"\nConfig  NFFT={NFFT}, WIN_LEN={WIN_LENGTH}, HOP={HOP_LENGTH}")
        # ─── STFT export ───────────────────────────────────────────────────
        stft_model   = STFT_Process(STFT_TYPE).eval()
        dummy_audio  = torch.randn(1, 1, INPUT_AUDIO_LENGTH)

        dyn_axes_sft = {'input_audio': {2: 'audio_len'}}
        if STFT_TYPE == 'stft_A':
            out_names = ['real']
            dyn_axes_sft['real'] = {2: 'signal_len'}
        else:
            out_names = ['real', 'imag']
            dyn_axes_sft['real'] = dyn_axes_sft['imag'] = {2: 'signal_len'}

        torch.onnx.export(
            stft_model, (dummy_audio,), export_path_stft,
            input_names=['input_audio'], output_names=out_names,
            dynamic_axes=dyn_axes_sft if DYNAMIC_AXES else None,
            opset_version=17, do_constant_folding=True
        )
        # ─── ISTFT export ──────────────────────────────────────────────────
        istft_model = STFT_Process(ISTFT_TYPE).eval()

        if ISTFT_TYPE == 'istft_A':
            dummy_mag   = torch.randn(1, HALF_NFFT + 1, STFT_SIGNAL_LENGTH)
            dummy_phase = torch.randn_like(dummy_mag)
            dummy_inp   = (dummy_mag, dummy_phase)
            in_names    = ['magnitude', 'phase']
            dyn_axes_ist = {
                'magnitude': {2: 'signal_len'},
                'phase'    : {2: 'signal_len'},
                'output_audio': {2: 'audio_len'}
            }
        else:  # istft_B
            dummy_real = torch.randn(1, HALF_NFFT + 1, STFT_SIGNAL_LENGTH)
            dummy_imag = torch.randn_like(dummy_real)
            dummy_inp  = (dummy_real, dummy_imag)
            in_names   = ['real', 'imag']
            dyn_axes_ist = {
                'real'  : {2: 'signal_len'},
                'imag'  : {2: 'signal_len'},
                'output_audio': {2: 'audio_len'}
            }

        torch.onnx.export(
            istft_model, dummy_inp, export_path_istft,
            input_names=in_names, output_names=['output_audio'],
            dynamic_axes=dyn_axes_ist if DYNAMIC_AXES else None,
            opset_version=17, do_constant_folding=True
        )

        # ─── quick comparisons ────────────────────────────────────────────
        print("\nTesting Custom STFT against torch.stft …")
        if STFT_TYPE == 'stft_A':
            test_onnx_stft_A(dummy_audio)
        else:
            test_onnx_stft_B(dummy_audio)

        print("\nTesting Custom ISTFT against torch.istft …")
        if ISTFT_TYPE == 'istft_A':
            test_onnx_istft_A(*dummy_inp)
        else:
            test_onnx_istft_B(*dummy_inp)


if __name__ == "__main__":
    main()
    
