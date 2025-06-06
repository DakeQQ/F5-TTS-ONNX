import numpy as np
import onnxruntime as ort
import torch

# To export your own STFT process ONNX model, set the following values.
# Next, click the IDE Run button or Launch the cmd to run 'python STFT_Process.py'

DYNAMIC_AXES = True         # Default dynamic axes is input audio (signal) length.
NFFT = 512                  # Number of FFT components for the STFT process
WIN_LENGTH = 400            # Length of the window function (can be different from NFFT)
HOP_LENGTH = 160            # Number of samples between successive frames in the STFT
INPUT_AUDIO_LENGTH = 16000  # Set for static axes. Length of the audio input signal in samples.
MAX_SIGNAL_LENGTH = 2048    # Maximum number of frames for the audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'kaiser'      # Type of window function used in the STFT
PAD_MODE = 'reflect'        # Select reflect or constant
STFT_TYPE = "stft_A"        # stft_A: output real_part only;  stft_B: outputs real_part & imag_part
ISTFT_TYPE = "istft_A"      # istft_A: Inputs = [magnitude, phase];  istft_B: Inputs = [magnitude, real_part, imag_part], The dtype of imag_part is float format.
export_path_stft = f"{STFT_TYPE}.onnx"      # The exported stft onnx model save path.
export_path_istft = f"{ISTFT_TYPE}.onnx"    # The exported istft onnx model save path.

# Precompute constants to avoid calculations at runtime
HALF_NFFT = NFFT // 2
STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1

# Sanity checks for parameters
NFFT = min(NFFT, INPUT_AUDIO_LENGTH)
WIN_LENGTH = min(WIN_LENGTH, NFFT)  # Window length cannot exceed NFFT
HOP_LENGTH = min(HOP_LENGTH, INPUT_AUDIO_LENGTH)

# Create window function lookup once
WINDOW_FUNCTIONS = {
    'bartlett': torch.bartlett_window,
    'blackman': torch.blackman_window,
    'hamming': torch.hamming_window,
    'hann': torch.hann_window,
    'kaiser': lambda x: torch.kaiser_window(x, periodic=True, beta=12.0)
}
# Define default window function
DEFAULT_WINDOW_FN = torch.hann_window


def create_padded_window(win_length, n_fft, window_type):
    """Create a window of win_length and pad/truncate to n_fft size"""
    window_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    window = window_fn(win_length).float()

    if win_length == n_fft:
        return window
    elif win_length < n_fft:
        # Pad the window to n_fft size
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        return torch.nn.functional.pad(window, (pad_left, pad_right), mode='constant', value=0)
    else:
        # Truncate the window to n_fft size (this shouldn't happen with our sanity check above)
        start = (win_length - n_fft) // 2
        return window[start:start + n_fft]


# Initialize window - only compute once
WINDOW = create_padded_window(WIN_LENGTH, NFFT, WINDOW_TYPE)


class STFT_Process(torch.nn.Module):
    def __init__(self, model_type, n_fft=NFFT, win_length=WIN_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH,
                 window_type=WINDOW_TYPE):
        super(STFT_Process, self).__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_len = hop_len
        self.max_frames = max_frames
        self.window_type = window_type
        self.half_n_fft = n_fft // 2  # Precompute once

        # Create the properly sized window
        window = create_padded_window(win_length, n_fft, window_type)

        # Register common buffers for all model types
        self.register_buffer('padding_zero', torch.zeros((1, 1, self.half_n_fft), dtype=torch.float32))

        # Pre-compute model-specific buffers
        if self.model_type in ['stft_A', 'stft_B']:
            # STFT forward pass preparation
            time_steps = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
            frequencies = torch.arange(self.half_n_fft + 1, dtype=torch.float32).unsqueeze(1)

            # Calculate omega matrix once
            omega = 2 * torch.pi * frequencies * time_steps / n_fft

            # Register conv kernels as buffers
            self.register_buffer('cos_kernel', (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1))
            self.register_buffer('sin_kernel', (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1))

        if self.model_type in ['istft_A', 'istft_B']:
            # ISTFT forward pass preparation
            # Pre-compute fourier basis
            fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
            fourier_basis = torch.vstack([
                torch.real(fourier_basis[:self.half_n_fft + 1, :]),
                torch.imag(fourier_basis[:self.half_n_fft + 1, :])
            ]).float()

            # Create forward and inverse basis
            forward_basis = window * fourier_basis.unsqueeze(1)
            inverse_basis = window * torch.linalg.pinv((fourier_basis * n_fft) / hop_len).T.unsqueeze(1)

            # Calculate window sum for overlap-add
            n = n_fft + hop_len * (max_frames - 1)
            window_sum = torch.zeros(n, dtype=torch.float32)

            # For window sum calculation, we need the original window (not padded to n_fft)
            original_window = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)(win_length).float()
            window_normalized = original_window / original_window.abs().max()

            # Pad the original window to n_fft for overlap-add calculation
            if win_length < n_fft:
                pad_left = (n_fft - win_length) // 2
                pad_right = n_fft - win_length - pad_left
                win_sq = torch.nn.functional.pad(window_normalized ** 2, (pad_left, pad_right), mode='constant',
                                                 value=0)
            else:
                win_sq = window_normalized ** 2

            # Calculate overlap-add weights
            for i in range(max_frames):
                sample = i * hop_len
                window_sum[sample: min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]

            # Register buffers
            self.register_buffer("forward_basis", forward_basis)
            self.register_buffer("inverse_basis", inverse_basis)
            self.register_buffer("window_sum_inv",
                                 n_fft / (window_sum * hop_len + 1e-7))  # Add epsilon to avoid division by zero

    def forward(self, *args):
        # Use direct method calls instead of if-else cascade for better ONNX export
        if self.model_type == 'stft_A':
            return self.stft_A_forward(*args)
        if self.model_type == 'stft_B':
            return self.stft_B_forward(*args)
        if self.model_type == 'istft_A':
            return self.istft_A_forward(*args)
        if self.model_type == 'istft_B':
            return self.istft_B_forward(*args)
        # In case none match, raise an error
        raise ValueError(f"Unknown model type: {self.model_type}")

    def stft_A_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        if pad_mode == 'reflect':
            x_padded = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=pad_mode)
        else:
            x_padded = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)

        # Single conv operation
        return torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)

    def stft_B_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        if pad_mode == 'reflect':
            x_padded = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=pad_mode)
        else:
            x_padded = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)

        # Perform convolutions
        real_part = torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)
        image_part = torch.nn.functional.conv1d(x_padded, self.sin_kernel, stride=self.hop_len)

        return real_part, image_part

    def istft_A_forward(self, magnitude, phase):
        # Pre-compute trig values
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Prepare input for transposed convolution
        complex_input = torch.cat((magnitude * cos_phase, magnitude * sin_phase), dim=1)

        # Perform transposed convolution
        inverse_transform = torch.nn.functional.conv_transpose1d(
            complex_input,
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )

        # Apply window correction
        output_len = inverse_transform.size(-1)
        start_idx = self.half_n_fft
        end_idx = output_len - self.half_n_fft

        return inverse_transform[:, :, start_idx:end_idx] * self.window_sum_inv[start_idx:end_idx]

    def istft_B_forward(self, magnitude, real, imag):
        # Calculate phase using atan2
        phase = torch.atan2(imag, real)

        # Pre-compute trig values directly instead of calling istft_A_forward
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Prepare input for transposed convolution
        complex_input = torch.cat((magnitude * cos_phase, magnitude * sin_phase), dim=1)

        # Perform transposed convolution
        inverse_transform = torch.nn.functional.conv_transpose1d(
            complex_input,
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )

        # Apply window correction
        output_len = inverse_transform.size(-1)
        start_idx = self.half_n_fft
        end_idx = output_len - self.half_n_fft

        return inverse_transform[:, :, start_idx:end_idx] * self.window_sum_inv[start_idx:end_idx]


def test_onnx_stft_A(input_signal):
    torch_stft_output = torch.view_as_real(torch.stft(
        input_signal.squeeze(0),
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        return_complex=True,
        window=WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(WIN_LENGTH),
        pad_mode=PAD_MODE,
        center=True
    ))
    pytorch_stft_real = torch_stft_output[..., 0].squeeze().numpy()
    ort_session = ort.InferenceSession(export_path_stft)
    ort_inputs = {ort_session.get_inputs()[0].name: input_signal.numpy()}
    onnx_stft_real = ort_session.run(None, ort_inputs)[0].squeeze()
    mean_diff_real = np.abs(pytorch_stft_real - onnx_stft_real).mean()
    print("\nSTFT Result: Mean Difference =", mean_diff_real)


def test_onnx_stft_B(input_signal):
    torch_stft_output = torch.view_as_real(torch.stft(
        input_signal.squeeze(0),
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        return_complex=True,
        window=WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(WIN_LENGTH),
        pad_mode=PAD_MODE,
        center=True
    ))
    pytorch_stft_real = torch_stft_output[..., 0].squeeze().numpy()
    pytorch_stft_imag = torch_stft_output[..., 1].squeeze().numpy()
    ort_session = ort.InferenceSession(export_path_stft)
    ort_inputs = {ort_session.get_inputs()[0].name: input_signal.numpy()}
    onnx_stft_real, onnx_stft_imag = ort_session.run(None, ort_inputs)
    onnx_stft_real = onnx_stft_real.squeeze()
    onnx_stft_imag = onnx_stft_imag.squeeze()
    mean_diff_real = np.abs(pytorch_stft_real - onnx_stft_real).mean()
    mean_diff_imag = np.abs(pytorch_stft_imag - onnx_stft_imag).mean()
    mean_diff = (mean_diff_real + mean_diff_imag) * 0.5
    print("\nSTFT Result: Mean Difference =", mean_diff)


def test_onnx_istft_A(magnitude, phase):
    complex_spectrum = torch.polar(magnitude, phase)
    pytorch_istft = torch.istft(
        complex_spectrum,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(WIN_LENGTH)
    ).squeeze().numpy()
    ort_session = ort.InferenceSession(export_path_istft)
    ort_inputs = {
        ort_session.get_inputs()[0].name: magnitude.numpy(),
        ort_session.get_inputs()[1].name: phase.numpy()
    }
    onnx_istft = ort_session.run(None, ort_inputs)[0].squeeze()
    print("\nISTFT Result: Mean Difference =", np.abs(onnx_istft - pytorch_istft).mean())


def test_onnx_istft_B(magnitude, real, imag):
    phase = torch.atan2(imag, real)
    complex_spectrum = torch.polar(magnitude, phase)
    pytorch_istft = torch.istft(
        complex_spectrum,
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW_FUNCTIONS.get(WINDOW_TYPE, DEFAULT_WINDOW_FN)(WIN_LENGTH)
    ).squeeze().numpy()
    ort_session = ort.InferenceSession(export_path_istft)
    ort_inputs = {
        ort_session.get_inputs()[0].name: magnitude.numpy(),
        ort_session.get_inputs()[1].name: real.numpy(),
        ort_session.get_inputs()[2].name: imag.numpy()
    }
    onnx_istft = ort_session.run(None, ort_inputs)[0].squeeze()
    print("\nISTFT Result: Mean Difference =", np.abs(onnx_istft - pytorch_istft).mean())


def main():
    with torch.inference_mode():
        print(f"\nConfiguration: NFFT={NFFT}, WIN_LENGTH={WIN_LENGTH}, HOP_LENGTH={HOP_LENGTH}")
        print("\nStart Export Custom STFT")
        stft_model = STFT_Process(model_type=STFT_TYPE).eval()
        dummy_stft_input = torch.randn((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.float32)
        input_names = ['input_audio']
        dynamic_axes_stft = {input_names[0]: {2: 'audio_len'}}
        if STFT_TYPE == 'stft_A':
            output_names = ['real']
        else:
            output_names = ['real', 'imag']
            dynamic_axes_stft[output_names[1]] = {2: 'signal_len'}
        dynamic_axes_stft[output_names[0]] = {2: 'signal_len'}
        torch.onnx.export(
            stft_model,
            (dummy_stft_input,),
            export_path_stft,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_stft if DYNAMIC_AXES else None,  # Set None for static using
            export_params=True,
            opset_version=17,
            do_constant_folding=True
        )

        print("\nStart Export Custom ISTFT")
        istft_model = STFT_Process(model_type=ISTFT_TYPE).eval()
        dynamic_axes_istft = {}
        if ISTFT_TYPE == 'istft_A':
            dummy_istft_input = tuple(
                torch.randn((1, HALF_NFFT + 1, STFT_SIGNAL_LENGTH), dtype=torch.float32) for _ in range(2))
            input_names = ["magnitude", "phase"]
        else:
            dummy_istft_input = tuple(
                torch.randn((1, HALF_NFFT + 1, STFT_SIGNAL_LENGTH), dtype=torch.float32) for _ in range(3))
            input_names = ["magnitude", "real", "imag"]
            dynamic_axes_istft[input_names[2]] = {2: 'signal_len'}
        dynamic_axes_istft[input_names[0]] = {2: 'signal_len'}
        dynamic_axes_istft[input_names[1]] = {2: 'signal_len'}
        output_names = ["output_audio"]
        dynamic_axes_istft[output_names[0]] = {2: 'audio_len'}
        torch.onnx.export(
            istft_model,
            dummy_istft_input,
            export_path_istft,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_istft if DYNAMIC_AXES else None,  # Set None for static using
            export_params=True,
            opset_version=17,
            do_constant_folding=True
        )

        print("\nTesting the Custom.STFT versus Pytorch.STFT ...")
        if STFT_TYPE == 'stft_A':
            test_onnx_stft_A(dummy_stft_input)
        else:
            test_onnx_stft_B(dummy_stft_input)

        print("\n\nTesting the Custom.ISTFT versus Pytorch.ISTFT ...")
        if ISTFT_TYPE == 'istft_A':
            test_onnx_istft_A(*dummy_istft_input)
        else:
            test_onnx_istft_B(*dummy_istft_input)


if __name__ == "__main__":
    main()
