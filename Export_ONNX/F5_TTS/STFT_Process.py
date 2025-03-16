import numpy as np
import onnxruntime as ort
import torch

# To export your own STFT process ONNX model, set the following values.

DYNAMIC_AXES = True                                 # Default dynamic axes is input audio (signal) length.
NFFT = 1024                                         # Number of FFT components for the STFT process
HOP_LENGTH = 256                                    # Number of samples between successive frames in the STFT
INPUT_AUDIO_LENGTH = 16000                          # Set for static axes. Length of the audio input signal in samples.
MAX_SIGNAL_LENGTH = 2048                            # Maximum number of frames for the audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE = 'kaiser'                              # Type of window function used in the STFT
PAD_MODE = 'reflect'                                # Select reflect or constant
STFT_TYPE = "stft_A"                                # stft_A: output real_part only;  stft_B: outputs real_part & imag_part
ISTFT_TYPE = "istft_A"                              # istft_A: Inputs = [magnitude, phase];  istft_B: Inputs = [magnitude, real_part, imag_part], The dtype of imag_part is float format.
export_path_stft = f"{STFT_TYPE}.onnx"              # The exported stft onnx model save path.
export_path_istft = f"{ISTFT_TYPE}.onnx"            # The exported istft onnx model save path.


HALF_NFFT = NFFT // 2
STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
if NFFT > INPUT_AUDIO_LENGTH:
    NFFT = INPUT_AUDIO_LENGTH
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


# Initialize window
WINDOW = {
    'bartlett': torch.bartlett_window,
    'blackman': torch.blackman_window,
    'hamming': torch.hamming_window,
    'hann': torch.hann_window,
    'kaiser': lambda x: torch.kaiser_window(x, periodic=True, beta=12.0)
}.get(WINDOW_TYPE, torch.hann_window)(NFFT).float()
# Without a padding process, the window length must equal the NFFT length.


class STFT_Process(torch.nn.Module):
    def __init__(self, model_type, n_fft=NFFT, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE):
        super(STFT_Process, self).__init__()
        self.model_type = model_type
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.max_frames = max_frames
        self.window_type = window_type
        self.half_n_fft = self.n_fft // 2
        window = {
            'bartlett': torch.bartlett_window,
            'blackman': torch.blackman_window,
            'hamming': torch.hamming_window,
            'hann': torch.hann_window,
            'kaiser': lambda x: torch.kaiser_window(x, periodic=True, beta=12.0)
        }.get(self.window_type, torch.hann_window)(self.n_fft).float()
        if self.model_type in ['stft_A', 'stft_B']:
            time_steps = torch.arange(self.n_fft).unsqueeze(0).float()
            frequencies = torch.arange(self.half_n_fft + 1).unsqueeze(1).float()
            omega = 2 * torch.pi * frequencies * time_steps / self.n_fft
            window = window.unsqueeze(0)
            self.register_buffer('cos_kernel', (torch.cos(omega) * window).unsqueeze(1))
            self.register_buffer('sin_kernel', (-torch.sin(omega) * window).unsqueeze(1))
            self.padding_zero = torch.zeros((1, 1, self.half_n_fft), dtype=torch.float32)

        elif self.model_type in ['istft_A', 'istft_B']:
            fourier_basis = torch.fft.fft(torch.eye(self.n_fft, dtype=torch.float32))
            fourier_basis = torch.vstack([
                torch.real(fourier_basis[:self.half_n_fft + 1, :]),
                torch.imag(fourier_basis[:self.half_n_fft + 1, :])
            ]).float()
            forward_basis = window * fourier_basis[:, None, :]
            inverse_basis = window * torch.linalg.pinv((fourier_basis * self.n_fft) / self.hop_len).T[:, None, :]
            n = self.n_fft + self.hop_len * (self.max_frames - 1)
            window_sum = torch.zeros(n, dtype=torch.float32)
            window_normalized = window / window.abs().max()
            total_pad = self.n_fft - window_normalized.shape[0]
            pad_left = total_pad // 2
            pad_right = total_pad - pad_left
            win_sq = torch.nn.functional.pad(window_normalized ** 2, (pad_left, pad_right), mode='constant', value=0)

            for i in range(self.max_frames):
                sample = i * self.hop_len
                window_sum[sample: min(n, sample + self.n_fft)] += win_sq[: max(0, min(self.n_fft, n - sample))]
            self.register_buffer("forward_basis", forward_basis)
            self.register_buffer("inverse_basis", inverse_basis)
            self.register_buffer("window_sum_inv", self.n_fft / (window_sum * self.hop_len))

    def forward(self, *args):
        if self.model_type == 'stft_A':
            return self.stft_A_forward(*args)
        if self.model_type == 'stft_B':
            return self.stft_B_forward(*args)
        elif self.model_type == 'istft_A':
            return self.istft_A_forward(*args)
        elif self.model_type== 'istft_B':
            return self.istft_B_forward(*args)

    def stft_A_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        if pad_mode == 'reflect':
            x = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=pad_mode)
        else:
            x = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)
        real_part = torch.nn.functional.conv1d(x, self.cos_kernel, stride=self.hop_len)
        return real_part

    def stft_B_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        if pad_mode == 'reflect':
            x = torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode=pad_mode)
        else:
            x = torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)
        real_part = torch.nn.functional.conv1d(x, self.cos_kernel, stride=self.hop_len)
        image_part = torch.nn.functional.conv1d(x, self.sin_kernel, stride=self.hop_len)
        return real_part, image_part

    def istft_A_forward(self, magnitude, phase):
        inverse_transform = torch.nn.functional.conv_transpose1d(
            torch.cat((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=1),
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )
        output = inverse_transform[:, :, self.half_n_fft: -self.half_n_fft] * self.window_sum_inv[self.half_n_fft: inverse_transform.size(-1) - self.half_n_fft]
        return output

    def istft_B_forward(self, magnitude, real, imag):
        phase = torch.atan2(imag, real)
        inverse_transform = torch.nn.functional.conv_transpose1d(
            torch.cat((magnitude * torch.cos(phase), magnitude * torch.sin(phase)), dim=1),
            self.inverse_basis,
            stride=self.hop_len,
            padding=0,
        )
        output = inverse_transform[:, :, self.half_n_fft: -self.half_n_fft] * self.window_sum_inv[self.half_n_fft: inverse_transform.size(-1) - self.half_n_fft]
        return output


def test_onnx_stft_A(input_signal):
    torch_stft_output = torch.view_as_real(torch.stft(
        input_signal.squeeze(0),
        n_fft=NFFT,
        hop_length=HOP_LENGTH,
        win_length=NFFT,
        return_complex=True,
        window=WINDOW,
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
        win_length=NFFT,
        return_complex=True,
        window=WINDOW,
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
    pytorch_istft = torch.istft(complex_spectrum, n_fft=NFFT, hop_length=HOP_LENGTH, window=WINDOW).squeeze().numpy()
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
    pytorch_istft = torch.istft(complex_spectrum, n_fft=NFFT, hop_length=HOP_LENGTH, window=WINDOW).squeeze().numpy()
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
            dynamic_axes=dynamic_axes_stft if DYNAMIC_AXES else None,   # Set None for static using
            export_params=True,
            opset_version=17,
            do_constant_folding=True
        )

        print("\nStart Export Custom ISTFT")
        istft_model = STFT_Process(model_type=ISTFT_TYPE).eval()
        dynamic_axes_istft = {}
        if ISTFT_TYPE == 'istft_A':
            dummy_istft_input = tuple(torch.randn((1, HALF_NFFT + 1, STFT_SIGNAL_LENGTH), dtype=torch.float32) for _ in range(2))
            input_names = ["magnitude", "phase"]
        else:
            dummy_istft_input = tuple(torch.randn((1, HALF_NFFT + 1, STFT_SIGNAL_LENGTH), dtype=torch.float32) for _ in range(3))
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