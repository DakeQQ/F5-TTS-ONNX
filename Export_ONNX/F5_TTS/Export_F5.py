import gc
import re
import shutil
import site
import time
import math
import torch
import torchaudio
import jieba
import onnxruntime
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style
from vocos import Vocos
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.
python_package_path = site.getsitepackages()[-1]

test_in_english      = False                                                                                 # Test the F5-TTS-ONNX model after the export process.
use_fp16_transformer = False                                                                                 # Export the F5_Transformer.onnx in float16 format.
F5_safetensors_path  = "/home/DakeQQ/Downloads/F5TTS_v1_Base/model_1250000.safetensors"                      # The F5-TTS model download path.           URL: https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base
vocab_path           = "/home/DakeQQ/Downloads/F5TTS_v1_Base/vocab.txt"                                      # The F5-TTS model vocab download path.     URL: https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base
vocos_model_path     = "/home/DakeQQ/Downloads/vocos-mel-24khz"                                              # The Vocos model download path.            URL: https://huggingface.co/charactr/vocos-mel-24khz/tree/main
onnx_model_A         = "/home/DakeQQ/Downloads/F5_ONNX/F5_Preprocess.onnx"                                   # The exported onnx model path.
onnx_model_B         = "/home/DakeQQ/Downloads/F5_ONNX/F5_Transformer.onnx"                                  # The exported onnx model path.
onnx_model_C         = "/home/DakeQQ/Downloads/F5_ONNX/F5_Decode.onnx"                                       # The exported onnx model path.
generated_audio      = "generated.wav"                                                                       # The generated audio path.

if test_in_english:
    reference_audio  = python_package_path + "/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text         = "Some call me nature, others call me mother nature."
    gen_text         = "Some call me Dake, others call me QQ."
else:
    reference_audio  = python_package_path + "/f5_tts/infer/examples/basic/basic_ref_zh.wav"                   # The reference audio path.
    ref_text         = "对，这就是我，万人敬仰的太乙真人。"                                                          # The ASR result of reference audio.
    gen_text         = "对，这就是我，万人敬仰的大可奇奇。"                                                          # The target TTS.


# ONNX Runtime Settings
ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
# Model Parameters
DYNAMIC_AXES = True                     # Default dynamic_axes is input audio length. Note, some providers only work for static axes.
NFE_STEP = 32                           # F5-TTS model setting, 0~31
FUSE_NFE = 1                            # '1' means no fuse. '2' means fuse every 2 NFE steps into one to reduce I/O binding times.
SAMPLE_RATE = 24000                     # F5-TTS model setting
CFG_STRENGTH = 2.0                      # F5-TTS model setting
SWAY_COEFFICIENT = -1.0                 # F5-TTS model setting
TARGET_RMS = 0.15                       # The root-mean-square value for the audio
SPEED = 1.0                             # Set for talking speed. Only works with dynamic_axes=True
RANDOM_SEED = 9527                      # Set seed to reproduce the generated audio
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT. It affects the generated audio length and speech speed.

# STFT/ISTFT Settings
N_MELS = 100                            # Number of Mel bands to generate in the Mel-spectrogram
NFFT = 1024                             # Number of FFT components for the STFT process
WINDOW_LENGTH = 960                     # Length of windowing, edit it carefully.
WINDOW_TYPE = 'hann'                    # Type of window function used in the STFT
MAX_SIGNAL_LENGTH = 4096                # Max frames for audio length after STFT processed. Set an appropriate larger value for long audio input, such as 4096.

# Setting for Static Axes using
AUDIO_LENGTH = 160000                   # Set for static axes export. Length of audio input signal in samples
TEXT_IDS_LENGTH = 60                    # Set for static axes export. Text_ids from [ref_text + gen_text]
MAX_GENERATED_LENGTH = 600              # Set for static axes export. Max signal features before passing to ISTFT
TEXT_EMBED_LENGTH = 512 + N_MELS        # Set for static axes export.

# Others
REFERENCE_SIGNAL_LENGTH = AUDIO_LENGTH // HOP_LENGTH + 1       # Reference audio length after STFT processed
MAX_DURATION = REFERENCE_SIGNAL_LENGTH + MAX_GENERATED_LENGTH  # Set for static axes export. MAX_DURATION <= MAX_SIGNAL_LENGTH
if MAX_DURATION > MAX_SIGNAL_LENGTH:
    MAX_DURATION = MAX_SIGNAL_LENGTH

# Load the vocab.txt
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)

# Replace the original source code.
# Note! please re-install the vocos after the export process.
# Note! please re-install the f5-tts after the export process.
shutil.copyfile('./modeling_modified/vocos/heads.py', python_package_path + '/vocos/heads.py')
shutil.copyfile('./modeling_modified/vocos/models.py', python_package_path + '/vocos/models.py')
shutil.copyfile('./modeling_modified/vocos/modules.py', python_package_path + '/vocos/modules.py')
shutil.copyfile('./modeling_modified/vocos/pretrained.py', python_package_path + '/vocos/pretrained.py')
shutil.copyfile('./modeling_modified/F5/dit.py', python_package_path + '/f5_tts/model/backbones/dit.py')
if use_fp16_transformer:
    shutil.copyfile('./modeling_modified/F5/fp16/modules.py', python_package_path + '/f5_tts/model/modules.py')
else:
    shutil.copyfile('./modeling_modified/F5/modules.py', python_package_path + '/f5_tts/model/modules.py')


from f5_tts.model import CFM, DiT  # Do not delete DiT
from f5_tts.infer.utils_infer import load_checkpoint


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
        freqs = freqs.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0).repeat(2, num_head, 1, 1)
        self.rope_cos = freqs.cos().half()
        self.rope_sin = freqs.sin().half()
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0)
        self.inv_int16 = float(1.0 / 32768.0)
        self.use_fp16 = use_fp16

    def forward(self,
                audio: torch.ShortTensor,
                text_ids: torch.IntTensor,
                max_duration: torch.LongTensor,
                ):
        audio = audio.float() * self.inv_int16
        audio = audio * self.target_rms / torch.sqrt(torch.mean(audio * audio))  # Optional process
        mel_signal_real, mel_signal_imag = self.custom_stft(audio, 'reflect')
        mel_signal = torch.matmul(self.fbank, torch.sqrt(mel_signal_real * mel_signal_real + mel_signal_imag * mel_signal_imag)).transpose(1, 2).clamp(min=1e-5).log()
        ref_signal_len = mel_signal.shape[1]
        zeros = torch.zeros((1, max_duration, self.num_channels), dtype=torch.float32)
        zeros_split_A = zeros[:, :-ref_signal_len]
        zeros_split_B = zeros[:, :-text_ids.shape[-1], 0]
        mel_signal = torch.cat((mel_signal, zeros_split_A), dim=1)
        noise = torch.randn_like(zeros)
        rope_cos_q = self.rope_cos[:, :, :max_duration]
        rope_sin_q = self.rope_sin[:, :, :max_duration]
        rope_cos_k = rope_cos_q.transpose(-1, -2)
        rope_sin_k = rope_sin_q.transpose(-1, -2)
        text, text_drop = self.f5_text_embed(torch.cat((text_ids + 1, zeros_split_B.to(text_ids.dtype)), dim=-1), max_duration[0])
        cat_mel_text = torch.cat((mel_signal, text), dim=-1)
        cat_mel_text_drop = torch.cat((zeros, text_drop), dim=-1)
        if self.use_fp16:
            return noise.half(), rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, cat_mel_text.half(), cat_mel_text_drop.half(), ref_signal_len
        return noise, rope_cos_q.float(), rope_sin_q.float(), rope_cos_k.float(), rope_sin_k.float(), cat_mel_text, cat_mel_text_drop, ref_signal_len


class F5Transformer(torch.nn.Module):
    def __init__(self, f5_model, cfg, steps, sway_coef, dtype, fuse_step):
        super(F5Transformer, self).__init__()
        self.f5_transformer = f5_model.transformer
        self.time_mlp = f5_model.transformer.time_embed.time_mlp
        self.freq_embed_dim = 256
        self.time_mlp_dim = 1024
        self.cfg_strength = cfg
        self.sway_sampling_coef = sway_coef
        t = torch.linspace(0, 1, steps, dtype=torch.float32)
        time_step = t + self.sway_sampling_coef * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
        self.delta_t = torch.diff(time_step).to(dtype)
        self.time_expand = torch.zeros((1, len(time_step), self.time_mlp_dim), dtype=torch.float32)
        half_dim = self.freq_embed_dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
        for i in range(len(time_step)):
            emb = time_step[i] * emb_factor
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
            self.time_expand[:, [i]] = self.time_mlp(emb)
        self.time_expand = self.time_expand.to(dtype)
        self.fuse_step = fuse_step

    def forward(self,
                noise: torch.FloatTensor,
                rope_cos_q: torch.FloatTensor,
                rope_sin_q: torch.FloatTensor,
                rope_cos_k: torch.FloatTensor,
                rope_sin_k: torch.FloatTensor,
                cat_mel_text: torch.FloatTensor,
                cat_mel_text_drop: torch.FloatTensor,
                time_step: torch.IntTensor
                ):
        for nfe in range(self.fuse_step):
            pred = self.f5_transformer(x=noise, cond=cat_mel_text, cond_drop=cat_mel_text_drop, time=self.time_expand[:, time_step], rope_cos_q=rope_cos_q, rope_sin_q=rope_sin_q, rope_cos_k=rope_cos_k, rope_sin_k=rope_sin_k)
            pred, pred1 = torch.split(pred, [1, 1], dim=0)
            noise += (pred + (pred - pred1) * self.cfg_strength) * self.delta_t[time_step]
            time_step += 1
        return noise, time_step


class F5Decode(torch.nn.Module):
    def __init__(self, vocos, custom_istft, target_rms, use_fp16):
        super(F5Decode, self).__init__()
        self.vocos = vocos
        self.custom_istft = custom_istft
        self.target_rms = float(target_rms)
        self.use_fp16 = use_fp16

    def forward(self,
                denoised: torch.FloatTensor,
                ref_signal_len: torch.LongTensor
                ):
        denoised = denoised[:, ref_signal_len:]
        if self.use_fp16:
            denoised = denoised.float()
        denoised = self.vocos.decode(denoised.transpose(1, 2))
        generated_signal = self.custom_istft(*denoised)
        generated_signal = generated_signal * self.target_rms / torch.sqrt(torch.mean(generated_signal * generated_signal))  # Optional process
        return (generated_signal * 32768.0).clamp(min=-32768.0, max=32767.0).to(torch.int16)


def load_model(ckpt_path):
    model_cfg = OmegaConf.load(python_package_path + "/f5_tts/configs/F5TTS_v1_Base.yaml")
    model_cls = globals()[model_cfg.model.backbone]
    model = CFM(
        transformer=model_cls(**model_cfg.model.arch, text_num_embeds=vocab_size, mel_dim=N_MELS),
        mel_spec_kwargs=dict(  # Not important here. Use the custom STFT/ISTFT instead.
            target_sample_rate=SAMPLE_RATE,
            n_mel_channels=N_MELS,
            hop_length=HOP_LENGTH,
        ),
        odeint_kwargs=dict(
            method='euler',     # Only the Euler method is implemented for ONNX here.
        ),
        vocab_char_map=vocab_char_map,
    ).to('cpu')
    return load_checkpoint(model, ckpt_path, 'cpu', use_ema=True), model_cfg.model.arch.heads, model_cfg.model.arch.dim


# From the official code
def convert_char_to_pinyin(text_list, polyphone=True):
    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True))
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)
    return final_text_list


# From the official code
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1
):
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    text = torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)
    

print("\n\nStart to Export the F5-TTS Preprocess Part.")
with torch.inference_mode():
    # Dummy for Export the F5_Preprocess part
    audio = torch.ones((1, 1, AUDIO_LENGTH), dtype=torch.int16)
    text_ids = torch.ones((1, TEXT_IDS_LENGTH), dtype=torch.int32)
    max_duration = torch.tensor([MAX_DURATION], dtype=torch.long)
    f5_model, NUM_HEAD, HIDDEN_SIZE = load_model(F5_safetensors_path)
    HEAD_DIM = HIDDEN_SIZE // NUM_HEAD
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    f5_preprocess = F5Preprocess(f5_model, custom_stft, nfft=NFFT, n_mels=N_MELS, sample_rate=SAMPLE_RATE, num_head=NUM_HEAD, head_dim=HEAD_DIM, target_rms=TARGET_RMS, use_fp16=use_fp16_transformer)
    torch.onnx.export(
        f5_preprocess,
        (audio, text_ids, max_duration),
        onnx_model_A,
        input_names=['audio', 'text_ids', 'max_duration'],
        output_names=['noise', 'rope_cos_q', 'rope_sin_q', 'rope_cos_k', 'rope_sin_k', 'cat_mel_text', 'cat_mel_text_drop', 'ref_signal_len'],
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'text_ids': {1: 'text_ids_len'},
            'noise': {1: 'max_duration'},
            'rope_cos_q': {2: 'max_duration'},
            'rope_sin_q': {2: 'max_duration'},
            'rope_cos_k': {3: 'max_duration'},
            'rope_sin_k': {3: 'max_duration'},
            'cat_mel_text': {1: 'max_duration'},
            'cat_mel_text_drop': {1: 'max_duration'}
        } if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17)
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
    # Pre scale
    for i in range(len(f5_model.transformer.transformer_blocks)):
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.to_q.weight.data *= scale_factor
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.to_q.bias.data *= scale_factor
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.to_k.weight.data *= scale_factor
        f5_model.transformer.transformer_blocks._modules[f'{i}'].attn.to_k.bias.data *= scale_factor

    noise = torch.ones((1, MAX_DURATION, N_MELS), dtype=dtype)
    rope_cos_q = torch.ones((2, NUM_HEAD, MAX_DURATION, HEAD_DIM), dtype=dtype)
    rope_sin_q = torch.ones((2, NUM_HEAD, MAX_DURATION, HEAD_DIM), dtype=dtype)
    rope_cos_k = rope_cos_q.transpose(-1, -2)
    rope_sin_k = rope_sin_q.transpose(-1, -2)
    cat_mel_text = torch.ones((1, MAX_DURATION, TEXT_EMBED_LENGTH), dtype=dtype)
    cat_mel_text_drop = torch.ones((1, MAX_DURATION, TEXT_EMBED_LENGTH), dtype=dtype)
    time_step = torch.tensor([0], dtype=torch.int32)
    if FUSE_NFE > 1:
        print('\nNote: NFE fusion is exporting. It may take a long time and create a large ONNX graph, potentially causing Netron to fail or increasing model loading time.')
    else:
        fuse_nfe_step = 1
    f5_transformer = F5Transformer(f5_model, cfg=CFG_STRENGTH, steps=NFE_STEP, sway_coef=SWAY_COEFFICIENT, dtype=dtype, fuse_step=FUSE_NFE)
    if use_fp16_transformer:
        f5_transformer = f5_transformer.half()
    torch.onnx.export(
        f5_transformer,
        (noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, cat_mel_text, cat_mel_text_drop, time_step),
        onnx_model_B,
        input_names=['noise', 'rope_cos_q', 'rope_sin_q', 'rope_cos_k', 'rope_sin_k', 'cat_mel_text', 'cat_mel_text_drop', 'time_step'],
        output_names=['denoised', 'time_step'],
        dynamic_axes={
            'noise': {1: 'max_duration'},
            'rope_cos_q': {2: 'max_duration'},
            'rope_sin_q': {2: 'max_duration'},
            'rope_cos_k': {3: 'max_duration'},
            'rope_sin_k': {3: 'max_duration'},
            'cat_mel_text': {1: 'max_duration'},
            'cat_mel_text_drop': {1: 'max_duration'},
            'denoised': {1: 'max_duration'}
        } if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17)
    del f5_transformer
    del noise
    del rope_cos_q
    del rope_sin_q
    del rope_cos_k
    del rope_sin_k
    del cat_mel_text
    del cat_mel_text_drop
    del time_step
    gc.collect()
    print("\nExport Done.")


print("\n\nStart to Export the F5-TTS Decode Part.")
with torch.inference_mode():
    # Dummy for Export the F5_Decode part
    denoised = torch.ones((1, MAX_DURATION, N_MELS), dtype=dtype)
    ref_signal_len = torch.tensor(REFERENCE_SIGNAL_LENGTH, dtype=torch.long)
    custom_istft = STFT_Process(model_type='istft_A', n_fft=NFFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=MAX_SIGNAL_LENGTH, window_type=WINDOW_TYPE).eval()
    # Vocos model preprocess
    vocos = Vocos.from_pretrained(vocos_model_path)
    vocos.backbone.norm.weight.data = (vocos.backbone.norm.weight.data * torch.sqrt(torch.tensor(vocos.backbone.norm.weight.data.shape[0], dtype=torch.float32))).view(1, -1, 1)
    vocos.backbone.norm.bias.data = vocos.backbone.norm.bias.data.view(1, -1, 1)
    vocos.backbone.final_layer_norm.weight.data = (vocos.backbone.final_layer_norm.weight.data * torch.sqrt(torch.tensor(vocos.backbone.final_layer_norm.weight.data.shape[0], dtype=torch.float32))).view(1, -1, 1)
    vocos.backbone.final_layer_norm.bias.data = vocos.backbone.final_layer_norm.bias.data.view(1, -1, 1)
    vocos.head.out.bias.data = vocos.head.out.bias.data.view(1, -1, 1)
    for i in range(len(vocos.backbone.convnext)):
        block = vocos.backbone.convnext._modules[f'{i}']
        block.norm.weight.data = (block.norm.weight.data * torch.sqrt(torch.tensor(block.norm.weight.data.shape[0], dtype=torch.float32))).view(1, -1, 1)
        block.norm.bias.data = block.norm.bias.data.view(1, -1, 1)
        block.pwconv1.weight.data = block.pwconv1.weight.data.unsqueeze(0)
        block.pwconv1.bias.data = block.pwconv1.bias.data.view(1, -1, 1)
        block.pwconv2.weight.data = (block.gamma.data.unsqueeze(-1) * block.pwconv2.weight.data).unsqueeze(0)
        block.pwconv2.bias.data = (block.gamma.data * block.pwconv2.bias.data).view(1, -1, 1)

    f5_decode = F5Decode(vocos, custom_istft, target_rms=TARGET_RMS, use_fp16=use_fp16_transformer)
    torch.onnx.export(
        f5_decode,
        (denoised, ref_signal_len),
        onnx_model_C,
        input_names=['denoised', 'ref_signal_len'],
        output_names=['output_audio'],
        dynamic_axes={
            'denoised': {1: 'max_duration'},
            'output_audio': {2: 'generated_len'},
        } if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17)
    del f5_decode
    del denoised
    del ref_signal_len
    del vocos
    del custom_istft
    gc.collect()
    print("\nExport Done.")


# ONNX Runtime settings
onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4        # fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
in_name_A2 = in_name_A[2].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name
out_name_A3 = out_name_A[3].name
out_name_A4 = out_name_A[4].name
out_name_A5 = out_name_A[5].name
out_name_A6 = out_name_A[6].name
out_name_A7 = out_name_A[7].name


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers)
# For Windows DirectML + Intel/AMD/Nvidia GPU,
# pip install onnxruntime-directml --upgrade
# ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['DmlExecutionProvider'])

in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
in_name_B1 = in_name_B[1].name
in_name_B2 = in_name_B[2].name
in_name_B3 = in_name_B[3].name
in_name_B4 = in_name_B[4].name
in_name_B5 = in_name_B[5].name
in_name_B6 = in_name_B[6].name
in_name_B7 = in_name_B[7].name
out_name_B0 = out_name_B[0].name
out_name_B1 = out_name_B[1].name


ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name


# Run F5-TTS by ONNX Runtime
audio = np.array(AudioSegment.from_file(reference_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)

zh_pause_punc = r"。，、；：？！"
ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
ref_audio_len = audio.shape[-1] // HOP_LENGTH + 1
max_duration = np.array([ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED)], dtype=np.int64)
gen_text = convert_char_to_pinyin([ref_text + gen_text])
text_ids = list_str_to_idx(gen_text, vocab_char_map).numpy()
time_step = np.array([0], dtype=np.int32)


print("\n\nRun F5-TTS by ONNX Runtime.")
start_count = time.time()
noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, cat_mel_text, cat_mel_text_drop, ref_signal_len = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2, out_name_A3, out_name_A4, out_name_A5, out_name_A6, out_name_A7],
        {
            in_name_A0: audio,
            in_name_A1: text_ids,
            in_name_A2: max_duration
        })

print("NFE_STEP: 0")
for i in range(0, NFE_STEP - 1, FUSE_NFE):
    noise, time_step = ort_session_B.run(
        [out_name_B0, out_name_B1],
        {
            in_name_B0: noise,
            in_name_B1: rope_cos_q,
            in_name_B2: rope_sin_q,
            in_name_B3: rope_cos_k,
            in_name_B4: rope_sin_k,
            in_name_B5: cat_mel_text,
            in_name_B6: cat_mel_text_drop,
            in_name_B7: time_step
        })
    print(f"NFE_STEP: {i + FUSE_NFE}")

generated_signal = ort_session_C.run(
        [out_name_C0],
        {
            in_name_C0: noise,
            in_name_C1: ref_signal_len
        })[0]
end_count = time.time()

# Save to audio
sf.write(generated_audio, generated_signal.reshape(-1), SAMPLE_RATE, format='WAVEX')

print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{end_count - start_count:.3f}")
