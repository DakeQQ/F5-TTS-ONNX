import re
import site
import time
import jieba
import torch
import onnxruntime
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style
python_package_path = site.getsitepackages()[-1]

vocab_path           = "/home/DakeQQ/Downloads/F5TTS_v1_Base/vocab.txt"                             # The F5-TTS model vocab download path.     URL: https://huggingface.co/SWivid/F5-TTS/tree/main/F5TTS_v1_Base
onnx_model_A         = "/home/DakeQQ/Downloads/F5_Optimized/F5_Preprocess.onnx"                     # The exported onnx model path.
onnx_model_B         = "/home/DakeQQ/Downloads/F5_Optimized/F5_Transformer.onnx"                    # The exported onnx model path.
onnx_model_C         = "/home/DakeQQ/Downloads/F5_Optimized/F5_Decode.onnx"                         # The exported onnx model path.
generated_audio      = "./generated_audio.wav"
test_in_english = False

if test_in_english:
    reference_audio  = python_package_path + "/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text         = "Some call me nature, others call me mother nature."
    gen_text         = "Some call me Dake, others call me QQ."
else:
    reference_audio  = python_package_path + "/f5_tts/infer/examples/basic/basic_ref_zh.wav"        # The reference audio path.
    ref_text         = "对，这就是我，万人敬仰的太乙真人。"                                               # The ASR result of reference audio.
    gen_text         = "对，这就是我，万人敬仰的大可奇奇。"                                               # The target TTS.


ORT_Accelerate_Providers = ['CPUExecutionProvider']       # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                          # else keep empty.
RANDOM_SEED = 9527                                        # Set seed to reproduce the generated audio
NFE_STEP = 32                                             # F5-TTS model setting, 0~31
FUSE_NFE = 1                                              # Maintain the same values as the exported model.
SPEED = 1.0                                               # Set for talking speed. Only works with dynamic_axes=True
MAX_THREADS = 8                                           # Max CPU parallel threads.
DEVICE_ID = 0                                             # The GPU id, default to 0.
MODEL_SAMPLE_RATE = 24000                                 # Do not modify it.
HOP_LENGTH = 256                                          # It affects the generated audio length and speech speed.

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': False,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,     # 8 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',  # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',      # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                          # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '0',
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                    # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
     provider_options = [
         {
             'device_id': DEVICE_ID,
             'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
             'device_filter': 'any'                         # [any, npu, gpu]
         }
     ]
else:
    # Please config by yourself for others providers.
    provider_options = None


with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)


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


# ONNX Runtime settings
onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4                  # fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
model_type = ort_session_A._inputs_meta[0].type
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

if "CPUExecutionProvider" in ORT_Accelerate_Providers or not ORT_Accelerate_Providers:
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
else:
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
ORT_Accelerate_Providers = ort_session_B.get_providers()[0]
# For Windows DirectML + Intel/AMD/Nvidia GPU,
# pip install onnxruntime-directml --upgrade
# ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['DmlExecutionProvider'])
print(f"\nUsable Providers: {ORT_Accelerate_Providers}")
model_dtype = ort_session_B._inputs_meta[0].type
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

session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name

# Load the input audio
print(f"\nReference Audio: {reference_audio}")
audio = np.array(AudioSegment.from_file(reference_audio).set_channels(1).set_frame_rate(MODEL_SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)

zh_pause_punc = r"。，、；：？！"
ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
ref_audio_len = audio_len // HOP_LENGTH + 1
max_duration = np.array([ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED)], dtype=np.int64)
gen_text = convert_char_to_pinyin([ref_text + gen_text])
text_ids = list_str_to_idx(gen_text, vocab_char_map).numpy()
time_step = np.array([0], dtype=np.int32)

if "CPUExecutionProvider" in ORT_Accelerate_Providers or not ORT_Accelerate_Providers:
    device_type = 'cpu'
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers or "TensorrtExecutionProvider" in ORT_Accelerate_Providers:
    device_type = 'cuda'
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    device_type = 'dml'
else:
    device_type = None

print("\n\nRun F5-TTS by ONNX Runtime.")
start_count = time.time()
noise, rope_cos_q, rope_sin_q, rope_cos_k, rope_sin_k, cat_mel_text, cat_mel_text_drop, ref_signal_len = ort_session_A.run(
        [out_name_A0, out_name_A1, out_name_A2, out_name_A3, out_name_A4, out_name_A5, out_name_A6, out_name_A7],
        {
            in_name_A0: audio,
            in_name_A1: text_ids,
            in_name_A2: max_duration
        })

if device_type:
    inputs = [
        onnxruntime.OrtValue.ortvalue_from_numpy(noise, device_type, DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(rope_cos_q, device_type, DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(rope_sin_q, device_type, DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(rope_cos_k, device_type, DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(rope_sin_k, device_type, DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(cat_mel_text, device_type, DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(cat_mel_text_drop, device_type, DEVICE_ID),
        onnxruntime.OrtValue.ortvalue_from_numpy(time_step, device_type, DEVICE_ID)
    ]
    outputs = [
        inputs[0],
        inputs[-1]
    ]

    io_binding = ort_session_B.io_binding()
    for i in range(len(inputs)):
        io_binding.bind_ortvalue_input(
            name=in_name_B[i].name,
            ortvalue=inputs[i]
        )
    for i in range(len(outputs)):
        io_binding.bind_ortvalue_output(
            name=out_name_B[i].name,
            ortvalue=outputs[i]
        )

    print("NFE_STEP: 0")
    for i in range(0, NFE_STEP - 1, FUSE_NFE):
        ort_session_B.run_with_iobinding(io_binding)
        print(f"NFE_STEP: {i + FUSE_NFE}")
    noise = onnxruntime.OrtValue.numpy(io_binding.get_outputs()[0])
else:
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
sf.write(generated_audio, generated_signal.reshape(-1), MODEL_SAMPLE_RATE, format='WAVEX')
print(f"\nAudio generation is complete.\n\nONNXRuntime Time Cost in Seconds:\n{end_count - start_count:.3f}")
