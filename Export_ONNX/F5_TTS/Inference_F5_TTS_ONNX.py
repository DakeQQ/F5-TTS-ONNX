import argparse
import logging
import site
import time
from pathlib import Path
try:
    import rjieba
except ImportError:
    rjieba = None
    import jieba
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style
python_package_path = site.getsitepackages()[-1]
SCRIPT_DIR = Path(__file__).resolve().parent


# ============================== Configuration ==============================
VOCAB_PATH = Path.home() / "Downloads" / "F5TTS_v1_Base" / "vocab.txt"
GENERATED_AUDIO_PATH = SCRIPT_DIR / "generated_audio.wav"
SPEED = 1.0                    # Talking speed; requires a dynamic-axis export.
DEVICE_ID = 0                  # Accelerator device index.
ORT_FP16 = False               # Enable FP16 runtime settings where supported.
SHOW_PROGRESS = True           # Print pipeline stages and generation progress.
# ===========================================================================


def print_progress(message):
    if SHOW_PROGRESS:
        print(f"[F5-TTS] {message}", flush=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        prog="F5-TTS-ONNX Inference",
        description="Infer F5-TTS via ONNX Runtime.",
        epilog="Authors: DakeQQ and contributors",
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=logging.getLevelNamesMapping().keys(),
        help="Set the logging level. Default: INFO.",
    )
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "F5_Optimized",
        help="Folder containing ONNX graphs. Default: F5_Optimized.",
    )
    parser.add_argument(
        "--vocab-path",
        "--vocab_path",
        type=Path,
        default=VOCAB_PATH,
        help="Path to the F5-TTS vocab.txt file.",
    )
    parser.add_argument("--metadata_path", "--metadata-path", type=Path, help="Path to F5_Metadata.onnx.")
    parser.add_argument("--preprocessmodel_path", "--preprocess-model-path", type=Path, help="Path to F5_Preprocess.onnx.")
    parser.add_argument("--transformermodel_path", "--transformer-model-path", type=Path, help="Path to F5_Transformer.onnx.")
    parser.add_argument("--decodermodel_path", "--decoder-model-path", type=Path, help="Path to F5_Decode.onnx.")
    parser.add_argument("--force_nfe", "--force-nfe", type=int, default=-1, help="Override metadata NFE steps with a positive value no greater than the exported count.")
    parser.add_argument(
        "--prepro_input_type",
        "--prepro-input-type",
        choices=("f16", "f32", "i16"),
        default="i16",
        help="Compatibility option; the actual dtype is read from F5_Preprocess.onnx.",
    )
    parser.add_argument("--ort_provider", "--ort-provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider.")
    parser.add_argument("--numthreads", "--num-threads", type=int, default=8, help="ONNX Runtime inter/intra-op thread count; 0 selects automatic sizing.")
    parser.add_argument("--profile", action="store_true", help="Enable ONNX Runtime JSON profiling.")
    parser.add_argument("--testlang", default="zh", help="Bundled test language: zh, en, or ar. Other values use zh.")
    parser.add_argument("--seed", type=int, default=9527, help="ONNX Runtime random seed.")
    parser.add_argument("--output_path", "--output-path", type=Path, default=GENERATED_AUDIO_PATH, help="Generated WAV path.")
    return parser.parse_args()


_ARGS = _parse_args()
logging.basicConfig(level=_ARGS.loglevel.upper())
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()
vocab_path = _ARGS.vocab_path.expanduser().resolve()

metadata_path = (_ARGS.metadata_path or onnx_folder / "F5_Metadata.onnx").expanduser().resolve()
onnx_model_Metadata = str(metadata_path)
generated_audio = str(_ARGS.output_path.expanduser().resolve())

language = _ARGS.testlang.casefold()
if language == "en":
    reference_audio = str(Path(python_package_path) / "f5_tts/infer/examples/basic/basic_ref_en.wav")
    ref_text = "Some call me nature, others call me mother nature."
    gen_text = "Some call me Dake, others call me QQ."
elif language == "ar":
    reference_audio = str(Path(python_package_path) / "silma_tts/infer/ref_audio_samples/ar.ref.24k.wav")
    ref_text = "ويدقق النظر في القرآن الكريم وسائر الكتب السماوية ويتبع مسالك الرسل العظام عليهم الصلاة والسلام."
    gen_text = "أنا نموذج جديد من سلمى لتحويل النص إلى كلام، يمكنني التحدث باللغة العربية مع أو بدون علامات التشكيل."
else:
    reference_audio = str(Path(python_package_path) / "f5_tts/infer/examples/basic/basic_ref_zh.wav")
    ref_text = "对，这就是我，万人敬仰的太乙真人。"
    gen_text = "对，这就是我，万人敬仰的大可奇奇。"

ORT_Accelerate_Providers = [_ARGS.ort_provider]
RANDOM_SEED = _ARGS.seed
MAX_THREADS = _ARGS.numthreads
ORT_LOG = _ARGS.loglevel.upper() == "DEBUG"

def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )

    def is_chinese(c):
        return "\u3100" <= c <= "\u9fff"

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        if rjieba is None:
            if jieba.dt.initialized is False:
                jieba.default_logger.setLevel(50)  # CRITICAL
                jieba.initialize()
            segments = jieba.cut(text)
        else:
            segments = rjieba.cut(text)
        for seg in segments:
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):
                seg_ = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:
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


def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],
    dtype,
    padding_value=-1
):
    max_length = max(map(len, text), default=0)
    text_ids = np.full((len(text), max_length), padding_value, dtype=dtype)
    get_idx = vocab_char_map.get
    for row_index, characters in enumerate(text):
        text_ids[row_index, :len(characters)] = [get_idx(character, 0) for character in characters]
    return text_ids


ORT_TYPE_TO_DTYPE = {
    "tensor(bool)": np.bool_,
    "tensor(double)": np.float64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
}


def numpy_dtype(argument):
    try:
        return ORT_TYPE_TO_DTYPE[argument.type]
    except KeyError as exc:
        raise TypeError(f"Unsupported ONNX tensor type: {argument.type}") from exc


def resolve_shape(argument, symbols=None, dynamic_sizes=()):
    symbols = {} if symbols is None else symbols
    sizes = iter(dynamic_sizes)
    shape = []
    for dimension in argument.shape:
        if isinstance(dimension, int):
            size = dimension
        elif dimension is not None and dimension in symbols:
            size = symbols[dimension]
        else:
            try:
                size = next(sizes)
            except StopIteration as exc:
                raise ValueError(f"Cannot resolve shape for {argument.name}: {argument.shape}") from exc
            if dimension is not None:
                symbols[dimension] = size
        shape.append(int(size))
    return tuple(shape)


def record_shape(argument, shape, symbols):
    for dimension, size in zip(argument.shape, shape):
        if dimension is not None and not isinstance(dimension, int):
            symbols[dimension] = int(size)


def resolve_from_shape(argument, actual_shape, symbols):
    dynamic_sizes = (
        actual_shape[index]
        for index, dimension in enumerate(argument.shape)
        if not isinstance(dimension, int)
    )
    shape = resolve_shape(argument, symbols, dynamic_sizes)
    record_shape(argument, shape, symbols)
    return shape


def numpy_for(argument, data, shape):
    return np.ascontiguousarray(np.asarray(data, dtype=numpy_dtype(argument)).reshape(shape))


def ortvalue_for(argument, data, shape, device):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        numpy_for(argument, data, shape), device, DEVICE_ID
    )


def empty_ortvalue(argument, shape, device):
    return onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        shape, numpy_dtype(argument), device, DEVICE_ID
    )


def same_layout(left, right):
    if left.type != right.type or len(left.shape) != len(right.shape):
        return False
    return all(
        not isinstance(left_dim, int)
        or not isinstance(right_dim, int)
        or left_dim == right_dim
        for left_dim, right_dim in zip(left.shape, right.shape)
    )


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def bind_inputs(binding, arguments, values):
    for argument in arguments:
        binding.bind_ortvalue_input(argument.name, values[argument.name])


def bind_outputs(binding, arguments, values):
    for argument in arguments:
        binding.bind_ortvalue_output(argument.name, values[argument.name])


onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.enable_profiling         = _ARGS.profile
session_opts.enable_cpu_mem_arena     = True
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False
    }]
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers or "TensorrtExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      8 * (1024 ** 3),    # 8 GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["kNextPowerOfTwo", "kSameAsRequested"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '1',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '1',
        'enable_cuda_graph':                  '0',                # Set to '0' to avoid potential errors when enabled.
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':              DEVICE_ID,
        'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
        'device_filter':          'gpu'                # [any, npu, gpu]
    }]
    device_type = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_type = C.OrtDevice(
    _ort_device_type,
    C.OrtDevice.default_memory(),
    DEVICE_ID,
)
_ort_host_device_type = (
    _ort_device_type
    if device_type == 'cpu'
    else C.OrtDevice(
        C.OrtDevice.cpu(),
        C.OrtDevice.default_memory(),
        DEVICE_ID,
    )
)

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}
packed_settings_cpu = {
    "_session_opts":        session_opts,
    "_providers":           ['CPUExecutionProvider'],
    "_provider_options":    None,
    "_disabled_optimizers": disabled_optimizers
}


pipeline_started = time.perf_counter()
print_progress(f"Reading package metadata: {onnx_model_Metadata}")
ort_session_Metadata = create_session(onnx_model_Metadata, **packed_settings_cpu)
model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map
expected_metadata_keys = {
    "sample_rate",
    "in_sample_rate",
    "out_sample_rate",
    "hop_length",
    "nfe_step",
    "max_signal_length",
    "model_file_name_preprocess",
    "model_file_name_transformer",
    "model_file_name_decode",
}
optional_metadata_keys = {"model_series"}
if not expected_metadata_keys <= set(model_meta) or set(model_meta) - expected_metadata_keys - optional_metadata_keys:
    raise ValueError(
        "F5-TTS metadata keys do not match the runtime contract: "
        f"missing={sorted(expected_metadata_keys - set(model_meta))}, "
        f"extra={sorted(set(model_meta) - expected_metadata_keys - optional_metadata_keys)}."
    )

INV_INT16 = float(1.0 / 32768.0)
MODEL_SERIES      = model_meta.get("model_series", "unknown")
MODEL_SAMPLE_RATE = int(model_meta["sample_rate"])
IN_SAMPLE_RATE    = int(model_meta["in_sample_rate"])
OUT_SAMPLE_RATE   = int(model_meta["out_sample_rate"])
HOP_LENGTH        = int(model_meta["hop_length"])
EXPORTED_NFE_STEP = int(model_meta["nfe_step"])
if _ARGS.force_nfe > EXPORTED_NFE_STEP:
    raise ValueError(
        f"force_nfe={_ARGS.force_nfe} exceeds the exported timestep table ({EXPORTED_NFE_STEP})."
    )
NFE_STEP = _ARGS.force_nfe if _ARGS.force_nfe > 0 else EXPORTED_NFE_STEP
MAX_SIGNAL_LENGTH = int(model_meta["max_signal_length"])
onnx_model_Preprocess = str(
    (_ARGS.preprocessmodel_path or metadata_path.parent / model_meta["model_file_name_preprocess"])
    .expanduser().resolve()
)
onnx_model_Transformer = str(
    (_ARGS.transformermodel_path or metadata_path.parent / model_meta["model_file_name_transformer"])
    .expanduser().resolve()
)
onnx_model_Decode = str(
    (_ARGS.decodermodel_path or metadata_path.parent / model_meta["model_file_name_decode"])
    .expanduser().resolve()
)
del ort_session_Metadata

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
print_progress(
    f"Metadata ready: series={MODEL_SERIES}, input={IN_SAMPLE_RATE} Hz,"
    f"output={OUT_SAMPLE_RATE} Hz, "
    f"hop_length={HOP_LENGTH}, nfe_steps={NFE_STEP}."
)
print_progress(f"Vocabulary ready: {len(vocab_char_map)} tokens.")

print_progress("Loading ONNX graph 1/3: F5_Preprocess.onnx")
ort_session_Preprocess = create_session(onnx_model_Preprocess, **packed_settings)
print_progress("Loading ONNX graph 2/3: F5_Transformer.onnx")
ort_session_Transformer = create_session(onnx_model_Transformer, **packed_settings)
print_progress(f"ONNX Runtime providers: {ort_session_Transformer.get_providers()}")
print_progress("Loading ONNX graph 3/3: F5_Decode.onnx")
ort_session_Decode = create_session(onnx_model_Decode, **packed_settings)

preprocess_inputs = tuple(ort_session_Preprocess.get_inputs())
preprocess_outputs = tuple(ort_session_Preprocess.get_outputs())
transformer_inputs = tuple(ort_session_Transformer.get_inputs())
transformer_outputs = tuple(ort_session_Transformer.get_outputs())
decode_inputs = tuple(ort_session_Decode.get_inputs())
decode_outputs = tuple(ort_session_Decode.get_outputs())
audio_input, text_input, duration_input = preprocess_inputs
transformer_output, = transformer_outputs
decode_output, = decode_outputs
downstream_inputs = {
    argument.name: argument
    for argument in (*transformer_inputs, *decode_inputs)
}

print_progress(f"Preparing reference audio: {reference_audio}")
audio = np.asarray(
    AudioSegment.from_file(reference_audio)
    .set_channels(1)
    .set_frame_rate(IN_SAMPLE_RATE)
    .set_sample_width(2)
    .get_array_of_samples()
)
if np.issubdtype(numpy_dtype(audio_input), np.floating):
    audio = (audio.astype(np.float32) * INV_INT16).astype(numpy_dtype(audio_input))
audio_len = audio.size

if len(ref_text[-1].encode("utf-8")) == 1:
    ref_text = ref_text + " "
local_speed = SPEED
if len(gen_text.encode("utf-8")) < 10:
    local_speed = 0.3
ref_text_len = len(ref_text.encode('utf-8'))
gen_text_len = len(gen_text.encode('utf-8'))
model_audio_len = int(audio_len * MODEL_SAMPLE_RATE / IN_SAMPLE_RATE)
ref_audio_len = model_audio_len // HOP_LENGTH
cond_signal_len = ref_audio_len + 1
text = convert_char_to_pinyin([ref_text + gen_text])
text_ids = list_str_to_idx(text, vocab_char_map, numpy_dtype(text_input))
duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / local_speed)
max_duration_value = max(max(text_ids.shape[-1], cond_signal_len) + 1, duration)
if max_duration_value > MAX_SIGNAL_LENGTH:
    raise ValueError(
        f"Requested max_duration {max_duration_value} exceeds exported max_signal_length {MAX_SIGNAL_LENGTH}. "
        "Use shorter text/audio or re-export with a larger MAX_SIGNAL_LENGTH."
    )
symbols = {"max_duration": max_duration_value}
audio_shape = resolve_shape(audio_input, symbols, (audio_len,))
record_shape(audio_input, audio_shape, symbols)
text_shape = resolve_from_shape(text_input, text_ids.shape, symbols)
duration_shape = resolve_shape(duration_input, symbols)
record_shape(duration_input, duration_shape, symbols)

preprocess_input_values = {
    audio_input.name: ortvalue_for(audio_input, audio, audio_shape, 'cpu'),
    text_input.name: ortvalue_for(text_input, text_ids, text_shape, 'cpu'),
    duration_input.name: ortvalue_for(
        duration_input, max_duration_value, duration_shape, 'cpu'
    ),
}
preprocess_output_values = {}
for argument in preprocess_outputs:
    downstream_input = downstream_inputs.get(argument.name)
    if downstream_input is None or not same_layout(argument, downstream_input):
        raise ValueError(f"No compatible downstream input for preprocess output {argument.name!r}.")
    shape = resolve_shape(downstream_input, symbols)
    record_shape(argument, shape, symbols)
    preprocess_output_values[argument.name] = empty_ortvalue(argument, shape, 'cpu')

step_input, = (
    argument for argument in transformer_inputs
    if argument.name not in preprocess_output_values
)
state_input, = (
    argument for argument in transformer_inputs
    if argument.name in preprocess_output_values and same_layout(argument, transformer_output)
)
constant_inputs = tuple(
    argument for argument in transformer_inputs
    if argument.name not in (state_input.name, step_input.name)
)

state_source = preprocess_output_values[state_input.name]
state_shape = resolve_from_shape(state_input, state_source.shape(), symbols)
output_shape = resolve_from_shape(transformer_output, state_shape, symbols)
device_transfers = []
if device_type == 'cpu':
    state_buffer = state_source
    constant_values = {
        argument.name: preprocess_output_values[argument.name]
        for argument in constant_inputs
    }
else:
    state_buffer = empty_ortvalue(state_input, state_shape, device_type)
    device_transfers.append((state_buffer, state_source))
    constant_values = {}
    for argument in constant_inputs:
        source = preprocess_output_values[argument.name]
        shape = resolve_from_shape(argument, source.shape(), symbols)
        value = empty_ortvalue(argument, shape, device_type)
        constant_values[argument.name] = value
        device_transfers.append((value, source))

state_buffers = (
    state_buffer,
    empty_ortvalue(transformer_output, output_shape, device_type),
)
step_shape = resolve_shape(step_input, symbols)
step_dtype = numpy_dtype(step_input)
step_values = np.broadcast_to(
    np.arange(NFE_STEP, dtype=step_dtype).reshape(
        (NFE_STEP,) + (1,) * len(step_shape)
    ),
    (NFE_STEP,) + step_shape,
).copy()
# Rebind immutable values; mutating a CPU OrtValue already bound to CUDA can use stale staged data.
step_device = device_type if device_type == 'cuda' else 'cpu'
step_buffers = tuple(
    ortvalue_for(step_input, step_values[step], step_shape, step_device)
    for step in range(NFE_STEP)
)
if len({value.data_ptr() for value in state_buffers}) != len(state_buffers):
    raise RuntimeError("Transformer state buffers must have unique addresses.")
if len({value.data_ptr() for value in step_buffers}) != len(step_buffers):
    raise RuntimeError("Transformer step buffers must have unique addresses.")

transformer_bindings = tuple(
    ort_session_Transformer.io_binding()
    for _ in range(2)
)
for index, binding in enumerate(transformer_bindings):
    input_values = dict(constant_values)
    input_values[state_input.name] = state_buffers[index]
    bind_inputs(binding, constant_inputs + (state_input,), input_values)
    bind_outputs(
        binding,
        transformer_outputs,
        {transformer_output.name: state_buffers[1 - index]},
    )

binding_Preprocess = ort_session_Preprocess.io_binding()
bind_inputs(binding_Preprocess, preprocess_inputs, preprocess_input_values)
bind_outputs(binding_Preprocess, preprocess_outputs, preprocess_output_values)

noise_final = state_buffers[NFE_STEP & 1]
decode_state_input, = (
    argument for argument in decode_inputs
    if argument.name not in preprocess_output_values
)
decode_input_values = {
    argument.name: preprocess_output_values[argument.name]
    for argument in decode_inputs
    if argument.name in preprocess_output_values
}
if device_type == 'cpu':
    decode_input_values[decode_state_input.name] = noise_final
else:
    decode_state_shape = resolve_from_shape(
        decode_state_input, noise_final.shape(), symbols
    )
    decode_state_buffer = empty_ortvalue(
        decode_state_input, decode_state_shape, 'cpu'
    )
    decode_input_values[decode_state_input.name] = decode_state_buffer

inference_started = time.perf_counter()
print_progress("Running acoustic preprocessing...")
run(ort_session_Preprocess, binding_Preprocess)
for destination, source in device_transfers:
    destination.update_inplace(source)
print_progress(f"Running flow matching ({NFE_STEP} steps)...")
progress_interval = max(1, NFE_STEP // 5)
for step in range(NFE_STEP):
    binding = transformer_bindings[step & 1]
    binding.bind_ortvalue_input(step_input.name, step_buffers[step])
    run(ort_session_Transformer, binding)
    completed_steps = step + 1
    if completed_steps % progress_interval == 0 or completed_steps == NFE_STEP:
        print_progress(
            f"Flow matching: {completed_steps}/{NFE_STEP} steps "
            f"({time.perf_counter() - inference_started:.2f}s elapsed)"
        )
if device_type != 'cpu':
    decode_state_buffer.update_inplace(noise_final)
binding_Decode = ort_session_Decode.io_binding()
bind_inputs(binding_Decode, decode_inputs, decode_input_values)
try:
    decode_output_shape = resolve_shape(decode_output, symbols)
except ValueError:
    generated_signal = None
    binding_Decode._iobinding.bind_output(
        decode_output.name,
        _ort_host_device_type,
    )
else:
    generated_signal = empty_ortvalue(decode_output, decode_output_shape, 'cpu')
    binding_Decode.bind_ortvalue_output(decode_output.name, generated_signal)
print_progress("Decoding waveform...")
run(ort_session_Decode, binding_Decode)
if generated_signal is None:
    generated_signal, = binding_Decode.get_outputs()
inference_elapsed = time.perf_counter() - inference_started

generated_signal = generated_signal.numpy()
audio_duration = generated_signal.size / OUT_SAMPLE_RATE
rtf = inference_elapsed / max(audio_duration, 1.0e-9)
print_progress(f"Writing generated audio: {generated_audio}")
Path(generated_audio).parent.mkdir(parents=True, exist_ok=True)
if generated_signal.dtype == np.float16:
    generated_signal = generated_signal.astype(np.float32)
output_subtype = "PCM_16" if np.issubdtype(generated_signal.dtype, np.integer) else "FLOAT"
sf.write(
    generated_audio,
    generated_signal.reshape(-1),
    OUT_SAMPLE_RATE,
    subtype=output_subtype,
    format='WAVEX',
)
print_progress(
    f"Pipeline complete in {time.perf_counter() - pipeline_started:.2f}s "
    f"(inference {inference_elapsed:.2f}s, audio {audio_duration:.2f}s, RTF={rtf:.3f})."
)
