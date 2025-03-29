import gc
import os
import argparse

import onnx.version_converter
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxslim import slim

parser = argparse.ArgumentParser(description='argument for each file')
parser.add_argument("--model", type=str, default="F5_Preprocess.onnx")
args = parser.parse_args()

print("model: ", args.model)

# Path Setting
original_folder_path = "/home/DakeQQ/Downloads/F5_ONNX"                                 # The fp32 saved folder.
optimized_folder_path = "/home/DakeQQ/Downloads/F5_Optimized"    
model_path = os.path.join(original_folder_path, args.model)                             # The original fp32 model name.
optimized_model_path = os.path.join(optimized_folder_path, args.model)                  # The optimized model name.

# model_path = os.path.join(original_folder_path, "F5_Preprocess.onnx")                 # The original fp32 model name.
# optimized_model_path = os.path.join(optimized_folder_path, "F5_Preprocess.onnx")      # The optimized model name.

# model_path = os.path.join(original_folder_path, "F5_Transformer.onnx")                # The original fp32 model name.
# optimized_model_path = os.path.join(optimized_folder_path, "F5_Transformer.onnx")     # The optimized model name.

# model_path = os.path.join(original_folder_path, "F5_Decode.onnx")                     # The original fp32 model name.
# optimized_model_path = os.path.join(optimized_folder_path, "F5_Decode.onnx")          # The optimized model name.

use_gpu_fp16 = True                                                                     # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                       # ['CPUExecutionProvider', 'CUDAExecutionProvider']
target_platform = "amd64"                                                               # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.


# First we set our optimisation to the ORT Optimizer defaults for the provided type
optimization_options = FusionOptions("bert")
# The ORT optimizer is designed for ORT GPU and CUDA
# To make things work with ORT DirectML, we disable some options
# The GroupNorm op has a very negative effect on VRAM and CPU use
optimization_options.enable_group_norm = False
# On by default in ORT optimizer, turned off as it causes performance issues
optimization_options.enable_nhwc_conv = False
# On by default in ORT optimizer, turned off because it has no effect
optimization_options.enable_qordered_matmul = False
optimizer = optimize_model(
    input = model_path,
    model_type = "bert",
    opt_level=0,
    num_heads=16,
    hidden_size=1024,
    provider=provider,
    optimization_options = optimization_options,
    only_onnxruntime = False,
    use_gpu=True
)

optimizer.topological_sort()

    
# collate external tensor files into one
onnx.save_model(
    optimizer.model,
    optimized_model_path,
    save_as_external_data=False,
    all_tensors_to_one_file=True,
    convert_attribute=False,
)    

del optimizer
gc.collect()
 

slim(
    model=optimized_model_path,
    output_model=optimized_model_path,
    no_shape_infer=True if 'F5_Preprocess' in model_path else False,                    # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False,
)

print("Model optimized for DML !")
