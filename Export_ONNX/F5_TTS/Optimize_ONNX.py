"""Optimize and quantize every exported F5-TTS ONNX graph.

Edit ``MODEL_PLANS`` and ``CONFIG`` below to choose each graph's precision,
quantization, optimization, and storage policy. The processing pipeline remains
centralized in ``Optimize_ONNX_Common.py``.

    Method       Backend                   Result
    "Q2/Q4/Q8"   matmul_nbits_quantizer    2/4/8-bit weight-only (MatMulNBits)
    "DYNAMIC"    quantize_dynamic          INT8 dynamic (DynamicQuantizeLinear)
    "F16"        convert_float_to_float16  float16 weights & activations
    "F32"        -                         keep float32 (optimize only)

For the DirectML-tuned optimization pass, use ``Optimize_ONNX_DML.py`` instead.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Optimize_ONNX_Common import (  # noqa: E402
    OptimizerConfig,
    Plan,
    resolve_plan,
    run_optimizer,
    validate_plan,
)


SOURCE_FOLDER = SCRIPT_DIR / "F5_ONNX"
OUTPUT_FOLDER = SCRIPT_DIR / "F5_Optimized"

# The DiT transformer uses 16 heads / hidden_size 1024; the same values feed the attention-fusion
# optimizer for every module (harmless for the non-attention Preprocess / Decode graphs).
# F5_Preprocess carries dynamic STFT shapes, so its onnxslim passes skip shape inference.
MODEL_PLANS: dict[str, Plan] = {
    "F5_Metadata": Plan(
        method="F32",
        transformer=False,
    ),
    "F5_Preprocess": Plan(
        method="F32",
        num_heads=0,
        hidden_size=0,
        opt_level=2,
    ),
    "F5_Transformer": Plan(
        method="F32",
        num_heads=16,
        hidden_size=1024,
        transformer=True,
        opt_level=2,
    ),
    "F5_Decode": Plan(
        method="F32",
        num_heads=0,
        hidden_size=0,
        opt_level=2,
    ),
}

CONFIG = OptimizerConfig(
    original_folder_path=str(SOURCE_FOLDER),
    optimized_folder_path=str(OUTPUT_FOLDER),
    model_plans=MODEL_PLANS,
    optimizer_level=2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Optimize one graph by name, with or without the .onnx suffix.")
    parser.add_argument("--source-folder", type=Path, default=SOURCE_FOLDER)
    parser.add_argument("--output-folder", type=Path, default=OUTPUT_FOLDER)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def select_plans(model: str | None) -> dict[str, Plan]:
    if model is None:
        return MODEL_PLANS
    model_name = Path(model).stem
    if model_name not in MODEL_PLANS:
        raise ValueError(f"Unknown model {model!r}; expected one of {tuple(MODEL_PLANS)}.")
    return {model_name: MODEL_PLANS[model_name]}


def resolve_plans(config: OptimizerConfig):
    resolved_plans = {}
    for name, plan in config.model_plans.items():
        resolved = resolve_plan(plan, config)
        validate_plan(name, resolved)
        resolved_plans[name] = resolved
    return resolved_plans


def validate_sources(config: OptimizerConfig) -> None:
    source_folder = Path(config.original_folder_path)
    missing = [
        source_folder / f"{name}.onnx"
        for name in config.model_plans
        if not (source_folder / f"{name}.onnx").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing F5-TTS graph(s): {missing}")


def main() -> None:
    args = parse_args()
    config = replace(
        CONFIG,
        original_folder_path=str(args.source_folder.expanduser().resolve()),
        optimized_folder_path=str(args.output_folder.expanduser().resolve()),
        model_plans=select_plans(args.model),
    )
    resolved_plans = resolve_plans(config)
    if args.check_only:
        quantized_count = sum(
            plan.method in {"Q2", "Q4", "Q8", "DYNAMIC"}
            for plan in resolved_plans.values()
        )
        print(
            f"F5-TTS optimizer plan is valid: {quantized_count} quantized graphs, "
            f"{len(resolved_plans)} graph(s) total."
        )
        return
    validate_sources(config)
    run_optimizer(config)


if __name__ == "__main__":
    main()
