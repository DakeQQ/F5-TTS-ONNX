from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, helper


_STANDARD_DOMAIN = {"", "ai.onnx"}
_FLOAT_TYPES = {
    TensorProto.FLOAT16,
    TensorProto.FLOAT,
    TensorProto.DOUBLE,
    TensorProto.BFLOAT16,
}


def _attribute(node: onnx.NodeProto, name: str, default: Any = None) -> Any:
    for attribute in node.attribute:
        if attribute.name == name:
            return helper.get_attribute_value(attribute)
    return default


def _is_position_conv(node: onnx.NodeProto, initializers: dict[str, onnx.TensorProto]) -> bool:
    if (
        node.op_type != "Conv"
        or node.domain not in _STANDARD_DOMAIN
        or len(node.input) not in (2, 3)
        or len(node.output) != 1
    ):
        return False
    weight = initializers.get(node.input[1])
    if weight is None or len(weight.dims) != 3 or weight.data_type not in _FLOAT_TYPES:
        return False
    if len(node.input) == 3:
        bias = initializers.get(node.input[2])
        if (
            bias is None
            or len(bias.dims) != 1
            or bias.dims[0] != weight.dims[0]
            or bias.data_type != weight.data_type
        ):
            return False
    return (
        weight.dims[0] > 0
        and weight.dims[1] > 0
        and weight.dims[-1] == 31
        and _attribute(node, "auto_pad", b"NOTSET") == b"NOTSET"
        and _attribute(node, "group", 1) == 16
        and list(_attribute(node, "kernel_shape", [])) == [31]
        and list(_attribute(node, "pads", [])) == [15, 15]
        and list(_attribute(node, "strides", [1])) == [1]
        and list(_attribute(node, "dilations", [1])) == [1]
    )


def _unique_node_name(existing: set[str], base: str) -> str:
    candidate = base
    suffix = 1
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    existing.add(candidate)
    return candidate


def rewrite_mish_subgraphs(
    raw_model_path: str | Path,
    final_model_path: str | Path,
    *,
    expected_matches: int = 2,
) -> dict[str, Any]:
    """Replace exactly two exported Mish decompositions with standard ONNX Mish nodes."""
    raw_path = Path(raw_model_path).resolve()
    final_path = Path(final_model_path).resolve()
    if raw_path == final_path:
        raise ValueError("Raw and final model paths must be different.")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw ONNX model was not found: {raw_path}")
    if raw_path.stat().st_size >= 2_000_000_000:
        raise ValueError("This rewrite requires an inline ONNX protobuf smaller than 2 GB.")

    model = onnx.load(raw_path, load_external_data=False)
    if any(initializer.data_location == TensorProto.EXTERNAL or initializer.external_data for initializer in model.graph.initializer):
        raise ValueError("External-data models are not supported by this narrowly scoped rewrite.")
    onnx.checker.check_model(model)

    interface_before = [
        (value.name, value.type.SerializeToString())
        for value in list(model.graph.input) + list(model.graph.output)
    ]
    metadata_before = [(item.key, item.value) for item in model.metadata_props]
    initializer_names_before = [initializer.name for initializer in model.graph.initializer]
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    nodes = list(model.graph.node)
    producers = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in nodes:
        for input_name in node.input:
            if input_name:
                consumers[input_name].append(node)
    node_indices = {id(node): index for index, node in enumerate(nodes)}

    matches: list[tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto, str]] = []
    for softplus in nodes:
        if (
            softplus.op_type != "Softplus"
            or softplus.domain not in _STANDARD_DOMAIN
            or softplus.attribute
            or len(softplus.input) != 1
            or len(softplus.output) != 1
        ):
            continue
        source = softplus.input[0]
        source_producer = producers.get(source)
        if source_producer is None or not _is_position_conv(source_producer, initializers):
            continue
        softplus_consumers = consumers.get(softplus.output[0], [])
        if len(softplus_consumers) != 1:
            continue
        tanh = softplus_consumers[0]
        if (
            tanh.op_type != "Tanh"
            or tanh.domain not in _STANDARD_DOMAIN
            or tanh.attribute
            or list(tanh.input) != [softplus.output[0]]
            or len(tanh.output) != 1
        ):
            continue
        tanh_consumers = consumers.get(tanh.output[0], [])
        if len(tanh_consumers) != 1:
            continue
        multiply = tanh_consumers[0]
        if (
            multiply.op_type != "Mul"
            or multiply.domain not in _STANDARD_DOMAIN
            or multiply.attribute
            or len(multiply.input) != 2
            or len(multiply.output) != 1
            or sorted(multiply.input) != sorted((source, tanh.output[0]))
        ):
            continue
        matches.append((softplus, tanh, multiply, source))

    existing_mish = sum(
        node.op_type == "Mish" and node.domain in _STANDARD_DOMAIN
        for node in nodes
    )
    if not matches and existing_mish == expected_matches:
        raise ValueError("The input model already contains the expected Mish rewrite.")
    if len(matches) != expected_matches:
        raise ValueError(
            f"Expected exactly {expected_matches} Conv1d Mish decompositions, found {len(matches)}."
        )

    matched_indices: set[int] = set()
    replacement_by_index: dict[int, onnx.NodeProto] = {}
    dead_value_info: set[str] = set()
    existing_names = {node.name for node in nodes if node.name}
    for match_index, (softplus, tanh, multiply, source) in enumerate(matches):
        indices = {node_indices[id(softplus)], node_indices[id(tanh)], node_indices[id(multiply)]}
        if matched_indices.intersection(indices):
            raise ValueError("Matched Mish decompositions overlap.")
        matched_indices.update(indices)
        replacement_by_index[node_indices[id(softplus)]] = helper.make_node(
            "Mish",
            inputs=[source],
            outputs=[multiply.output[0]],
            name=_unique_node_name(existing_names, f"F5_ConvPosition_Mish_{match_index}"),
        )
        dead_value_info.update((softplus.output[0], tanh.output[0]))

    retained_serialized = [
        node.SerializeToString()
        for index, node in enumerate(nodes)
        if index not in matched_indices
    ]
    rewritten_nodes: list[onnx.NodeProto] = []
    for index, node in enumerate(nodes):
        replacement = replacement_by_index.get(index)
        if replacement is not None:
            rewritten_nodes.append(replacement)
        if index not in matched_indices:
            rewritten_nodes.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    retained_value_info = [value for value in model.graph.value_info if value.name not in dead_value_info]
    removed_value_info = len(model.graph.value_info) - len(retained_value_info)
    if removed_value_info:
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_info)

    interface_after = [
        (value.name, value.type.SerializeToString())
        for value in list(model.graph.input) + list(model.graph.output)
    ]
    if interface_after != interface_before:
        raise RuntimeError("Graph input/output interface changed during Mish rewrite.")
    if [(item.key, item.value) for item in model.metadata_props] != metadata_before:
        raise RuntimeError("Model metadata changed during Mish rewrite.")
    if [initializer.name for initializer in model.graph.initializer] != initializer_names_before:
        raise RuntimeError("Initializers changed during Mish rewrite.")
    retained_after = [
        node.SerializeToString()
        for node in model.graph.node
        if node.op_type != "Mish" or node.domain not in _STANDARD_DOMAIN
    ]
    if retained_after != retained_serialized:
        raise RuntimeError("An unrelated ONNX node changed during Mish rewrite.")
    if sum(node.op_type == "Mish" and node.domain in _STANDARD_DOMAIN for node in model.graph.node) != expected_matches:
        raise RuntimeError("The rewritten graph does not contain the expected Mish nodes.")
    onnx.checker.check_model(model)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{final_path.stem}.",
        suffix=final_path.suffix,
        dir=final_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        onnx.save(model, temporary_path)
        onnx.checker.check_model(temporary_path)
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    report = {
        "raw_model": str(raw_path),
        "final_model": str(final_path),
        "matched_subgraphs": len(matches),
        "inserted_nodes": expected_matches,
        "deleted_nodes": expected_matches * 3,
        "net_node_reduction": expected_matches * 2,
        "rewired_edges": 0,
        "transformed_initializers": 0,
        "deleted_initializers": 0,
        "removed_value_info": removed_value_info,
        "operator": "ai.onnx::Mish-18",
        "custom_domains_added": 0,
        "opset_changes": 0,
    }
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite F5 ConvPosition Mish decompositions.")
    parser.add_argument("raw_model", type=Path)
    parser.add_argument("final_model", type=Path)
    parser.add_argument("--expected-matches", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    result = rewrite_mish_subgraphs(
        arguments.raw_model,
        arguments.final_model,
        expected_matches=arguments.expected_matches,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
