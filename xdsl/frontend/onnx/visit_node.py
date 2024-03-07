from onnx import NodeProto

from xdsl.dialects import onnx
from xdsl.irdl import IRDLOperation

from .ctx import Ctx

OP_BY_OP_TYPE: dict[str, type[IRDLOperation]] = {
    "Add": onnx.Add,
    "Sub": onnx.Sub,
}


def visit_node(node: NodeProto, ctx: Ctx) -> None:
    if node.op_type not in OP_BY_OP_TYPE:
        raise ValueError(f"Unknown ONNX op name {node.op_type}")

    op = OP_BY_OP_TYPE[node.op_type]

    operands = tuple(ctx.value_by_name[name] for name in node.input)
    result_types = tuple(ctx.type_by_name[name] for name in node.output)

    results = op.build(operands=operands, result_types=result_types).results

    for output_name, result in zip(node.output, results, strict=True):
        ctx.value_by_name[output_name] = result
