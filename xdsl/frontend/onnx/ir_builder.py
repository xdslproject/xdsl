from onnx import NodeProto, ValueInfoProto

from xdsl.dialects import onnx
from xdsl.frontend.onnx.type import get_type
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import IRDLOperation

OP_BY_OP_TYPE: dict[str, type[IRDLOperation]] = {
    "Add": onnx.Add,
    "Sub": onnx.Sub,
}
"""Associate the name of the operations with the respective operation in ONNX dialect."""


class OnnxXdslMapping:
    """The representation of the onnx context."""

    type_by_name: dict[str, Attribute]
    value_by_name: dict[str, SSAValue]

    def __init__(self):
        self.type_by_name = {}
        self.value_by_name = {}


def visit_node(node: NodeProto, ctx: OnnxXdslMapping) -> IRDLOperation:
    """Update the onnx context with the current node of the onnx graph."""
    if node.op_type not in OP_BY_OP_TYPE:
        raise ValueError(f"Unknown ONNX op name {node.op_type}")

    op_class = OP_BY_OP_TYPE[node.op_type]

    operands = tuple(ctx.value_by_name[name] for name in node.input)
    result_types = tuple(ctx.type_by_name[name] for name in node.output)

    op = op_class.build(operands=operands, result_types=result_types)
    results = op.results

    for output_name, result in zip(node.output, results, strict=True):
        ctx.value_by_name[output_name] = result

    return op


def visit_value_info(i: ValueInfoProto, ctx: OnnxXdslMapping) -> Attribute:
    """Given the onnx ValueInforProto, it returns the corresponding Attribute stored in the context."""
    name = i.name

    if name in ctx.type_by_name:
        return ctx.type_by_name[name]

    t = get_type(i.type)
    ctx.type_by_name[name] = t
    return t
