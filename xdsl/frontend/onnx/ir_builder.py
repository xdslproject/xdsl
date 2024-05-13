from onnx import GraphProto, NodeProto, ValueInfoProto

from xdsl.dialects import func, onnx
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.onnx.type import get_type
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import IRDLOperation

OP_BY_OP_TYPE: dict[str, type[IRDLOperation]] = {
    "Add": onnx.Add,
    "Sub": onnx.Sub,
    "MatMul": onnx.MatMul,
    "Transpose": onnx.Transpose,
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


def build_module(graph: GraphProto) -> ModuleOp:
    """Create the ModuleOp based on the onnx graph provided."""

    ctx = OnnxXdslMapping()
    fn = visit_graph(graph, ctx)
    module = ModuleOp([fn])

    return module


def visit_graph(g: GraphProto, ctx: OnnxXdslMapping) -> IRDLOperation:
    """
    Visit the onnx graph to update the onnx context.
    """

    name = g.name

    input_types = tuple(visit_value_info(input, ctx) for input in g.input)
    output_types = tuple(visit_value_info(output, ctx) for output in g.output)

    fn = func.FuncOp(name, (input_types, output_types))

    for input, arg in zip(g.input, fn.body.block.args, strict=True):
        ctx.value_by_name[input.name] = arg

    for node in g.node:
        results = visit_node(node, ctx)
        fn.body.block.add_op(results)

    returned_values = tuple(ctx.value_by_name[output.name] for output in g.output)
    retfn = func.Return(*returned_values)
    fn.body.block.add_op(retfn)

    return fn
