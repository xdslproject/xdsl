from onnx import GraphProto, NodeProto, ValueInfoProto

from xdsl.dialects import func, onnx
from xdsl.dialects.builtin import ModuleOp
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

    op = OP_BY_OP_TYPE[node.op_type]

    operands = tuple(ctx.value_by_name[name] for name in node.input)
    result_types = tuple(ctx.type_by_name[name] for name in node.output)

    op = op.build(operands=operands, result_types=result_types)

    for output_name, result in zip(node.output, op.results, strict=True):
        ctx.value_by_name[output_name] = result

    return op


def build_module(graph: GraphProto) -> ModuleOp:
    """Create the ModuleOp based on the onnx graph provided."""
    module = ModuleOp([])

    ctx = OnnxXdslMapping()
    fn = visit_graph(graph, ctx)
    module.regions[0].block.add_op(fn)

    return module


def visit_graph(g: GraphProto, ctx: OnnxXdslMapping) -> IRDLOperation:
    """
    Visit the onnx graph to update the onnx context.
    The nodes of the graph are visited only if all the inputs have already been visited.
    When the node is visited, the associated state switches from 0 to 2.
    The process is iterated until all the nodes are generated.
    """

    name = g.name

    input_types = tuple(visit_value_info(input, ctx) for input in g.input)
    output_types = tuple(visit_value_info(output, ctx) for output in g.output)

    fn = func.FuncOp(name, (input_types, output_types))

    for input, arg in zip(g.input, fn.body.block.args, strict=True):
        ctx.value_by_name[input.name] = arg

    linearized: list[NodeProto] = []
    nodes: list[NodeProto] = []
    for node in g.node:
        nodes.append(node)
    for node in nodes:
        _linearize_dag(node, nodes, linearized)

    for node in linearized:
        results = visit_node(node, ctx)
        fn.body.block.add_op(results)

    returned_values = tuple(ctx.value_by_name[output.name] for output in g.output)
    retfn = func.Return(*returned_values)
    fn.body.block.add_op(retfn)

    return fn


def visit_value_info(i: ValueInfoProto, ctx: OnnxXdslMapping) -> Attribute:
    """Given the onnx ValueInforProto, it returns the corresponding Attribute stored in the context."""
    name = i.name
    t = get_type(i.type)
    ctx.type_by_name[name] = t
    return t


def _linearize_dag(
    node: NodeProto, nodes: list[NodeProto], linearized: list[NodeProto]
) -> None:
    """
    Linearization of the DAG. The nodes are sorted so that for each nodes of the list, the dependencies are only on the "left".
    """
    if node in linearized:
        return
    inputs: list[NodeProto] = _get_input_nodes(node, nodes)
    for input in inputs:
        _linearize_dag(input, nodes, linearized)
    linearized.append(node)


def _get_input_nodes(node: NodeProto, nodes: list[NodeProto]) -> list[NodeProto]:
    """
    Get the references of the input nodes of a given node.
    """
    inputs: list[NodeProto] = []
    input_names = node.input
    for curr_node in nodes:
        output_names = curr_node.output
        for output_name in output_names:
            if output_name in input_names:
                inputs.append(curr_node)
                break
    return inputs
