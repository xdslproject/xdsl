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

    state: dict[str, int] = _generate_node_state_init(g)

    for input, arg in zip(g.input, fn.body.block.args, strict=True):
        ctx.value_by_name[input.name] = arg

    while _all_nodes_generated(state) is False:
        for node in g.node:
            ready: bool = _all_input_generated(node, state)
            if ready and state[node.name] == 0:
                results = visit_node(node, ctx)
                fn.body.block.add_op(results)
                state[node.name] = 2

    returned_values = tuple(ctx.value_by_name[output.name] for output in g.output)
    print(returned_values)
    retfn = func.Return(*returned_values)
    fn.body.block.add_op(retfn)

    print(ctx.type_by_name)

    return fn


def visit_value_info(i: ValueInfoProto, ctx: OnnxXdslMapping) -> Attribute:
    """Given the onnx ValueInforProto, it returns the corresponding Attribute stored in the context."""
    name = i.name
    t = get_type(i.type)
    ctx.type_by_name[name] = t
    return t


def _generate_node_state_init(graph: GraphProto) -> dict[str, int]:
    """
    Generate the init state of the graph. It returns a dictionary with the information of visited nodes.
    The keys are the names of the nodes, while the related value is an integer representing the visiting state.
    The integer can assume two values: 0 if the node has not been visited yet, 2 if the node was visited.
    Op nodes are initialised to 0, while inputs and outputs are initialsed to 2.
    """
    init_state: dict[str, int] = {}

    for node in graph.node:
        init_state[node.name] = 0

    for input in graph.input:
        init_state[input.name] = 2

    for output in graph.output:
        init_state[output.name] = 2

    return init_state


def _all_input_generated(node: NodeProto, state: dict[str, int]) -> bool:
    """Check if all the input of a given node were visited."""
    for input_name in node.input:
        if state[input_name] != 2:
            return False
    return True


def _all_nodes_generated(state: dict[str, int]) -> bool:
    """Check if all the nodes of the graph were visisted."""
    state_keys: list[str] = list(state.keys())
    for key in state_keys:
        if state[key] != 2:
            return False
    return True
