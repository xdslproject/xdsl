from onnx import GraphProto
from utils import visit_value_info

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func

from .ctx import Ctx
from .visit_node import visit_node


def visit_graph(g: GraphProto, ctx: Ctx) -> None:
    name = g.name

    input_types = tuple(visit_value_info(input, ctx) for input in g.input)
    output_types = tuple(visit_value_info(output, ctx) for output in g.output)

    fn = func.FuncOp(name, (input_types, output_types))

    with ImplicitBuilder(fn.body) as args:
        for input, arg in zip(g.input, args, strict=True):
            ctx.value_by_name[input.name] = arg

        for node in g.node:
            visit_node(node, ctx)

        returned_values = tuple(ctx.value_by_name[output.name] for output in g.output)
        func.Return(*returned_values)
