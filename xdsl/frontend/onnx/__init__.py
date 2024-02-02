from onnx import GraphProto, NodeProto, TensorShapeProto, TypeProto, ValueInfoProto

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, onnx
from xdsl.dialects.builtin import ModuleOp, TensorType, f32
from xdsl.ir import Attribute, SSAValue
from xdsl.irdl import IRDLOperation


class Ctx:
    type_by_name: dict[str, Attribute]
    value_by_name: dict[str, SSAValue]

    def __init__(self):
        self.type_by_name = {}
        self.value_by_name = {}


ELEM_TYPE = {
    1: f32,
}


def get_elem_type(code: int) -> Attribute:
    if code in ELEM_TYPE:
        return ELEM_TYPE[code]
    else:
        raise ValueError(f"Unknown elem_type: {code}")


def get_shape(s: TensorShapeProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in s.dim)


def get_tensor_type(tensor: TypeProto.Tensor) -> TensorType[Attribute]:
    elem_type = get_elem_type(tensor.elem_type)
    shape = get_shape(tensor.shape)
    return TensorType(elem_type, shape)


def _get_type(type: TypeProto) -> Attribute:
    tt = get_tensor_type(type.tensor_type)
    return tt


def _visit_value_info(i: ValueInfoProto, ctx: Ctx) -> Attribute:
    name = i.name
    t = _get_type(i.type)
    ctx.type_by_name[name] = t
    return t


OP_BY_OP_TYPE: dict[str, type[IRDLOperation]] = {
    "Add": onnx.Add,
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


def visit_graph(g: GraphProto, ctx: Ctx) -> None:
    name = g.name

    input_types = tuple(_visit_value_info(input, ctx) for input in g.input)
    output_types = tuple(_visit_value_info(output, ctx) for output in g.output)

    fn = func.FuncOp(name, (input_types, output_types))

    with ImplicitBuilder(fn.body) as args:
        for input, arg in zip(g.input, args, strict=True):
            ctx.value_by_name[input.name] = arg

        for node in g.node:
            visit_node(node, ctx)

        returned_values = tuple(ctx.value_by_name[output.name] for output in g.output)
        func.Return(*returned_values)


def build_module(graph: GraphProto) -> ModuleOp:
    module = ModuleOp([])

    ctx = Ctx()
    with ImplicitBuilder(module.body):
        visit_graph(graph, ctx)

    return module
