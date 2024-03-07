from onnx import TypeProto, ValueInfoProto

from xdsl.ir import Attribute

from .ctx import Ctx
from .shape_type import get_tensor_type


def get_type(type: TypeProto) -> Attribute:
    tt = get_tensor_type(type.tensor_type)
    return tt


def visit_value_info(i: ValueInfoProto, ctx: Ctx) -> Attribute:
    name = i.name
    t = get_type(i.type)
    ctx.type_by_name[name] = t
    return t
