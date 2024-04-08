from onnx import ValueInfoProto

from xdsl.frontend.onnx.type import get_type
from xdsl.ir import Attribute


class OnnxXdslMapping:
    """The representation of the onnx context."""

    type_by_name: dict[str, Attribute]

    def __init__(self):
        self.type_by_name = {}


def visit_value_info(i: ValueInfoProto, ctx: OnnxXdslMapping) -> Attribute:
    """Given the onnx ValueInforProto, it returns the corresponding Attribute stored in the context."""
    name = i.name

    if name in ctx.type_by_name:
        return ctx.type_by_name[name]

    t = get_type(i.type)
    ctx.type_by_name[name] = t
    return t
