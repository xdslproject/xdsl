from onnx import GraphProto

from xdsl.builder import ImplicitBuilder
from xdsl.dialects.builtin import ModuleOp

from .ctx import Ctx
from .visit_graph import visit_graph


def build_module(graph: GraphProto) -> ModuleOp:
    module = ModuleOp([])

    ctx = Ctx()
    with ImplicitBuilder(module.body):
        visit_graph(graph, ctx)

    return module
