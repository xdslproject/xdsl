# RUN: python %s | mlir-opt --mlir-print-op-generic | filecheck %s

from xdsl.ir import Attribute
from xdsl.dialects.builtin import ModuleOp

from xdsl.dialects import memref
from xdsl.dialects.builtin import f32
from xdsl.printer import Printer
from xdsl.builder import Builder
from xdsl.utils.hints import isa


@ModuleOp
@Builder.implicit_region
def subview_module():
    input = memref.Alloc.get(f32, 0, [100, 200, 300, 400])
    assert isa(input.memref.typ, memref.MemRefType[Attribute])

    subview = memref.Subview.from_static_parameters(
        input, input.memref.typ, [1, 2, 3, 4], [90, 95, 75, 80], [3, 4, 1, 2]
    )
    assert isa(subview.result.typ, memref.MemRefType[Attribute])

    memref.Subview.from_static_parameters(
        subview, subview.result.typ, [2, 5, 6, 1], [70, 50, 20, 64], [1, 5, 3, 2]
    )


p = Printer()
p.print_op(subview_module)

# CHECK:
