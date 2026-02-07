# RUN: python %s | mlir-opt | filecheck %s

from xdsl.builder import Builder
from xdsl.dialects import memref
from xdsl.dialects.builtin import ModuleOp, f32
from xdsl.printer import Printer
from xdsl.utils.hints import isa


@ModuleOp
@Builder.implicit_region
def subview_module():
    input = memref.AllocOp.get(f32, 0, [100, 200, 300, 400])
    assert isa(input.memref.type, memref.MemRefType)

    subview = memref.SubviewOp.from_static_parameters(
        input, input.memref.type, [1, 2, 3, 4], [30, 95, 1, 80], [3, 2, 1, 4]
    )
    assert isa(subview.result.type, memref.MemRefType)

    memref.SubviewOp.from_static_parameters(
        subview, subview.result.type, [2, 5, 0, 6], [20, 15, 1, 20], [1, 5, 1, 3]
    )

    subview_reduced = memref.SubviewOp.from_static_parameters(
        input,
        input.memref.type,
        [1, 2, 3, 4],
        [30, 95, 1, 80],
        [3, 2, 1, 4],
        reduce_rank=True,
    )
    assert isa(subview_reduced.result.type, memref.MemRefType)

    memref.SubviewOp.from_static_parameters(
        subview_reduced,
        subview_reduced.result.type,
        [2, 5, 6],
        [20, 1, 20],
        [1, 5, 3],
        reduce_rank=True,
    )


p = Printer()
p.print_op(subview_module)

# Check the rank-reduced output more than just MLIR verification, cause MLIR verification is a bit more flexible here than desired by the constructor
# CHECK:      %subview_1 = memref.subview %alloc[1, 2, 3, 4] [30, 95, 1, 80] [3, 2, 1, 4] : memref<100x200x300x400xf32> to memref<30x95x80xf32, strided<[72000000, 240000, 4], offset: 24241204>>
# CHECK-NEXT: %subview_2 = memref.subview %subview_1[2, 5, 6] [20, 1, 20] [1, 5, 3] : memref<30x95x80xf32, strided<[72000000, 240000, 4], offset: 24241204>> to memref<20x20xf32, strided<[72000000, 12], offset: 169441228>>
