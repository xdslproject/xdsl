from typing import Any

from xdsl.builder import Builder
from xdsl.dialects import arith, func, linalg, memref
from xdsl.dialects.builtin import AffineMapAttr, f32
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.printer import Printer


def test_linalg_on_memrefs():
    @Builder.implicit_region(())
    def funcBody(args: tuple[Any, ...]):
        mem = memref.Alloc.get(f32, shape=[100])
        constant = arith.Constant.from_float_and_width(0.0, f32)

        inputs = [constant.results[0]]
        outputs = [mem.results[0]]

        @Builder.implicit_region((f32, f32))
        def body(args: tuple[Any, ...]):
            linalg.Yield(args[0])

        indexing = AffineExpr.dimension(0)
        indexing_map = AffineMap(1, 0, [indexing])

        indexing_maps = [
            AffineMapAttr(AffineMap(1, 0, [])),
            AffineMapAttr(indexing_map),
        ]

        iterators = [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)]

        linalg.Generic(inputs, outputs, body, indexing_maps, iterators)

        func.Return()

    foo = func.FuncOp("foo", ([], []), funcBody)

    printer = Printer()
    printer.print(foo)


test_linalg_on_memrefs()
