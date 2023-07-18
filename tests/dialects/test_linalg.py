from typing import Any

from xdsl.builder import Builder
from xdsl.dialects import arith, func, linalg, memref
from xdsl.dialects.builtin import AffineMapAttr, f32
from xdsl.ir.affine import AffineExpr, AffineMap


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

    func.FuncOp("foo", ([], []), funcBody)


def test_loop_range_methods():
    A = memref.Alloc.get(f32, shape=[100, 50])
    B = memref.Alloc.get(f32, shape=[50, 100])
    C = memref.Alloc.get(f32, shape=[100, 100])

    @Builder.implicit_region((f32, f32, f32))
    def body(args: tuple[Any, ...]):
        a, b, c = args
        linalg.Yield(arith.Addf(arith.Mulf(a, b), c))

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)
    k = AffineExpr.dimension(2)

    indexing_maps = [
        AffineMapAttr(AffineMap(3, 0, [i, k])),
        AffineMapAttr(AffineMap(3, 0, [k, j])),
        AffineMapAttr(AffineMap(3, 0, [i, j])),
    ]
    iterators = [
        linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
        linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
        linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
    ]

    op = linalg.Generic(
        [A.results[0], B.results[0]], [C.results[0]], body, indexing_maps, iterators
    )

    loops = op.get_static_loop_ranges()
    assert loops == [100, 50, 50]
