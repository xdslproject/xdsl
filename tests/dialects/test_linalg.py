from typing import Any

from xdsl.builder import Builder
from xdsl.dialects import arith, func, linalg, memref
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AffineMapAttr,
    FloatAttr,
    MemRefType,
    TensorType,
    f32,
)
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.utils.test_value import create_ssa_value


def test_linalg_on_memrefs():
    @Builder.implicit_region(())
    def funcBody(args: tuple[Any, ...]):
        mem = memref.AllocOp.get(f32, shape=[100])
        constant = arith.ConstantOp(FloatAttr(0.0, f32))

        inputs = [constant.results[0]]
        outputs = [mem.results[0]]

        @Builder.implicit_region((f32, f32))
        def body(args: tuple[Any, ...]):
            linalg.YieldOp(args[0])

        indexing = AffineExpr.dimension(0)
        indexing_map = AffineMap(1, 0, (indexing,))

        indexing_maps = [
            AffineMapAttr(AffineMap(1, 0, ())),
            AffineMapAttr(indexing_map),
        ]

        iterators = [linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)]

        linalg.GenericOp(inputs, outputs, body, indexing_maps, iterators)

        func.ReturnOp()

    func.FuncOp("foo", ([], []), funcBody)


def test_matmul_on_memrefs():
    a = memref.AllocOp.get(f32, shape=[100, 50])
    b = memref.AllocOp.get(f32, shape=[50, 100])
    c = memref.AllocOp.get(f32, shape=[100, 100])

    matmul_op = linalg.MatmulOp(inputs=(a.memref, b.memref), outputs=(c.memref,))

    assert matmul_op.result_types == ()


def test_loop_range_methods():
    A = memref.AllocOp.get(f32, shape=[100, 50])
    B = memref.AllocOp.get(f32, shape=[50, 100])
    C = memref.AllocOp.get(f32, shape=[100, 100])

    @Builder.implicit_region((f32, f32, f32))
    def body(args: tuple[Any, ...]):
        a, b, c = args
        linalg.YieldOp(arith.AddfOp(arith.MulfOp(a, b), c))

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)
    k = AffineExpr.dimension(2)

    indexing_maps = [
        AffineMapAttr(AffineMap(3, 0, (i, k))),
        AffineMapAttr(AffineMap(3, 0, (k, j))),
        AffineMapAttr(AffineMap(3, 0, (i, j))),
    ]
    iterators = [
        linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
        linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
        linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
    ]

    op = linalg.GenericOp(
        [A.results[0], B.results[0]], [C.results[0]], body, indexing_maps, iterators
    )

    loops = op.get_static_loop_ranges()
    assert loops == (100, 100, 50)


def test_get_loop_bound_sources_dynamic_memref():
    lhs = create_ssa_value(MemRefType(f32, [DYNAMIC_INDEX, 4]))
    rhs = create_ssa_value(MemRefType(f32, [DYNAMIC_INDEX, 4]))
    out = create_ssa_value(MemRefType(f32, [DYNAMIC_INDEX, 4]))

    @Builder.implicit_region((f32, f32, f32))
    def body(args: tuple[Any, ...]):
        lhs_element, rhs_element, out_element = args
        linalg.YieldOp(
            arith.AddfOp(arith.AddfOp(lhs_element, rhs_element), out_element)
        )

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)

    op = linalg.GenericOp(
        [lhs, rhs],
        [out],
        body,
        [
            AffineMapAttr(AffineMap(2, 0, (i, j))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
        ],
        [
            linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
            linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
        ],
    )

    assert op.get_loop_bound_sources() == (
        linalg.LoopBoundSource(lhs, 0, DYNAMIC_INDEX),
        linalg.LoopBoundSource(lhs, 1, 4),
    )


def test_get_loop_bound_sources_skips_scalar_operand():
    scalar_operand = create_ssa_value(f32)
    tensor_operand = create_ssa_value(TensorType(f32, [7, 3]))
    output = create_ssa_value(MemRefType(f32, [7, 3]))

    @Builder.implicit_region((f32, f32, f32))
    def body(args: tuple[Any, ...]):
        scalar, tensor_element, output_element = args
        linalg.YieldOp(
            arith.AddfOp(arith.AddfOp(scalar, tensor_element), output_element)
        )

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)

    op = linalg.GenericOp(
        [scalar_operand, tensor_operand],
        [output],
        body,
        [
            AffineMapAttr(AffineMap(2, 0, ())),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
        ],
        [
            linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
            linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL),
        ],
    )

    assert op.get_loop_bound_sources() == (
        linalg.LoopBoundSource(tensor_operand, 0, 7),
        linalg.LoopBoundSource(tensor_operand, 1, 3),
    )
