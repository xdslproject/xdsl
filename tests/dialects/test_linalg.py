from collections.abc import Callable, Sequence
from typing import Any

import pytest

from xdsl.builder import Builder
from xdsl.dialects import arith, func, linalg, memref, test
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AffineMapAttr,
    ArrayAttr,
    DenseArrayBase,
    FloatAttr,
    MemRefType,
    TensorType,
    f32,
    i64,
)
from xdsl.dialects.linalg.abstract_ops import LinalgMemoryEffect, LinalgSpeculatable
from xdsl.ir import BlockArgument, Operation, SSAValue
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.traits import (
    EffectInstance,
    MemoryEffectKind,
    Pure,
    SingleBlockImplicitTerminator,
    get_effects,
    is_speculatable,
)
from xdsl.utils.hints import isa
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
            linalg.ops.YieldOp(args[0])

        indexing = AffineExpr.dimension(0)
        indexing_map = AffineMap(1, 0, (indexing,))

        indexing_maps = [
            AffineMapAttr(AffineMap(1, 0, ())),
            AffineMapAttr(indexing_map),
        ]

        iterators = [linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL)]

        linalg.ops.GenericOp(inputs, outputs, body, indexing_maps, iterators)

        func.ReturnOp()

    func.FuncOp("foo", ([], []), funcBody)


def test_matmul_on_memrefs():
    a = memref.AllocOp.get(f32, shape=[100, 50])
    b = memref.AllocOp.get(f32, shape=[50, 100])
    c = memref.AllocOp.get(f32, shape=[100, 100])

    matmul_op = linalg.ops.MatmulOp(inputs=(a.memref, b.memref), outputs=(c.memref,))

    assert matmul_op.result_types == ()


def test_loop_range_methods():
    A = memref.AllocOp.get(f32, shape=[100, 50])
    B = memref.AllocOp.get(f32, shape=[50, 100])
    C = memref.AllocOp.get(f32, shape=[100, 100])

    @Builder.implicit_region((f32, f32, f32))
    def body(args: tuple[Any, ...]):
        a, b, c = args
        linalg.ops.YieldOp(arith.AddfOp(arith.MulfOp(a, b), c))

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)
    k = AffineExpr.dimension(2)

    indexing_maps = [
        AffineMapAttr(AffineMap(3, 0, (i, k))),
        AffineMapAttr(AffineMap(3, 0, (k, j))),
        AffineMapAttr(AffineMap(3, 0, (i, j))),
    ]
    iterators = [
        linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
        linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
        linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
    ]

    op = linalg.ops.GenericOp(
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
        linalg.ops.YieldOp(
            arith.AddfOp(arith.AddfOp(lhs_element, rhs_element), out_element)
        )

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)

    op = linalg.ops.GenericOp(
        [lhs, rhs],
        [out],
        body,
        [
            AffineMapAttr(AffineMap(2, 0, (i, j))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
        ],
        [
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
        ],
    )

    assert op.get_loop_bound_sources() == (
        linalg.abstract_ops.LoopBoundSource(lhs, 0, DYNAMIC_INDEX),
        linalg.abstract_ops.LoopBoundSource(lhs, 1, 4),
    )


def test_get_loop_bound_sources_skips_scalar_operand():
    scalar_operand = create_ssa_value(f32)
    tensor_operand = create_ssa_value(TensorType(f32, [7, 3]))
    output = create_ssa_value(MemRefType(f32, [7, 3]))

    @Builder.implicit_region((f32, f32, f32))
    def body(args: tuple[Any, ...]):
        scalar, tensor_element, output_element = args
        linalg.ops.YieldOp(
            arith.AddfOp(arith.AddfOp(scalar, tensor_element), output_element)
        )

    i = AffineExpr.dimension(0)
    j = AffineExpr.dimension(1)

    op = linalg.ops.GenericOp(
        [scalar_operand, tensor_operand],
        [output],
        body,
        [
            AffineMapAttr(AffineMap(2, 0, ())),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
            AffineMapAttr(AffineMap(2, 0, (i, j))),
        ],
        [
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
            linalg.attrs.IteratorTypeAttr(linalg.attrs.IteratorType.PARALLEL),
        ],
    )

    assert op.get_loop_bound_sources() == (
        linalg.abstract_ops.LoopBoundSource(tensor_operand, 0, 7),
        linalg.abstract_ops.LoopBoundSource(tensor_operand, 1, 3),
    )


def test_deprecated_top_level_linalg_api_still_works():
    with pytest.warns(
        DeprecationWarning,
        match="Importing 'GenericOp' directly from 'xdsl.dialects.linalg' is deprecated",
    ):
        generic_op = linalg.GenericOp

    assert generic_op is linalg.ops.GenericOp

    with pytest.warns(
        DeprecationWarning,
        match="Importing 'IteratorTypeAttr' directly from 'xdsl.dialects.linalg' is deprecated",
    ):
        iterator_type_attr = linalg.IteratorTypeAttr

    assert iterator_type_attr is linalg.attrs.IteratorTypeAttr

    with pytest.warns(
        DeprecationWarning,
        match="Importing 'LinalgStructuredOperation' directly from 'xdsl.dialects.linalg' is deprecated",
    ):
        structured_op = linalg.LinalgStructuredOperation

    assert structured_op is linalg.abstract_ops.LinalgStructuredOperation


@pytest.mark.parametrize(
    "op_type",
    [
        linalg.ops.ExpOp,
        linalg.ops.LogOp,
        linalg.ops.SqrtOp,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (1,),
        (3, 1),
        (4, 3, 2),
    ],
)
def test_unary_elementwise_op_identity_map_and_parallel_iters(
    op_type: type[linalg.ops.ExpOp | linalg.ops.LogOp | linalg.ops.SqrtOp],
    shape: tuple[int, ...],
):
    t = TensorType(f32, shape)

    op = op_type(
        [create_ssa_value(t)],
        [create_ssa_value(t)],
    )

    # Should have exactly 2 access maps: [in, out]
    assert op.get_indexing_maps() == ArrayAttr(
        (
            AffineMapAttr(AffineMap.identity(len(shape))),
            AffineMapAttr(AffineMap.identity(len(shape))),
        )
    )

    # Should have all parallel iterators
    assert op.get_iterator_types() == ArrayAttr(
        [linalg.attrs.IteratorTypeAttr.parallel() for _ in range(len(shape))]
    )


def _generic(
    ins: Sequence[SSAValue],
    outs: Sequence[SSAValue],
    payload: Callable[[tuple[BlockArgument, ...]], SSAValue],
) -> linalg.ops.GenericOp:
    """1-D parallel `linalg.generic` whose payload yields `payload(args)`."""
    arg_types = linalg.abstract_ops.NamedOperation.body_arg_types((*ins, *outs))

    @Builder.implicit_region(arg_types)
    def body(args: tuple[BlockArgument, ...]):
        linalg.ops.YieldOp(payload(args))

    identity = AffineMapAttr(AffineMap.identity(1))
    return linalg.ops.GenericOp(
        ins,
        outs,
        body,
        [identity] * (len(ins) + len(outs)),
        [linalg.attrs.IteratorTypeAttr.parallel()],
        [o.type for o in outs if isa(o.type, TensorType)],
    )


def _addf(args: Sequence[SSAValue]) -> SSAValue:
    return arith.AddfOp(args[0], args[1]).result


def _reduce(input: SSAValue, init: SSAValue) -> linalg.ops.ReduceOp:
    @Builder.implicit_region((f32, f32))
    def body(args: tuple[BlockArgument, ...]):
        linalg.ops.YieldOp(_addf(args))

    return linalg.ops.ReduceOp(input, init, DenseArrayBase.from_list(i64, [0]), body)


def _effects(*items: tuple[MemoryEffectKind, SSAValue]) -> set[EffectInstance]:
    return {EffectInstance(kind, value) for kind, value in items}


@pytest.mark.parametrize(
    "op_type",
    [
        op
        for op in linalg.Linalg.operations
        if issubclass(op, linalg.abstract_ops.DestinationStyleOperation)
    ],
)
def test_dps_op_traits(op_type: type[Operation]):
    for trait in (
        LinalgMemoryEffect,
        LinalgSpeculatable,
        SingleBlockImplicitTerminator,
    ):
        assert op_type.has_trait(trait)


@pytest.mark.parametrize("op_type", [linalg.ops.YieldOp, linalg.ops.IndexOp])
def test_payload_op_pure(op_type: type[Operation]):
    assert op_type.has_trait(Pure)


M = MemRefType(f32, (4,))
T = TensorType(f32, (4,))


def test_effects_tensor():
    t = create_ssa_value(T)
    op = _generic([t], [create_ssa_value(T)], _addf)
    assert get_effects(op) == set()
    assert is_speculatable(op)


def test_effects_memref_init_unused():
    a, b, c = (create_ssa_value(M) for _ in range(3))
    assert get_effects(linalg.ops.AddOp((a, b), (c,))) == _effects(
        (MemoryEffectKind.READ, a),
        (MemoryEffectKind.READ, b),
        (MemoryEffectKind.WRITE, c),
    )


def test_effects_memref_init_used():
    a, c = create_ssa_value(M), create_ssa_value(M)
    op = _generic([a], [c], _addf)
    assert get_effects(op) == _effects(
        (MemoryEffectKind.READ, a),
        (MemoryEffectKind.READ, c),
        (MemoryEffectKind.WRITE, c),
    )
    assert not is_speculatable(op)


def test_effects_scalar_input_ignored():
    s, c = create_ssa_value(f32), create_ssa_value(M)
    assert get_effects(linalg.ops.FillOp((s,), (c,))) == _effects(
        (MemoryEffectKind.WRITE, c)
    )


def test_effects_mixed_tensor_memref():
    t, c = create_ssa_value(T), create_ssa_value(M)
    op = _generic([t], [c], lambda args: args[0])
    assert get_effects(op) == _effects((MemoryEffectKind.WRITE, c))
    assert not is_speculatable(op)


def test_effects_broadcast():
    a, c = create_ssa_value(M), create_ssa_value(MemRefType(f32, (2, 4)))
    op = linalg.ops.BroadcastOp(a, c, DenseArrayBase.from_list(i64, [0]))
    assert get_effects(op) == _effects(
        (MemoryEffectKind.READ, a), (MemoryEffectKind.WRITE, c)
    )


def test_effects_reduce():
    a, c = create_ssa_value(MemRefType(f32, (2, 4))), create_ssa_value(M)
    assert get_effects(_reduce(a, c)) == _effects(
        (MemoryEffectKind.READ, a),
        (MemoryEffectKind.READ, c),
        (MemoryEffectKind.WRITE, c),
    )
    assert is_speculatable(
        _reduce(create_ssa_value(TensorType(f32, (2, 4))), create_ssa_value(T))
    )


def test_effects_include_payload():
    t, p = create_ssa_value(T), create_ssa_value(M)
    op = _generic(
        [t],
        [create_ssa_value(T)],
        lambda a: test.TestReadOp((p,), (f32,)).results[0],
    )
    assert get_effects(op) == {EffectInstance(MemoryEffectKind.READ)}


def test_effects_unknown_payload():
    t = create_ssa_value(T)
    op = _generic(
        [t], [create_ssa_value(T)], lambda a: test.TestOp((), (f32,)).results[0]
    )
    assert get_effects(op) is None
    assert not is_speculatable(op)


def test_memory_effect_requires_dps_op():
    with pytest.raises(TypeError, match="DestinationStyleOperation"):
        LinalgMemoryEffect.get_effects(test.TestOp())
