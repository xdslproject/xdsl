"""
Test the usage of scf dialect.
"""

from typing import cast

import pytest

from xdsl.builder import Builder
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IndexType, ModuleOp, i32, i64
from xdsl.dialects.scf import (
    ExecuteRegionOp,
    ForOp,
    IfOp,
    ParallelOp,
    ReduceOp,
    ReduceReturnOp,
    WhileOp,
    YieldOp,
)
from xdsl.dialects.test import TestTermOp
from xdsl.ir import Block, BlockArgument, Region
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.transforms.canonicalize import CanonicalizationRewritePattern
from xdsl.utils.exceptions import DiagnosticException, VerifyException


def test_for_with_loop_carried_verify():
    """Test for with loop-carried variables"""

    lower = ConstantOp.from_int_and_width(0, IndexType())
    upper = ConstantOp.from_int_and_width(42, IndexType())
    step = ConstantOp.from_int_and_width(3, IndexType())
    carried = ConstantOp.from_int_and_width(1, IndexType())

    @Builder.implicit_region((IndexType(), IndexType()))
    def body(_: tuple[BlockArgument, ...]) -> None:
        YieldOp(carried)

    for_op = ForOp(lower, upper, step, [carried], body)

    assert for_op.lb is lower.result
    assert for_op.ub is upper.result
    assert for_op.step is step.result
    assert for_op.iter_args == tuple([carried.result])
    assert for_op.body is body

    assert len(for_op.results) == 1
    assert for_op.results[0].type == carried.result.type
    assert tuple(for_op.operands) == (
        lower.result,
        upper.result,
        step.result,
        carried.result,
    )
    assert for_op.regions == (body,)
    assert for_op.attributes == {}

    for_op.verify()


def test_for_without_loop_carried_verify():
    """Test for without loop-carried variables"""

    lower = ConstantOp.from_int_and_width(0, IndexType())
    upper = ConstantOp.from_int_and_width(42, IndexType())
    step = ConstantOp.from_int_and_width(3, IndexType())

    @Builder.implicit_region((IndexType(),))
    def body(_: tuple[BlockArgument, ...]) -> None:
        YieldOp()

    for_op = ForOp(lower, upper, step, [], body)

    assert for_op.lb is lower.result
    assert for_op.ub is upper.result
    assert for_op.step is step.result
    assert for_op.iter_args == tuple()
    assert for_op.body is body

    assert len(for_op.results) == 0
    assert tuple(for_op.operands) == (
        lower.result,
        upper.result,
        step.result,
    )
    assert for_op.regions == (body,)
    assert for_op.attributes == {}

    for_op.verify()


def test_parallel_no_init_vals():
    lbi = ConstantOp.from_int_and_width(0, IndexType())
    lbj = ConstantOp.from_int_and_width(1, IndexType())
    lbk = ConstantOp.from_int_and_width(18, IndexType())

    ubi = ConstantOp.from_int_and_width(10, IndexType())
    ubj = ConstantOp.from_int_and_width(110, IndexType())
    ubk = ConstantOp.from_int_and_width(92, IndexType())

    si = ConstantOp.from_int_and_width(1, IndexType())
    sj = ConstantOp.from_int_and_width(3, IndexType())
    sk = ConstantOp.from_int_and_width(8, IndexType())

    body = Region()

    lowerBounds = [lbi, lbj, lbk]
    upperBounds = [ubi, ubj, ubk]
    steps = [si, sj, sk]

    p = ParallelOp(lowerBounds, upperBounds, steps, body)

    assert isinstance(p, ParallelOp)
    assert p.lowerBound == tuple(l.result for l in lowerBounds)
    assert p.upperBound == tuple(l.result for l in upperBounds)
    assert p.step == tuple(l.result for l in steps)
    assert p.body is body


def test_parallel_with_init_vals():
    lbi = ConstantOp.from_int_and_width(0, IndexType())
    ubi = ConstantOp.from_int_and_width(10, IndexType())
    si = ConstantOp.from_int_and_width(1, IndexType())

    body = Region()

    init_val = ConstantOp.from_int_and_width(10, i32)

    initVals = [init_val]

    lowerBounds = [lbi]
    upperBounds = [ubi]
    steps = [si]

    p = ParallelOp(lowerBounds, upperBounds, steps, body, initVals)

    assert isinstance(p, ParallelOp)
    assert p.lowerBound == tuple(l.result for l in lowerBounds)
    assert p.upperBound == tuple(l.result for l in upperBounds)
    assert p.step == tuple(l.result for l in steps)
    assert p.body is body
    assert p.initVals == tuple(l.result for l in initVals)


def test_parallel_verify_one_block():
    lbi = ConstantOp.from_int_and_width(0, IndexType())
    ubi = ConstantOp.from_int_and_width(10, IndexType())
    si = ConstantOp.from_int_and_width(1, IndexType())

    body = Region()
    p = ParallelOp([lbi], [ubi], [si], body)
    with pytest.raises(DiagnosticException):
        p.verify()


def test_parallel_verify_num_bounds_equal():
    lbi = ConstantOp.from_int_and_width(0, IndexType())
    ubi = ConstantOp.from_int_and_width(10, IndexType())
    si = ConstantOp.from_int_and_width(1, IndexType())

    lowerBounds = [lbi, lbi]
    upperBounds = [ubi]
    steps = [si]

    body = Region(Block())

    p = ParallelOp(lowerBounds, upperBounds, steps, body)
    with pytest.raises(VerifyException):
        p.verify()

    upperBounds = [ubi, ubi]

    body2 = Region(Block())
    p2 = ParallelOp(lowerBounds, upperBounds, steps, body2)
    with pytest.raises(VerifyException):
        p2.verify()

    lowerBounds = [lbi]
    upperBounds = [ubi]
    steps = [si, si]

    body3 = Region(Block())
    p3 = ParallelOp(lowerBounds, upperBounds, steps, body3)
    with pytest.raises(VerifyException):
        p3.verify()


def test_parallel_verify_only_induction_in_block():
    lbi = ConstantOp.from_int_and_width(0, IndexType())
    ubi = ConstantOp.from_int_and_width(10, IndexType())
    si = ConstantOp.from_int_and_width(1, IndexType())

    init_val = ConstantOp.from_int_and_width(10, i32)

    initVals = [init_val]

    b = Block(arg_types=[IndexType(), i32])
    b.add_op(YieldOp(init_val))

    body = Region(b)
    p = ParallelOp([lbi], [ubi], [si], body, initVals)
    with pytest.raises(VerifyException):
        p.verify()

    b2 = Block(arg_types=[IndexType(), i32, i32])
    b2.add_op(YieldOp(init_val))
    body2 = Region(b2)
    p2 = ParallelOp([lbi], [ubi], [si], body2, initVals)
    with pytest.raises(VerifyException):
        p2.verify()


def test_parallel_block_arg_indextype():
    lbi = ConstantOp.from_int_and_width(0, IndexType())
    ubi = ConstantOp.from_int_and_width(10, IndexType())
    si = ConstantOp.from_int_and_width(1, IndexType())

    b = Block(arg_types=[IndexType()])
    b.add_op(ReduceOp())

    body = Region(b)
    p = ParallelOp([lbi], [ubi], [si], body)
    p.verify()

    b2 = Block(arg_types=[i32])
    b2.add_op(ReduceOp())
    body2 = Region(b2)
    p2 = ParallelOp([lbi], [ubi], [si], body2)
    with pytest.raises(VerifyException):
        p2.verify()


def test_parallel_verify_reduction_and_block_type():
    lbi = ConstantOp.from_int_and_width(0, IndexType())
    ubi = ConstantOp.from_int_and_width(10, IndexType())
    si = ConstantOp.from_int_and_width(1, IndexType())

    init_val = ConstantOp.from_int_and_width(10, i32)

    initVals = [init_val]

    b = Block(arg_types=[IndexType()])

    reduce_constant = ConstantOp.from_int_and_width(100, i32)
    rro = ReduceReturnOp(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    b.add_op(ReduceOp((init_val,), (Region(reduce_block),)))

    body = Region(b)
    p = ParallelOp([lbi], [ubi], [si], body, initVals)
    # This should verify
    p.verify()


def test_parallel_verify_reduction_and_block_type_fails():
    lbi = ConstantOp.from_int_and_width(0, IndexType())
    ubi = ConstantOp.from_int_and_width(10, IndexType())
    si = ConstantOp.from_int_and_width(1, IndexType())

    init_val = ConstantOp.from_int_and_width(10, i64)

    initVals = [init_val]

    b = Block(arg_types=[IndexType()])

    reduce_constant = ConstantOp.from_int_and_width(100, i32)
    rro = ReduceReturnOp(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    b.add_op(ReduceOp((init_val,), (Region(reduce_block),)))

    body = Region(b)
    p = ParallelOp([lbi], [ubi], [si], body, initVals)
    with pytest.raises(VerifyException):
        p.verify()


def test_reduce_op():
    init_val = ConstantOp.from_int_and_width(10, i32)

    reduce_op = ReduceOp((init_val,), (Region(Block(arg_types=[i32, i32])),))

    assert reduce_op.args[0] is init_val.results[0]
    assert reduce_op.args[0].type is i32
    assert len(reduce_op.reductions[0].blocks) == 1
    assert len(reduce_op.reductions[0].block.args) == 2
    assert (
        reduce_op.reductions[0].block.args[0].type
        == reduce_op.reductions[0].block.args[0].type
        == i32
    )


def test_reduce_op_num_block_args():
    init_val = ConstantOp.from_int_and_width(10, i32)
    reduce_constant = ConstantOp.from_int_and_width(100, i32)

    with pytest.raises(
        VerifyException,
        match="scf.reduce block must have exactly two arguments, but ",
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp(
            (init_val,), (Region(Block([rro], arg_types=[i32, i32, i32])),)
        ).verify()

    with pytest.raises(
        VerifyException,
        match="scf.reduce block must have exactly two arguments, but ",
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp((init_val,), (Region(Block([rro], arg_types=[i32])),)).verify()

    with pytest.raises(
        VerifyException,
        match="scf.reduce block must have exactly two arguments, but ",
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp((init_val,), (Region(Block([rro], arg_types=[])),)).verify()


def test_reduce_op_num_block_arg_types():
    init_val = ConstantOp.from_int_and_width(10, i32)
    reduce_constant = ConstantOp.from_int_and_width(100, i32)

    with pytest.raises(
        VerifyException,
        match="scf.reduce block argument types must be the same but have",
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp((init_val,), (Region(Block([rro], arg_types=[i32, i64])),)).verify()

    with pytest.raises(
        VerifyException,
        match="scf.reduce block argument types must be the same but have",
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp((init_val,), (Region(Block([rro], arg_types=[i64, i32])),)).verify()


def test_reduce_op_num_block_arg_types_match_operand_type():
    init_val = ConstantOp.from_int_and_width(10, i32)

    with pytest.raises(VerifyException):
        ReduceOp((init_val,), (Region(Block(arg_types=[i64, i64])),)).verify()


def test_reduce_return_op_at_end():
    reduce_constant = ConstantOp.from_int_and_width(100, i32)
    rro = ReduceReturnOp(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    init_val = ConstantOp.from_int_and_width(10, i32)
    ReduceOp((init_val,), (Region(reduce_block),)).verify()

    with pytest.raises(
        VerifyException,
        match="'scf.reduce' terminates with operation test.termop instead of scf.reduce.return",
    ):
        ReduceOp(
            (init_val,), (Region(Block([TestTermOp.create()], arg_types=[i32, i32])),)
        ).verify()


def test_reduce_return_type_is_arg_type():
    reduce_constant = ConstantOp.from_int_and_width(100, i32)
    rro = ReduceReturnOp(reduce_constant)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    init_val = ConstantOp.from_int_and_width(10, i64)
    with pytest.raises(VerifyException):
        ReduceOp((init_val,), (Region(reduce_block),)).verify()


def test_reduce_return_op():
    reduce_constant = ConstantOp.from_int_and_width(100, i32)
    rro = ReduceReturnOp(reduce_constant)

    assert rro.result is reduce_constant.results[0]
    assert rro.result.type is i32


def test_reduce_return_type_is_operand_type():
    reduce_constant = ConstantOp.from_int_and_width(100, i32)
    reduce_constant_wrong_type = ConstantOp.from_int_and_width(100, i64)
    rro = ReduceReturnOp(reduce_constant_wrong_type)
    reduce_block = Block(arg_types=[i32, i32])
    reduce_block.add_ops([reduce_constant, rro])

    init_val = ConstantOp.from_int_and_width(10, i32)
    with pytest.raises(
        VerifyException,
        match="scf.reduce.return result type at end of scf.reduce block must",
    ):
        ReduceOp((init_val,), (Region(reduce_block),)).verify()


def test_empty_else():
    # create if without an else block:
    m = ModuleOp(
        [
            t := ConstantOp.from_int_and_width(1, 1),
            IfOp(
                t,
                [],
                [
                    YieldOp(),
                ],
            ),
        ]
    )

    assert len(cast(IfOp, list(m.ops)[1]).false_region.blocks) == 0


def test_while():
    before_block = Block(arg_types=(i32, i32))
    after_block = Block(arg_types=(i32, i32))
    a = ConstantOp.from_int_and_width(0, i32)
    b = ConstantOp.from_int_and_width(0, i32)
    while_loop = WhileOp(
        arguments=[a, b],
        result_types=[i32, i32],
        before_region=[before_block],
        after_region=[after_block],
    )
    assert (len(while_loop.results)) == 2
    assert (len(while_loop.operands)) == 2


def test_execute_region_with_no_blocks_canonicalization():
    execute_region = ExecuteRegionOp((), Region(()))
    mod = ModuleOp([execute_region])
    CanonicalizationRewritePattern().match_and_rewrite(
        execute_region, PatternRewriter(execute_region)
    )
    assert not mod.body.ops
