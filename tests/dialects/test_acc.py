"""
Test the usage of the acc (OpenACC) dialect.
"""

import pytest

from xdsl.dialects import acc
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import IntegerType, MemRefType, ModuleOp, f32, i1, i32
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import VerifyException


def _empty_parallel() -> acc.ParallelOp:
    """acc.parallel with an empty body (just the required yield)."""
    return acc.ParallelOp(
        operands=[[], [], [], [], [], [], [], [], [], [], []],
        regions=[Region(Block([acc.YieldOp()]))],
    )


def test_parallel_empty_verifies():
    op = _empty_parallel()
    op.verify()

    assert len(op.regions) == 1
    assert len(op.region.blocks) == 1
    assert len(op.async_operands) == 0
    assert len(op.wait_operands) == 0
    assert len(op.num_gangs) == 0
    assert len(op.num_workers) == 0
    assert len(op.vector_length) == 0
    assert op.if_cond is None
    assert op.self_cond is None
    assert len(op.reduction_operands) == 0
    assert len(op.private_operands) == 0
    assert len(op.firstprivate_operands) == 0
    assert len(op.data_clause_operands) == 0
    assert isinstance(op.region.block.last_op, acc.YieldOp)


def test_parallel_with_operands_verifies():
    """Populate several operand groups and verify segment bookkeeping."""
    async_val = ConstantOp.from_int_and_width(1, i32)
    num_gangs_val = ConstantOp.from_int_and_width(4, i32)
    num_workers_val = ConstantOp.from_int_and_width(8, i32)
    if_cond_val = ConstantOp.from_int_and_width(1, i1)
    data_val = TestOp(result_types=[MemRefType(f32, [10])])

    op = acc.ParallelOp(
        operands=[
            [async_val.result],
            [],
            [num_gangs_val.result],
            [num_workers_val.result],
            [],
            [if_cond_val.result],
            [],
            [],
            [],
            [],
            [data_val.res[0]],
        ],
        regions=[Region(Block([acc.YieldOp()]))],
    )
    op.verify()

    assert op.async_operands[0] is async_val.result
    assert op.num_gangs[0] is num_gangs_val.result
    assert op.num_workers[0] is num_workers_val.result
    assert op.if_cond is if_cond_val.result
    assert op.self_cond is None
    assert op.data_clause_operands[0] is data_val.res[0]


def test_parallel_empty_block_fails():
    op = acc.ParallelOp(
        operands=[[], [], [], [], [], [], [], [], [], [], []],
        regions=[Region(Block([]))],
    )
    with pytest.raises(
        VerifyException,
        match="acc.parallel contains empty block in single-block region",
    ):
        op.verify()


def test_parallel_wrong_terminator_fails():
    """
    A non-terminator op as the last op in the single-block region triggers the
    core-region check before SingleBlockImplicitTerminator.
    """
    body_op = TestOp(result_types=[i32])
    op = acc.ParallelOp(
        operands=[[], [], [], [], [], [], [], [], [], [], []],
        regions=[Region(Block([body_op]))],
    )
    with pytest.raises(
        VerifyException,
        match="terminates block in single-block region but is not a terminator",
    ):
        op.verify()


def test_parallel_non_integer_async_operand_fails():
    """async_operands must be integer or index typed."""
    bad = TestOp(result_types=[MemRefType(f32, [1])])
    op = acc.ParallelOp(
        operands=[
            [bad.res[0]],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        ],
        regions=[Region(Block([acc.YieldOp()]))],
    )
    with pytest.raises(VerifyException):
        op.verify()


def test_parallel_if_cond_non_i1_fails():
    """if_cond must be i1."""
    not_i1 = ConstantOp.from_int_and_width(1, IntegerType(32))
    op = acc.ParallelOp(
        operands=[
            [],
            [],
            [],
            [],
            [],
            [not_i1.result],
            [],
            [],
            [],
            [],
            [],
        ],
        regions=[Region(Block([acc.YieldOp()]))],
    )
    with pytest.raises(VerifyException):
        op.verify()


def test_yield_construction_and_operands():
    """YieldOp inherits AbstractYieldOperation: variadic operands of any type."""
    v = TestOp(result_types=[MemRefType(f32, [10])])
    yield_op = acc.YieldOp(v.res[0])

    assert yield_op.name == "acc.yield"
    assert len(yield_op.arguments) == 1
    assert yield_op.arguments[0] is v.res[0]

    empty_yield = acc.YieldOp()
    assert len(empty_yield.arguments) == 0


def test_yield_outside_parallel_fails():
    """HasParent(ParallelOp) on acc.yield. The yield needs a real parent op
    (not acc.parallel) to exercise the check — HasParent skips detached ops."""
    yield_op = acc.YieldOp()
    module = ModuleOp([yield_op])
    with pytest.raises(
        VerifyException,
        match="'acc.yield' expects parent op 'acc.parallel'",
    ):
        module.verify()


def test_yield_inside_parallel_ok():
    """Building a yield inside acc.parallel's region should verify."""
    op = _empty_parallel()
    op.verify()
    last = op.region.block.last_op
    assert isinstance(last, acc.YieldOp)
    last.verify()


def test_dialect_registration():
    """The dialect exposes exactly the two bootstrap ops."""
    op_names = {op.name for op in acc.ACC.operations}
    assert op_names == {"acc.parallel", "acc.yield"}
    assert acc.ACC.name == "acc"
