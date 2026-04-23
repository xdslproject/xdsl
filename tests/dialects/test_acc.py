"""
Test the usage of the acc (OpenACC) dialect.
"""

from xdsl.dialects import acc
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import MemRefType, f32, i1, i32
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region


def _empty_parallel() -> acc.ParallelOp:
    """acc.parallel with an empty body (just the required yield)."""
    return acc.ParallelOp(region=Region(Block([acc.YieldOp()])))


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
        region=Region(Block([acc.YieldOp()])),
        async_operands=[async_val.result],
        num_gangs=[num_gangs_val.result],
        num_workers=[num_workers_val.result],
        if_cond=if_cond_val.result,
        data_clause_operands=[data_val.res[0]],
    )
    op.verify()

    assert op.async_operands[0] is async_val.result
    assert op.num_gangs[0] is num_gangs_val.result
    assert op.num_workers[0] is num_workers_val.result
    assert op.if_cond is if_cond_val.result
    assert op.self_cond is None
    assert op.data_clause_operands[0] is data_val.res[0]


def test_yield_construction_and_operands():
    """YieldOp inherits AbstractYieldOperation: variadic operands of any type."""
    v = TestOp(result_types=[MemRefType(f32, [10])])
    yield_op = acc.YieldOp(v.res[0])

    assert yield_op.name == "acc.yield"
    assert len(yield_op.arguments) == 1
    assert yield_op.arguments[0] is v.res[0]

    empty_yield = acc.YieldOp()
    assert len(empty_yield.arguments) == 0


def test_yield_inside_parallel_ok():
    """Building a yield inside acc.parallel's region should verify."""
    op = _empty_parallel()
    op.verify()
    last = op.region.block.last_op
    assert isinstance(last, acc.YieldOp)
    last.verify()
