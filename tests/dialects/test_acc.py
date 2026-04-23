"""
Test the usage of the acc (OpenACC) dialect.
"""

from xdsl.dialects import acc
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    BoolAttr,
    DenseArrayBase,
    IntegerType,
    MemRefType,
    UnitAttr,
    f32,
    i1,
    i32,
)
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
    assert op.async_operands_device_type is None
    assert op.async_only is None
    assert op.wait_operands_segments is None
    assert op.wait_operands_device_type is None
    assert op.has_wait_devnum is None
    assert op.wait_only is None
    assert op.num_gangs_segments is None
    assert op.num_gangs_device_type is None
    assert op.num_workers_device_type is None
    assert op.vector_length_device_type is None
    assert op.self_attr is None
    assert op.default_attr is None
    assert op.combined is None
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


def test_parallel_accepts_device_type_attrs():
    """Per-device-type array attributes land on the op as properties."""
    nvidia = acc.DeviceTypeAttr(acc.DeviceType.NVIDIA)
    host = acc.DeviceTypeAttr(acc.DeviceType.HOST)
    op = acc.ParallelOp(
        region=Region(Block([acc.YieldOp()])),
        async_operands_device_type=ArrayAttr([nvidia]),
        async_only=ArrayAttr([host]),
        wait_operands_device_type=ArrayAttr([nvidia, host]),
        wait_operands_segments=DenseArrayBase.from_list(IntegerType(32), [1, 1]),
        has_wait_devnum=ArrayAttr(
            [BoolAttr.from_bool(False), BoolAttr.from_bool(True)]
        ),
        wait_only=ArrayAttr([host]),
        num_gangs_device_type=ArrayAttr([nvidia]),
        num_gangs_segments=DenseArrayBase.from_list(IntegerType(32), [1]),
        num_workers_device_type=ArrayAttr([nvidia]),
        vector_length_device_type=ArrayAttr([nvidia]),
    )
    op.verify()

    assert op.async_operands_device_type == ArrayAttr([nvidia])
    assert op.async_only == ArrayAttr([host])
    assert op.wait_operands_device_type == ArrayAttr([nvidia, host])
    assert op.wait_operands_segments == DenseArrayBase.from_list(
        IntegerType(32), [1, 1]
    )
    assert op.has_wait_devnum == ArrayAttr(
        [BoolAttr.from_bool(False), BoolAttr.from_bool(True)]
    )
    assert op.wait_only == ArrayAttr([host])
    assert op.num_gangs_device_type == ArrayAttr([nvidia])
    assert op.num_gangs_segments == DenseArrayBase.from_list(IntegerType(32), [1])
    assert op.num_workers_device_type == ArrayAttr([nvidia])
    assert op.vector_length_device_type == ArrayAttr([nvidia])


def test_parallel_unit_and_default_attrs():
    """self_attr / combined accept bool shortcuts; default_attr accepts enum."""
    op = acc.ParallelOp(
        region=Region(Block([acc.YieldOp()])),
        self_attr=True,
        default_attr=acc.ClauseDefaultValue.PRESENT,
        combined=True,
    )
    op.verify()

    assert isinstance(op.self_attr, UnitAttr)
    assert isinstance(op.combined, UnitAttr)
    assert op.default_attr == acc.ClauseDefaultValueAttr(acc.ClauseDefaultValue.PRESENT)

    # The shortcut `False` leaves the properties unset.
    op_off = acc.ParallelOp(region=Region(Block([acc.YieldOp()])))
    assert op_off.self_attr is None
    assert op_off.combined is None

    # Explicitly passing the attribute instance is also supported.
    op_explicit = acc.ParallelOp(
        region=Region(Block([acc.YieldOp()])),
        self_attr=UnitAttr(),
        default_attr=acc.ClauseDefaultValueAttr(acc.ClauseDefaultValue.NONE),
        combined=UnitAttr(),
    )
    assert isinstance(op_explicit.self_attr, UnitAttr)
    assert isinstance(op_explicit.combined, UnitAttr)
    assert op_explicit.default_attr == acc.ClauseDefaultValueAttr(
        acc.ClauseDefaultValue.NONE
    )


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
