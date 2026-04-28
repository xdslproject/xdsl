"""
Test the usage of the acc (OpenACC) dialect.
"""

import io

from xdsl.dialects import acc
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    IntegerAttr,
    MemRefType,
    StringAttr,
    UnitAttr,
    f32,
    i1,
    i32,
)
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region
from xdsl.printer import Printer
from xdsl.utils.test_value import create_ssa_value


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
    assert op.wait_operands_device_type is None
    assert op.wait_operands_segments is None
    assert op.has_wait_devnum is None
    assert op.wait_only is None
    assert op.num_gangs_device_type is None
    assert op.num_workers_device_type is None
    assert op.vector_length_device_type is None
    assert op.self_attr is None
    assert op.default_attr is None
    assert op.combined is None
    assert isinstance(op.region.block.last_op, acc.YieldOp)

    # The optional num_gangs clause is anchored on `is_present`, so an empty
    # op should never emit it (covers NumGangs.print's empty-operands guard).
    out = io.StringIO()
    Printer(stream=out).print_op(op)
    assert "num_gangs" not in out.getvalue()


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
        wait_only=ArrayAttr([host]),
        num_gangs_device_type=ArrayAttr([nvidia]),
        num_workers_device_type=ArrayAttr([nvidia]),
        vector_length_device_type=ArrayAttr([nvidia]),
    )
    op.verify()

    assert op.async_operands_device_type == ArrayAttr([nvidia])
    assert op.async_only == ArrayAttr([host])
    assert op.wait_operands_device_type == ArrayAttr([nvidia, host])
    assert op.wait_only == ArrayAttr([host])
    assert op.num_gangs_device_type == ArrayAttr([nvidia])
    assert op.num_workers_device_type == ArrayAttr([nvidia])
    assert op.vector_length_device_type == ArrayAttr([nvidia])


def test_parallel_num_gangs_segments():
    """num_gangs_segments rides through as an i32 DenseArrayBase property."""
    nvidia = acc.DeviceTypeAttr(acc.DeviceType.NVIDIA)
    default = acc.DeviceTypeAttr(acc.DeviceType.DEFAULT)
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)
    c = ConstantOp.from_int_and_width(3, i32)

    op = acc.ParallelOp(
        region=Region(Block([acc.YieldOp()])),
        num_gangs=[a.result, b.result, c.result],
        num_gangs_device_type=ArrayAttr([default, nvidia]),
        num_gangs_segments=DenseArrayBase.from_list(i32, [1, 2]),
    )
    op.verify()

    assert op.num_gangs_device_type == ArrayAttr([default, nvidia])
    segments = op.num_gangs_segments
    assert isinstance(segments, DenseArrayBase)
    assert segments.get_values() == (1, 2)


def test_parallel_num_gangs_print_without_segments_or_dt():
    """NumGangs.print falls back to a single #none group when DT/segments are unset."""
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)

    op = acc.ParallelOp(
        region=Region(Block([acc.YieldOp()])),
        num_gangs=[a.result, b.result],
    )
    op.verify()
    assert op.num_gangs_device_type is None
    assert op.num_gangs_segments is None

    out = io.StringIO()
    Printer(stream=out).print_op(op)
    assert "num_gangs({%0 : i32, %1 : i32})" in out.getvalue()


def test_parallel_wait_print_without_metadata():
    """WaitClause.print falls back when device_types / segments / has_devnum are unset.

    None of these branches are reachable via filecheck round-trip — the parser
    always sets all three properties — so they need a Python-constructed op.
    """
    a = ConstantOp.from_int_and_width(1, i32)
    b = ConstantOp.from_int_and_width(2, i32)

    op = acc.ParallelOp(
        region=Region(Block([acc.YieldOp()])),
        wait_operands=[a.result, b.result],
    )
    op.verify()
    assert op.wait_operands_device_type is None
    assert op.wait_operands_segments is None
    assert op.has_wait_devnum is None

    out = io.StringIO()
    Printer(stream=out).print_op(op)
    text = out.getvalue()
    assert "wait({%0 : i32, %1 : i32})" in text
    assert "devnum:" not in text


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


def test_serial_empty_verifies():
    op = acc.SerialOp(region=Region(Block([acc.YieldOp()])))
    op.verify()

    assert len(op.regions) == 1
    assert len(op.region.blocks) == 1
    assert len(op.async_operands) == 0
    assert len(op.wait_operands) == 0
    assert op.if_cond is None
    assert op.self_cond is None
    assert len(op.reduction_operands) == 0
    assert len(op.private_operands) == 0
    assert len(op.firstprivate_operands) == 0
    assert len(op.data_clause_operands) == 0
    assert op.async_operands_device_type is None
    assert op.async_only is None
    assert op.wait_operands_device_type is None
    assert op.wait_operands_segments is None
    assert op.has_wait_devnum is None
    assert op.wait_only is None
    assert op.self_attr is None
    assert op.default_attr is None
    assert op.combined is None
    assert isinstance(op.region.block.last_op, acc.YieldOp)


def test_serial_with_operands_verifies():
    """Populate several operand groups and verify segment bookkeeping."""
    async_val = ConstantOp.from_int_and_width(1, i32)
    if_cond_val = ConstantOp.from_int_and_width(1, i1)
    data_val = TestOp(result_types=[MemRefType(f32, [10])])
    private_val = TestOp(result_types=[MemRefType(f32, [10])])

    op = acc.SerialOp(
        region=Region(Block([acc.YieldOp()])),
        async_operands=[async_val.result],
        if_cond=if_cond_val.result,
        data_clause_operands=[data_val.res[0]],
        private_operands=[private_val.res[0]],
    )
    op.verify()

    assert op.async_operands[0] is async_val.result
    assert op.if_cond is if_cond_val.result
    assert op.self_cond is None
    assert op.data_clause_operands[0] is data_val.res[0]
    assert op.private_operands[0] is private_val.res[0]


def test_serial_accepts_device_type_attrs():
    """Per-device-type array attributes land on the op as properties."""
    nvidia = acc.DeviceTypeAttr(acc.DeviceType.NVIDIA)
    host = acc.DeviceTypeAttr(acc.DeviceType.HOST)
    op = acc.SerialOp(
        region=Region(Block([acc.YieldOp()])),
        async_operands_device_type=ArrayAttr([nvidia]),
        async_only=ArrayAttr([host]),
        wait_operands_device_type=ArrayAttr([nvidia, host]),
        wait_only=ArrayAttr([host]),
    )
    op.verify()

    assert op.async_operands_device_type == ArrayAttr([nvidia])
    assert op.async_only == ArrayAttr([host])
    assert op.wait_operands_device_type == ArrayAttr([nvidia, host])
    assert op.wait_only == ArrayAttr([host])


def test_serial_unit_and_default_attrs():
    """self_attr / combined accept bool shortcuts; default_attr accepts enum."""
    op = acc.SerialOp(
        region=Region(Block([acc.YieldOp()])),
        self_attr=True,
        default_attr=acc.ClauseDefaultValue.PRESENT,
        combined=True,
    )
    op.verify()

    assert isinstance(op.self_attr, UnitAttr)
    assert isinstance(op.combined, UnitAttr)
    assert op.default_attr == acc.ClauseDefaultValueAttr(acc.ClauseDefaultValue.PRESENT)

    op_off = acc.SerialOp(region=Region(Block([acc.YieldOp()])))
    assert op_off.self_attr is None
    assert op_off.combined is None

    op_explicit = acc.SerialOp(
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


def test_kernels_init_bool_shortcuts():
    """The bool-shortcut branch in KernelsOp.__init__ (self_attr=True / combined=True
    / default_attr=enum) cannot be reached via the parser — filecheck would have to
    pass already-constructed attributes — so this Python-only branch lives here."""
    op = acc.KernelsOp(
        region=Region(Block()),
        self_attr=True,
        default_attr=acc.ClauseDefaultValue.PRESENT,
        combined=True,
    )
    op.verify()

    assert isinstance(op.self_attr, UnitAttr)
    assert isinstance(op.combined, UnitAttr)
    assert op.default_attr == acc.ClauseDefaultValueAttr(acc.ClauseDefaultValue.PRESENT)

    op_off = acc.KernelsOp(region=Region(Block()))
    assert op_off.self_attr is None
    assert op_off.combined is None


def test_data_clause_modifier_attr_constructor():
    """The bit-enum constructor accepts a frozenset of enum members and stores
    it on `.data`. Pretty-form printing/parsing is covered by filecheck in
    `tests/filecheck/dialects/acc/attrs.mlir`; the constructor surface is
    Python-only and lives here."""
    none = acc.DataClauseModifierAttr(frozenset[acc.DataClauseModifier]())
    readonly = acc.DataClauseModifierAttr(frozenset({acc.DataClauseModifier.READONLY}))
    multi = acc.DataClauseModifierAttr(
        frozenset({acc.DataClauseModifier.READONLY, acc.DataClauseModifier.ZERO})
    )

    assert none.data == frozenset()
    assert readonly.data == frozenset({acc.DataClauseModifier.READONLY})
    assert multi.data == frozenset(
        {acc.DataClauseModifier.READONLY, acc.DataClauseModifier.ZERO}
    )


def test_copyin_minimal_defaulted_props_absent_from_dict():
    """Defaulted props (`dataClause` / `structured` / `implicit` / `modifiers`)
    must be *absent* from `op.properties` when not explicitly set, even though
    the accessor reads back the default value. This is the load-bearing
    invariant that drives attr-dict elision on print — filecheck observes the
    elided text but cannot distinguish "absent from dict" from "present and
    matching default", so the dict-state assertion lives here."""
    op = acc.CopyinOp(var=create_ssa_value(MemRefType(f32, [10])))
    op.verify()

    assert "dataClause" not in op.properties
    assert "structured" not in op.properties
    assert "implicit" not in op.properties
    assert "modifiers" not in op.properties


def test_copyin_builder_shortcuts():
    """The Python `__init__` accepts bool / str / `DataClause` shortcuts and
    converts them to the right attribute kinds. This is a builder-only
    code path: the parser never sees these Python types, so filecheck
    cannot exercise the conversions."""
    op = acc.CopyinOp(
        var=create_ssa_value(MemRefType(f32, [10])),
        data_clause=acc.DataClause.ACC_COPYIN_READONLY,
        structured=False,
        implicit=True,
        var_name="foo",
    )
    op.verify()

    assert op.data_clause == acc.DataClauseAttr(acc.DataClause.ACC_COPYIN_READONLY)
    assert op.structured == IntegerAttr.from_bool(False)
    assert op.implicit == IntegerAttr.from_bool(True)
    assert op.var_name == StringAttr("foo")
