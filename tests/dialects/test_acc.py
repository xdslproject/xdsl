"""
Test the usage of the acc (OpenACC) dialect.
"""

import io

from xdsl.dialects import acc
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    IndexType,
    IntegerAttr,
    MemRefType,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    f32,
    i1,
    i32,
    i64,
)
from xdsl.dialects.test import TestOp
from xdsl.ir import Block, Region
from xdsl.printer import Printer
from xdsl.traits import NoMemoryEffect
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


def test_parallel_wait_only_keyword_only_no_operands_print():
    """`_print_wait_body` has a printer-only branch — `wait_only` set to a
    non-default device-type list with no `wait_operands` — that the parser
    can never produce (`_parse_wait_body` requires a `,` after the
    `[dt-list]`). This is the Python-only state the roadmap exception
    covers: a property combination constructable only from the builder."""
    nvidia = acc.DeviceTypeAttr(acc.DeviceType.NVIDIA)
    op = acc.ParallelOp(
        region=Region(Block([acc.YieldOp()])),
        wait_only=ArrayAttr([nvidia]),
    )
    op.verify()
    assert len(op.wait_operands) == 0

    out = io.StringIO()
    Printer(stream=out).print_op(op)
    assert "wait([#acc.device_type<nvidia>])" in out.getvalue()


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
        region=Region(Block([acc.TerminatorOp()])),
        self_attr=True,
        default_attr=acc.ClauseDefaultValue.PRESENT,
        combined=True,
    )
    op.verify()

    assert isinstance(op.self_attr, UnitAttr)
    assert isinstance(op.combined, UnitAttr)
    assert op.default_attr == acc.ClauseDefaultValueAttr(acc.ClauseDefaultValue.PRESENT)

    op_off = acc.KernelsOp(region=Region(Block([acc.TerminatorOp()])))
    assert op_off.self_attr is None
    assert op_off.combined is None


def test_terminator_construction():
    """TerminatorOp's `__init__` is unreachable from the round-trip parser
    (declarative `attr-dict` builds the op via IRDL's generic constructor),
    so its body lives or dies by a Python-side construction call."""
    term = acc.TerminatorOp()
    assert term.name == "acc.terminator"


def test_data_init_enum_shortcut():
    """The `default_attr=ClauseDefaultValue.X` enum-shortcut branch in
    DataOp.__init__ is not reachable via the parser (which produces a
    fully-formed ClauseDefaultValueAttr)."""
    op = acc.DataOp(
        region=Region(Block([acc.TerminatorOp()])),
        default_attr=acc.ClauseDefaultValue.PRESENT,
    )
    assert op.default_attr == acc.ClauseDefaultValueAttr(acc.ClauseDefaultValue.PRESENT)


def test_host_data_init_bool_shortcut():
    """The `if_present=True` bool-shortcut branch in HostDataOp.__init__ is
    not reachable via the parser (which produces a UnitAttr directly)."""
    use_dev = acc.UseDeviceOp(var=create_ssa_value(MemRefType(f32, [10])))
    op = acc.HostDataOp(
        region=Region(Block([acc.TerminatorOp()])),
        data_clause_operands=[use_dev],
        if_present=True,
    )
    assert isinstance(op.if_present, UnitAttr)


def _empty_kernel_environment_block() -> Block:
    """`acc.kernel_environment` has `NoTerminator` so the body is a single
    block that need not end in a terminator. Build one with a single
    placeholder op so the SizedRegion<1> verifier sees a non-empty region."""
    return Block([TestOp()])


def test_kernel_environment_empty_verifies():
    op = acc.KernelEnvironmentOp(region=Region(_empty_kernel_environment_block()))
    op.verify()

    assert len(op.data_clause_operands) == 0
    assert len(op.async_operands) == 0
    assert len(op.wait_operands) == 0
    assert op.async_operands_device_type is None
    assert op.async_only is None
    assert op.wait_operands_segments is None
    assert op.wait_operands_device_type is None
    assert op.has_wait_devnum is None
    assert op.wait_only is None


def test_kernel_environment_unset_props_absent_from_dict():
    """The optional clause properties are *absent* from `op.properties`
    when not provided (rather than stored as None). The MLIR-interop
    round-trip relies on this — extra all-None entries would print as
    explicit attributes in the trailing attr-dict."""
    op = acc.KernelEnvironmentOp(region=Region(_empty_kernel_environment_block()))

    for prop in (
        "asyncOperandsDeviceType",
        "asyncOnly",
        "waitOperandsSegments",
        "waitOperandsDeviceType",
        "hasWaitDevnum",
        "waitOnly",
    ):
        assert prop not in op.properties


def test_kernel_environment_wait_only_no_operands_print():
    """`_print_wait_body` has a printer-only branch — `wait_only` set to a
    non-default device-type list with no `wait_operands` — that the parser
    can never produce (`_parse_wait_body` requires a `,` after the `[dt-list]`).
    This is the Python-only state the roadmap exception covers: a property
    combination constructable only from the builder."""
    nvidia = acc.DeviceTypeAttr(acc.DeviceType.NVIDIA)
    op = acc.KernelEnvironmentOp(
        region=Region(_empty_kernel_environment_block()),
        wait_only=ArrayAttr([nvidia]),
    )
    op.verify()
    assert len(op.wait_operands) == 0

    out = io.StringIO()
    Printer(stream=out).print_op(op)
    assert "wait([#acc.device_type<nvidia>])" in out.getvalue()


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


def test_copyout_minimal_defaulted_props_absent_from_dict():
    """Same load-bearing invariant as `acc.copyin` — defaulted props on the
    `_DataExitOperationWithVarPtr` mixin must be *absent* from the dict when
    not set, otherwise attr-dict elision on print silently fails."""
    acc_var = create_ssa_value(MemRefType(f32, [10]))
    var = create_ssa_value(MemRefType(f32, [10]))
    op = acc.CopyoutOp(acc_var=acc_var, var=var)
    op.verify()

    assert "dataClause" not in op.properties
    assert "structured" not in op.properties
    assert "implicit" not in op.properties
    assert "modifiers" not in op.properties
    assert op.acc_var is acc_var
    assert op.var is var


def test_copyout_builder_shortcuts():
    """Bool / str / `DataClause` shortcuts on `_DataExitOperationWithVarPtr`'s
    `__init__`. Builder-only code path; the parser only sees attribute
    instances, so filecheck cannot exercise these conversions."""
    acc_var = create_ssa_value(MemRefType(f32, [10]))
    var = create_ssa_value(MemRefType(f32, [10]))
    op = acc.CopyoutOp(
        acc_var=acc_var,
        var=var,
        data_clause=acc.DataClause.ACC_COPYOUT_ZERO,
        structured=False,
        implicit=True,
        var_name="myvar",
    )
    op.verify()

    assert op.data_clause == acc.DataClauseAttr(acc.DataClause.ACC_COPYOUT_ZERO)
    assert op.structured == IntegerAttr.from_bool(False)
    assert op.implicit == IntegerAttr.from_bool(True)
    assert op.var_name == StringAttr("myvar")


def test_delete_minimal_defaulted_props_absent_from_dict():
    """Same invariant for the `_DataExitOperationNoVarPtr` mixin (no `var`,
    no `varType`). `acc.delete` exercises the no-host-pointer branch of the
    exit-shape `__init__`; the dict-state assertion is independently
    load-bearing because this mixin's `__init__` is a separate method that
    must reproduce the absence-on-default behavior."""
    acc_var = create_ssa_value(MemRefType(f32, [10]))
    op = acc.DeleteOp(acc_var=acc_var)
    op.verify()

    assert "dataClause" not in op.properties
    assert "structured" not in op.properties
    assert "implicit" not in op.properties
    assert "modifiers" not in op.properties
    assert op.acc_var is acc_var


def test_delete_builder_shortcuts():
    """Bool / str / `DataClause` shortcuts on `_DataExitOperationNoVarPtr`'s
    `__init__`. Builder-only conversions; the parser only sees attribute
    instances."""
    acc_var = create_ssa_value(MemRefType(f32, [10]))
    op = acc.DeleteOp(
        acc_var=acc_var,
        data_clause=acc.DataClause.ACC_CREATE,
        structured=False,
        implicit=True,
        var_name="d",
    )
    op.verify()

    assert op.data_clause == acc.DataClauseAttr(acc.DataClause.ACC_CREATE)
    assert op.structured == IntegerAttr.from_bool(False)
    assert op.implicit == IntegerAttr.from_bool(True)
    assert op.var_name == StringAttr("d")


def test_cache_op_has_no_memory_effect_trait():
    """`acc.cache` is the only entry data-clause op that carries
    `NoMemoryEffect` (per upstream's td definition). Tested in pytest because
    a trait's presence on a class isn't observable via filecheck — it shapes
    what *transformations* are allowed, not how the op is printed."""
    assert acc.CacheOp.has_trait(NoMemoryEffect)
    # The other entry data-clause ops should *not* carry NoMemoryEffect —
    # upstream models them as touching runtime counters / device memory.
    assert not acc.CopyinOp.has_trait(NoMemoryEffect)
    assert not acc.AttachOp.has_trait(NoMemoryEffect)
    assert not acc.DeclareDeviceResidentOp.has_trait(NoMemoryEffect)


def test_private_recipe_builder_shortcuts():
    """The Python `__init__` accepts a `str` shortcut for `sym_name` and
    defaults a missing `destroy_region` to an empty `Region()`. Both
    branches are unreachable from filecheck — the parser only sees an
    already-built `StringAttr` and always supplies a (possibly empty)
    region — so they live here."""
    init_block = Block(arg_types=[i32])
    init_block.add_op(acc.YieldOp(init_block.args[0]))
    op = acc.PrivateRecipeOp(
        sym_name="priv",
        var_type=i32,
        init_region=Region([init_block]),
    )
    op.verify()

    assert op.sym_name == StringAttr("priv")
    assert op.var_type == i32
    assert len(op.init_region.blocks) == 1
    # `destroy_region=None` defaults to an empty Region, not absent.
    assert len(op.destroy_region.blocks) == 0


def test_reduction_recipe_builder_shortcuts():
    """The Python `__init__` accepts `str` for `sym_name` and a
    `ReductionOpKind` value for `reduction_operator`, defaulting a missing
    `destroy_region` to an empty `Region()`. All three branches are
    builder-only — the parser only sees an already-built `StringAttr`,
    a `ReductionOpKindAttr`, and (via the optional format clause) either
    a parsed region or no region argument at all — so they live here."""
    init_block = Block(arg_types=[i64])
    init_block.add_op(acc.YieldOp(init_block.args[0]))
    combiner_block = Block(arg_types=[i64, i64])
    combiner_block.add_op(acc.YieldOp(combiner_block.args[0]))
    op = acc.ReductionRecipeOp(
        sym_name="red",
        var_type=i64,
        reduction_operator=acc.ReductionOpKind.ADD,
        init_region=Region([init_block]),
        combiner_region=Region([combiner_block]),
    )
    op.verify()

    assert op.sym_name == StringAttr("red")
    assert op.var_type == i64
    assert op.reduction_operator == acc.ReductionOpKindAttr(acc.ReductionOpKind.ADD)
    assert len(op.init_region.blocks) == 1
    assert len(op.combiner_region.blocks) == 1
    assert len(op.destroy_region.blocks) == 0


def test_firstprivate_recipe_builder_shortcuts():
    """Same builder-only branches as `PrivateRecipeOp`'s test, but on the
    firstprivate constructor — the `__init__` is a separate method that
    must independently exercise the `str` → `StringAttr` shortcut and the
    `destroy_region=None` default."""
    init_block = Block(arg_types=[i32])
    init_block.add_op(acc.YieldOp(init_block.args[0]))
    copy_block = Block(arg_types=[i32, i32])
    copy_block.add_op(acc.YieldOp())
    op = acc.FirstprivateRecipeOp(
        sym_name="fp",
        var_type=i32,
        init_region=Region([init_block]),
        copy_region=Region([copy_block]),
    )
    op.verify()

    assert op.sym_name == StringAttr("fp")
    assert op.var_type == i32
    assert len(op.init_region.blocks) == 1
    assert len(op.copy_region.blocks) == 1
    assert len(op.destroy_region.blocks) == 0


def test_data_bounds_op_builder():
    """The `DataBoundsOp` Python builder isn't reached via filecheck (the
    parser uses IRDL `create`, which bypasses `__init__`) — pytest is the
    only place its `__init__` body runs."""
    lb = create_ssa_value(IndexType())
    ub = create_ssa_value(IndexType())
    extent = create_ssa_value(IndexType())
    stride = create_ssa_value(IndexType())
    start_idx = create_ssa_value(IndexType())
    op = acc.DataBoundsOp(
        lowerbound=lb,
        upperbound=ub,
        extent=extent,
        stride=stride,
        start_idx=start_idx,
        stride_in_bytes=IntegerAttr.from_bool(True),
    )
    op.verify()

    assert op.lowerbound is lb
    assert op.upperbound is ub
    assert op.extent is extent
    assert op.stride is stride
    assert op.start_idx is start_idx
    assert op.stride_in_bytes == IntegerAttr.from_bool(True)
    assert isinstance(op.result.type, acc.DataBoundsType)


def test_data_bounds_accessor_builders():
    """The shared `_DataBoundsAccessorOp.__init__` is exercised by all four
    accessor ops; constructing each from Python covers the single shared
    builder body that filecheck never invokes."""
    bounds = acc.DataBoundsOp(
        lowerbound=create_ssa_value(IndexType()),
        upperbound=create_ssa_value(IndexType()),
    )
    for cls in (
        acc.GetLowerboundOp,
        acc.GetUpperboundOp,
        acc.GetExtentOp,
        acc.GetStrideOp,
    ):
        op = cls(bounds.result)
        op.verify()
        assert op.bounds is bounds.result
        assert isinstance(op.result.type, IndexType)


def test_copyin_explicit_var_and_acc_var_type():
    """Both `var_type` and `acc_var_type` defaulting branches in
    `_DataEntryOperation.__init__` are skipped when callers pass them
    explicitly. The parser supplies these via the assembly format
    (`type($var)`, `$varType`), so the explicit-value branches are only
    reachable from a Python builder caller."""
    var = create_ssa_value(MemRefType(f32, [10]))
    explicit_var_type = f32
    explicit_acc_var_type = MemRefType(f32, [10])
    op = acc.CopyinOp(
        var=var,
        var_type=explicit_var_type,
        acc_var_type=explicit_acc_var_type,
    )
    op.verify()

    assert op.var_type is explicit_var_type
    assert op.acc_var.type is explicit_acc_var_type


def test_copyout_explicit_var_type():
    """`var_type=` shortcut on `_DataExitOperationWithVarPtr.__init__`
    bypasses the `_default_var_type` fallback. Builder-only branch — the
    parser always supplies `varType` explicitly via the assembly format."""
    explicit_var_type = f32
    op = acc.CopyoutOp(
        acc_var=create_ssa_value(MemRefType(f32, [10])),
        var=create_ssa_value(MemRefType(f32, [10])),
        var_type=explicit_var_type,
    )
    op.verify()

    assert op.var_type is explicit_var_type


def test_enter_data_init_bool_shortcuts():
    """The `async_attr` / `wait_attr` bool-shortcut branches in
    EnterDataOp.__init__ aren't reachable via the parser (which produces
    a UnitAttr directly via the custom directive)."""
    create = acc.CreateOp(var=create_ssa_value(MemRefType(f32, [10])))
    op = acc.EnterDataOp(
        data_clause_operands=[create],
        async_attr=True,
        wait_attr=True,
    )
    op.verify()

    assert isinstance(op.async_attr, UnitAttr)
    assert isinstance(op.wait_attr, UnitAttr)


def test_exit_data_init_bool_shortcuts():
    """Mirrors the EnterDataOp test for `ExitDataOp` plus its `finalize`
    bool shortcut. Same parser-bypass story: bare-keyword UnitAttrs come
    from the custom directive on parse, so the builder bool conversion
    is only exercised from Python."""
    devptr = acc.GetDevicePtrOp(var=create_ssa_value(MemRefType(f32, [10])))
    op = acc.ExitDataOp(
        data_clause_operands=[devptr],
        async_attr=True,
        wait_attr=True,
        finalize=True,
    )
    op.verify()

    assert isinstance(op.async_attr, UnitAttr)
    assert isinstance(op.wait_attr, UnitAttr)
    assert isinstance(op.finalize, UnitAttr)


def test_update_init():
    """Smoke test for `UpdateOp` — verifies operand wiring and the
    `if_present` bool-shortcut branch (only reachable from Python; on
    parse it lands as a UnitAttr in attr-dict)."""
    update_dev = acc.UpdateDeviceOp(var=create_ssa_value(MemRefType(f32, [10])))
    op = acc.UpdateOp(
        data_clause_operands=[update_dev],
        if_present=True,
    )
    op.verify()

    assert len(op.data_clause_operands) == 1
    assert op.data_clause_operands[0] is update_dev.acc_var
    assert isinstance(op.if_present, UnitAttr)
    assert op.if_cond is None
    assert len(op.async_operands) == 0
    assert len(op.wait_operands) == 0


def test_declare_family_init_bodies():
    """The three declare ops' `__init__` bodies are only reachable from
    Python — the parser builds via IRDL `Operation.create()` and bypasses
    them. Smoke-test all three to keep them covered. The non-trivial
    branches: `DeclareExitOp` packing the optional `token` into a 0/1-list
    (covered by both `token=...` and `token=None` calls), and `DeclareOp`
    accepting the region keyword-only."""
    var = create_ssa_value(MemRefType(f32, [10]))
    copyin = acc.CopyinOp(var=var)

    enter = acc.DeclareEnterOp(data_clause_operands=[copyin])
    assert isinstance(enter.token.type, acc.DeclareTokenType)

    exit_with_token = acc.DeclareExitOp(
        token=enter.token, data_clause_operands=[copyin]
    )
    assert exit_with_token.token is enter.token

    exit_no_token = acc.DeclareExitOp(data_clause_operands=[copyin])
    assert exit_no_token.token is None

    decl = acc.DeclareOp(region=Region(Block()), data_clause_operands=[copyin])
    assert len(decl.region.blocks) == 1


def _empty_loop(*, default_independent: bool = True) -> acc.LoopOp:
    """`acc.loop` with an empty single-block body holding just an `acc.yield`.

    Defaults `independent = [#acc.device_type<none>]` so the verifier's
    "at least one of auto/independent/seq" check is satisfied.
    """
    independent = (
        ArrayAttr([acc.DeviceTypeAttr(acc.DeviceType.NONE)])
        if default_independent
        else None
    )
    return acc.LoopOp(
        region=Region(Block([acc.YieldOp()])),
        independent=independent,
    )


def test_loop_par_mode_shortcut():
    """The `par_mode=` keyword argument fills in the matching seq /
    independent / auto array. This is a builder-only path: the parser only
    sees the already-built `ArrayAttr[DeviceTypeAttr]`, so the conversion
    branch is unreachable from filecheck."""
    op_seq = acc.LoopOp(
        region=Region(Block([acc.YieldOp()])),
        par_mode=acc.LoopParMode.SEQ,
    )
    op_seq.verify()
    assert op_seq.seq == ArrayAttr([acc.DeviceTypeAttr(acc.DeviceType.NONE)])
    assert op_seq.independent is None
    assert op_seq.auto_ is None

    op_auto = acc.LoopOp(
        region=Region(Block([acc.YieldOp()])),
        par_mode=acc.LoopParMode.AUTO,
    )
    op_auto.verify()
    assert op_auto.auto_ == ArrayAttr([acc.DeviceTypeAttr(acc.DeviceType.NONE)])
    assert op_auto.independent is None
    assert op_auto.seq is None

    op_indep = acc.LoopOp(
        region=Region(Block([acc.YieldOp()])),
        par_mode=acc.LoopParMode.INDEPENDENT,
    )
    op_indep.verify()
    assert op_indep.independent == ArrayAttr([acc.DeviceTypeAttr(acc.DeviceType.NONE)])
    assert op_indep.seq is None
    assert op_indep.auto_ is None

    # An explicit `seq=` argument wins over `par_mode=...`. Builder-only
    # branch; the parser would never pass both.
    nvidia = acc.DeviceTypeAttr(acc.DeviceType.NVIDIA)
    op_explicit = acc.LoopOp(
        region=Region(Block([acc.YieldOp()])),
        par_mode=acc.LoopParMode.SEQ,
        seq=ArrayAttr([nvidia]),
        independent=ArrayAttr([acc.DeviceTypeAttr(acc.DeviceType.NONE)]),
    )
    op_explicit.verify()
    assert op_explicit.seq == ArrayAttr([nvidia])


def test_loop_unit_and_combined_shortcuts():
    """`unstructured` accepts a bool shortcut; `combined` accepts a
    `CombinedConstructsType` value as well as the wrapped attribute."""
    op = acc.LoopOp(
        region=Region(Block([acc.YieldOp()])),
        independent=ArrayAttr([acc.DeviceTypeAttr(acc.DeviceType.NONE)]),
        unstructured=True,
        combined=acc.CombinedConstructsType.PARALLEL_LOOP,
    )
    op.verify()

    assert isinstance(op.unstructured, UnitAttr)
    assert op.combined == acc.CombinedConstructsTypeAttr(
        acc.CombinedConstructsType.PARALLEL_LOOP
    )

    op_off = _empty_loop()
    assert op_off.unstructured is None
    assert op_off.combined is None

    op_explicit = acc.LoopOp(
        region=Region(Block([acc.YieldOp()])),
        independent=ArrayAttr([acc.DeviceTypeAttr(acc.DeviceType.NONE)]),
        unstructured=UnitAttr(),
        combined=acc.CombinedConstructsTypeAttr(
            acc.CombinedConstructsType.KERNELS_LOOP
        ),
    )
    assert isinstance(op_explicit.unstructured, UnitAttr)
    assert op_explicit.combined == acc.CombinedConstructsTypeAttr(
        acc.CombinedConstructsType.KERNELS_LOOP
    )


def test_init_op_builder_branches():
    """Exercises the InitOp `__init__` paths that the parser bypasses
    (parsing goes through `Operation.create()`, not `__init__`): empty op
    plus a fully-populated op with both operands and the `device_types`
    property."""
    nvidia = acc.DeviceTypeAttr(acc.DeviceType.NVIDIA)

    empty = acc.InitOp()
    empty.verify()
    assert empty.device_num is None
    assert empty.if_cond is None
    assert empty.device_types is None

    op = acc.InitOp(
        device_num=create_ssa_value(i64),
        if_cond=create_ssa_value(i1),
        device_types=ArrayAttr([nvidia]),
    )
    op.verify()
    assert op.device_types == ArrayAttr([nvidia])


def test_shutdown_op_builder_branches():
    """Mirrors `test_init_op_builder_branches` for ShutdownOp."""
    default = acc.DeviceTypeAttr(acc.DeviceType.DEFAULT)

    empty = acc.ShutdownOp()
    empty.verify()
    assert empty.device_num is None
    assert empty.if_cond is None
    assert empty.device_types is None

    op = acc.ShutdownOp(
        device_num=create_ssa_value(i64),
        if_cond=create_ssa_value(i1),
        device_types=ArrayAttr([default]),
    )
    op.verify()
    assert op.device_types == ArrayAttr([default])


def test_set_op_init_smoke():
    """Filecheck builds via `Operation.create()` and never calls `SetOp.__init__`,
    leaving the constructor's `super().__init__(...)` line uncovered. A single
    Python construction hits it (the body is branchless)."""
    op = acc.SetOp(
        default_async=create_ssa_value(i32),
        device_num=create_ssa_value(i64),
        if_cond=create_ssa_value(i1),
        device_type=acc.DeviceTypeAttr(acc.DeviceType.NVIDIA),
    )
    op.verify()


def test_wait_op_init_smoke():
    """Mirrors `test_set_op_init_smoke` for WaitOp."""
    op = acc.WaitOp(
        wait_operands=[create_ssa_value(i64)],
        async_operand=create_ssa_value(i32),
        wait_devnum=create_ssa_value(i32),
        if_cond=create_ssa_value(i1),
    )
    op.verify()


def test_routine_op_builder_shortcuts():
    """The `RoutineOp.__init__` accepts `str` shortcuts for both `sym_name`
    (→ StringAttr) and `func_name` (→ SymbolRefAttr), plus `bool`
    shortcuts for the `nohost` / `implicit` UnitAttrs. None of these
    branches is reachable via the filecheck parser — it always supplies
    the wrapped attribute forms — so they're exercised here."""
    op = acc.RoutineOp(sym_name="r1", func_name="f1")
    op.verify()
    assert op.sym_name == StringAttr("r1")
    assert op.func_name == SymbolRefAttr("f1")
    assert op.nohost is None
    assert op.implicit is None
    assert op.bind_id_name is None
    assert op.bind_str_name is None
    assert op.bind_id_name_device_type is None
    assert op.bind_str_name_device_type is None
    assert op.gang is None
    assert op.gang_dim is None
    assert op.gang_dim_device_type is None
    assert op.worker is None
    assert op.vector is None
    assert op.seq is None

    op_with_bools = acc.RoutineOp(
        sym_name=StringAttr("r2"),
        func_name=SymbolRefAttr("f2"),
        nohost=True,
        implicit=True,
    )
    op_with_bools.verify()
    assert isinstance(op_with_bools.nohost, UnitAttr)
    assert isinstance(op_with_bools.implicit, UnitAttr)

    # `bool=False` and explicit `UnitAttr=None` must both produce a missing
    # property (covers the `(UnitAttr() if x else None)` branch + the
    # explicit-`UnitAttr` passthrough).
    op_explicit = acc.RoutineOp(
        sym_name="r3",
        func_name="f3",
        nohost=False,
        implicit=UnitAttr(),
    )
    assert op_explicit.nohost is None
    assert isinstance(op_explicit.implicit, UnitAttr)


def test_routine_op_full_population():
    """Drive every optional property through the constructor so each
    `properties=` slot has been written at least once. The filecheck
    parser builds these via `Operation.create()` (which bypasses
    `__init__`), so this test is the only place `RoutineOp.__init__`'s
    body runs to completion."""
    nvidia = acc.DeviceTypeAttr(acc.DeviceType.NVIDIA)
    radeon = acc.DeviceTypeAttr(acc.DeviceType.RADEON)
    host = acc.DeviceTypeAttr(acc.DeviceType.HOST)
    multicore = acc.DeviceTypeAttr(acc.DeviceType.MULTICORE)

    op = acc.RoutineOp(
        sym_name="rt",
        func_name="callee",
        bind_id_name=ArrayAttr([SymbolRefAttr("alt")]),
        bind_str_name=ArrayAttr([StringAttr("alt_str")]),
        bind_id_name_device_type=ArrayAttr([nvidia]),
        bind_str_name_device_type=ArrayAttr([radeon]),
        gang=ArrayAttr([nvidia]),
        gang_dim=ArrayAttr([IntegerAttr(1, i64)]),
        gang_dim_device_type=ArrayAttr([nvidia]),
        worker=ArrayAttr([radeon]),
        vector=ArrayAttr([host]),
        seq=ArrayAttr([multicore]),
    )
    op.verify()

    assert op.bind_id_name == ArrayAttr([SymbolRefAttr("alt")])
    assert op.bind_str_name == ArrayAttr([StringAttr("alt_str")])
    assert op.bind_id_name_device_type == ArrayAttr([nvidia])
    assert op.bind_str_name_device_type == ArrayAttr([radeon])
    assert op.gang == ArrayAttr([nvidia])
    assert op.gang_dim == ArrayAttr([IntegerAttr(1, i64)])
    assert op.gang_dim_device_type == ArrayAttr([nvidia])
    assert op.worker == ArrayAttr([radeon])
    assert op.vector == ArrayAttr([host])
    assert op.seq == ArrayAttr([multicore])


def test_global_ctor_dtor_builder_shortcuts():
    """`GlobalConstructorOp` / `GlobalDestructorOp` accept both a `str` and a
    `StringAttr` for `sym_name`. The filecheck parser bypasses `__init__`
    (it goes through IRDL's generic constructor), so the str → StringAttr
    branch and the StringAttr-passthrough branch are only exercised here."""
    ctor = acc.GlobalConstructorOp(
        sym_name="acc_constructor",
        region=Region(Block([acc.TerminatorOp()])),
    )
    assert ctor.sym_name == StringAttr("acc_constructor")

    dtor = acc.GlobalDestructorOp(
        sym_name=StringAttr("acc_destructor"),
        region=Region(Block([acc.TerminatorOp()])),
    )
    assert dtor.sym_name == StringAttr("acc_destructor")
