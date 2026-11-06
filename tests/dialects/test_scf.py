"""
Test the usage of scf dialect.
"""

import re
from typing import cast

import pytest
from typing_extensions import TypeVar

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects.arith import Arith, ConstantOp
from xdsl.dialects.builtin import (
    Builtin,
    IndexType,
    IntegerAttr,
    ModuleOp,
    i1,
    i32,
    i64,
)
from xdsl.dialects.scf import (
    ConditionOp,
    ExecuteRegionOp,
    ForOp,
    IfOp,
    IndexSwitchOp,
    ParallelOp,
    ReduceOp,
    ReduceReturnOp,
    Scf,
    WhileOp,
    YieldOp,
)
from xdsl.dialects.test import Test, TestTermOp
from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue
from xdsl.parser import Parser
from xdsl.traits import (
    InvocationBounds,
    RegionBranchOpTrait,
    RegionBranchPoint,
    RegionBranchTerminatorOpTrait,
    RegionSuccessor,
)
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
        match=re.escape("scf.reduce block must have exactly two arguments, but "),
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp(
            (init_val,), (Region(Block([rro], arg_types=[i32, i32, i32])),)
        ).verify()

    with pytest.raises(
        VerifyException,
        match=re.escape("scf.reduce block must have exactly two arguments, but "),
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp((init_val,), (Region(Block([rro], arg_types=[i32])),)).verify()

    with pytest.raises(
        VerifyException,
        match=re.escape("scf.reduce block must have exactly two arguments, but "),
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp((init_val,), (Region(Block([rro], arg_types=[])),)).verify()


def test_reduce_op_num_block_arg_types():
    init_val = ConstantOp.from_int_and_width(10, i32)
    reduce_constant = ConstantOp.from_int_and_width(100, i32)

    with pytest.raises(
        VerifyException,
        match=re.escape("scf.reduce block argument types must be the same but have"),
    ):
        rro = ReduceReturnOp(reduce_constant)
        ReduceOp((init_val,), (Region(Block([rro], arg_types=[i32, i64])),)).verify()

    with pytest.raises(
        VerifyException,
        match=re.escape("scf.reduce block argument types must be the same but have"),
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
        match=re.escape(
            "'scf.reduce' terminates with operation test.termop instead of scf.reduce.return"
        ),
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
        match=re.escape(
            "scf.reduce.return result type at end of scf.reduce block must"
        ),
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


# RegionBranchOpInterface implementations, ported from MLIR's `SCF.cpp`.


def _parse(ir: str) -> ModuleOp:
    ctx = Context()
    for dialect in (Builtin, Arith, Scf, Test):
        ctx.load_dialect(dialect)
    module = Parser(ctx, ir).parse_module()
    module.verify()
    return module


_OpT = TypeVar("_OpT", bound=Operation)


def _find(module: ModuleOp, op_type: type[_OpT]) -> _OpT:
    return next(op for op in module.walk() if isinstance(op, op_type))


def _name_to_ssa(module: ModuleOp) -> dict[str, SSAValue]:
    """Map the name hint of every named result and block argument to its value."""
    values: dict[str, SSAValue] = {}
    for op in module.walk():
        values |= {res.name_hint: res for res in op.results if res.name_hint}
        for region in op.regions:
            for block in region.blocks:
                values |= {arg.name_hint: arg for arg in block.args if arg.name_hint}
    return values


def _trait(op: Operation) -> RegionBranchOpTrait:
    trait = op.get_trait(RegionBranchOpTrait)
    assert trait is not None
    return trait


def _successors(op: Operation, point: RegionBranchPoint) -> list[RegionSuccessor]:
    return list(_trait(op).successor_regions(op, point))


_TRUE = IntegerAttr(1, i1)
_FALSE = IntegerAttr(0, i1)


def test_for_region_branch():
    module = _parse("""
    %lb, %ub, %step, %init = "test.op"() : () -> (index, index, index, i32)
    %res = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (i32) {
      scf.yield %acc : i32
    }
    """)
    for_op, yield_op = _find(module, ForOp), _find(module, YieldOp)
    ssa = _name_to_ssa(module)
    body, exit_ = RegionSuccessor(for_op.body), RegionSuccessor(for_op)
    assert _successors(for_op, RegionBranchPoint.PARENT) == [body, exit_]
    assert _successors(for_op, RegionBranchPoint(yield_op)) == [body, exit_]
    entry_operands = for_op.entry_successor_operands(body)
    assert list(entry_operands) == [ssa["init"]]
    assert entry_operands.op_operands == ((for_op, 3),)
    assert list(for_op.successor_inputs(body)) == [ssa["acc"]]
    assert list(for_op.successor_inputs(exit_)) == [ssa["res"]]
    assert _trait(for_op).has_loop(for_op)
    assert _trait(for_op).is_repetitive_region(for_op, 0)


@pytest.mark.parametrize(
    ("bounds", "trip_count"),
    [
        ("%lb = arith.constant 3 : index\n%ub = arith.constant 3 : index", 0),
        ("%lb = arith.constant 5 : index\n%ub = arith.constant 2 : index", 0),
        ("%lb = arith.constant 0 : index\n%ub = arith.constant 1 : index", 1),
        ("%lb = arith.constant 0 : index\n%ub = arith.constant 10 : index", 4),
        ('%lb = arith.constant 0 : index\n%ub = "test.op"() : () -> index', None),
        ('%lb = "test.op"() : () -> index\n%ub = arith.constant 10 : index', None),
        # ub = lb + 6 without signed wrap: (6 / 3) = 2 iterations.
        (
            '%lb = "test.op"() : () -> index\n%c6 = arith.constant 6 : index\n'
            "%ub = arith.addi %lb, %c6 overflow<nsw> : index",
            2,
        ),
        # Without the nsw flag, the difference is not trusted.
        (
            '%lb = "test.op"() : () -> index\n%c6 = arith.constant 6 : index\n'
            "%ub = arith.addi %lb, %c6 : index",
            None,
        ),
        # ub = lb - 6 without signed wrap: zero iterations.
        (
            '%lb = "test.op"() : () -> index\n%c6 = arith.constant -6 : index\n'
            "%ub = arith.addi %lb, %c6 overflow<nsw> : index",
            0,
        ),
    ],
)
def test_for_static_trip_count(bounds: str, trip_count: int | None):
    module = _parse(f"""
    {bounds}
    %step = arith.constant 3 : index
    scf.for %iv = %lb to %ub step %step {{
    }}
    """)
    for_op, yield_op = _find(module, ForOp), _find(module, YieldOp)
    body, exit_ = RegionSuccessor(for_op.body), RegionSuccessor(for_op)
    from_parent = _successors(for_op, RegionBranchPoint.PARENT)
    from_yield = _successors(for_op, RegionBranchPoint(yield_op))
    if trip_count is None:
        assert (from_parent, from_yield) == ([body, exit_], [body, exit_])
    elif trip_count == 0:
        assert (from_parent, from_yield) == ([exit_], [body, exit_])
    elif trip_count == 1:
        assert (from_parent, from_yield) == ([body], [exit_])
    else:
        assert (from_parent, from_yield) == ([body], [body, exit_])


def test_for_static_trip_count_same_bounds():
    # The same non-constant value as both bounds: zero iterations.
    module = _parse("""
    %bound, %step = "test.op"() : () -> (index, index)
    scf.for %iv = %bound to %bound step %step {
    }
    """)
    for_op = _find(module, ForOp)
    exit_ = RegionSuccessor(for_op)
    assert _successors(for_op, RegionBranchPoint.PARENT) == [exit_]


def test_for_static_trip_count_step_as_upper_bound():
    # A zero lower bound and the step as upper bound: one iteration.
    module = _parse("""
    %lb = arith.constant 0 : index
    %step = "test.op"() : () -> index
    scf.for %iv = %lb to %step step %step {
    }
    """)
    for_op, yield_op = _find(module, ForOp), _find(module, YieldOp)
    body, exit_ = RegionSuccessor(for_op.body), RegionSuccessor(for_op)
    assert _successors(for_op, RegionBranchPoint.PARENT) == [body]
    assert _successors(for_op, RegionBranchPoint(yield_op)) == [exit_]


@pytest.mark.parametrize("step", ["0", "-1"])
def test_for_static_trip_count_unknown_step(step: str):
    module = _parse(f"""
    %lb = arith.constant 0 : i32
    %ub = arith.constant 10 : i32
    %step = arith.constant {step} : i32
    scf.for %iv = %lb to %ub step %step : i32 {{
    }}
    """)
    for_op = _find(module, ForOp)
    body, exit_ = RegionSuccessor(for_op.body), RegionSuccessor(for_op)
    assert _successors(for_op, RegionBranchPoint.PARENT) == [body, exit_]


def test_for_successor_mapping_with_repeated_operand():
    # `%c0` is both the lower bound and the initial value of `%acc`.
    module = _parse("""
    %c0 = arith.constant 0 : index
    %n, %c1 = "test.op"() : () -> (index, index)
    %res = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %c0) -> (index) {
      scf.yield %acc : index
    }
    """)
    for_op, yield_op = _find(module, ForOp), _find(module, YieldOp)
    ssa = _name_to_ssa(module)
    mapping = _trait(for_op).successor_operand_input_mapping(for_op)
    assert mapping == {
        (for_op, 3): (ssa["acc"], ssa["res"]),
        (yield_op, 0): (ssa["acc"], ssa["res"]),
    }


def test_if_region_branch():
    module = _parse("""
    %cond = "test.op"() : () -> i1
    %res = scf.if %cond -> (i32) {
      %a = "test.op"() : () -> i32
      scf.yield %a : i32
    } else {
      %b = "test.op"() : () -> i32
      scf.yield %b : i32
    }
    """)
    if_op = _find(module, IfOp)
    then_yield = if_op.true_region.block.last_op
    assert isinstance(then_yield, YieldOp)
    then, else_ = (
        RegionSuccessor(if_op.true_region),
        RegionSuccessor(if_op.false_region),
    )
    exit_ = RegionSuccessor(if_op)
    assert _successors(if_op, RegionBranchPoint.PARENT) == [then, else_]
    assert _successors(if_op, RegionBranchPoint(then_yield)) == [exit_]
    assert list(if_op.successor_inputs(exit_)) == list(if_op.results)
    assert list(if_op.successor_inputs(then)) == []
    assert list(if_op.entry_successor_regions([None])) == [then, else_]
    assert list(if_op.entry_successor_regions([_TRUE])) == [then]
    assert list(if_op.entry_successor_regions([_FALSE])) == [else_]
    assert list(if_op.region_invocation_bounds([None])) == [InvocationBounds(0, 1)] * 2
    assert list(if_op.region_invocation_bounds([_TRUE])) == [
        InvocationBounds(0, 1),
        InvocationBounds(0, 0),
    ]
    assert list(if_op.region_invocation_bounds([_FALSE])) == [
        InvocationBounds(0, 0),
        InvocationBounds(0, 1),
    ]
    assert not _trait(if_op).has_loop(if_op)


def test_if_without_else_region_branch():
    module = _parse("""
    %cond = "test.op"() : () -> i1
    scf.if %cond {
    }
    """)
    if_op = _find(module, IfOp)
    then, exit_ = RegionSuccessor(if_op.true_region), RegionSuccessor(if_op)
    assert _successors(if_op, RegionBranchPoint.PARENT) == [then, exit_]
    assert list(if_op.entry_successor_regions([None])) == [then, exit_]
    assert list(if_op.entry_successor_regions([_TRUE])) == [then]
    assert list(if_op.entry_successor_regions([_FALSE])) == [exit_]


def test_while_region_branch():
    module = _parse("""
    %init = "test.op"() : () -> i32
    %res = scf.while (%arg = %init) : (i32) -> i32 {
      %cond = "test.op"(%arg) : (i32) -> i1
      scf.condition(%cond) %arg : i32
    } do {
    ^bb0(%body_arg : i32):
      scf.yield %body_arg : i32
    }
    """)
    while_op = _find(module, WhileOp)
    condition, yield_op = _find(module, ConditionOp), _find(module, YieldOp)
    ssa = _name_to_ssa(module)
    before = RegionSuccessor(while_op.before_region)
    after = RegionSuccessor(while_op.after_region)
    exit_ = RegionSuccessor(while_op)
    assert _successors(while_op, RegionBranchPoint.PARENT) == [before]
    assert _successors(while_op, RegionBranchPoint(condition)) == [exit_, after]
    assert _successors(while_op, RegionBranchPoint(yield_op)) == [before]
    assert list(while_op.entry_successor_operands(before)) == [ssa["init"]]
    assert list(while_op.successor_inputs(before)) == [ssa["arg"]]
    assert list(while_op.successor_inputs(after)) == [ssa["body_arg"]]
    assert list(while_op.successor_inputs(exit_)) == [ssa["res"]]
    condition_operands = condition.successor_operands(after)
    assert list(condition_operands) == [ssa["arg"]]
    assert condition_operands.op_operands == ((condition, 1),)
    assert list(condition.successor_regions([None, None])) == [after, exit_]
    assert list(condition.successor_regions([_TRUE, None])) == [after]
    assert list(condition.successor_regions([_FALSE, None])) == [exit_]
    assert _trait(while_op).has_loop(while_op)
    assert _trait(while_op).is_repetitive_region(while_op, 0)
    assert _trait(while_op).is_repetitive_region(while_op, 1)


def test_while_successor_mapping_with_repeated_operand():
    # `%cond` is both the condition and the forwarded value.
    module = _parse("""
    %init = "test.op"() : () -> i1
    %res = scf.while (%arg = %init) : (i1) -> i1 {
      scf.condition(%arg) %arg : i1
    } do {
    ^bb0(%body_arg : i1):
      scf.yield %body_arg : i1
    }
    """)
    while_op, condition = _find(module, WhileOp), _find(module, ConditionOp)
    ssa = _name_to_ssa(module)
    mapping = _trait(while_op).successor_operand_input_mapping(
        while_op, RegionBranchPoint(condition)
    )
    assert mapping == {(condition, 1): (ssa["res"], ssa["body_arg"])}


def test_execute_region_region_branch():
    module = _parse("""
    %res = scf.execute_region -> (i32) {
      %a = "test.op"() : () -> i32
      scf.yield %a : i32
    }
    """)
    op, yield_op = _find(module, ExecuteRegionOp), _find(module, YieldOp)
    body, exit_ = RegionSuccessor(op.region), RegionSuccessor(op)
    assert _successors(op, RegionBranchPoint.PARENT) == [body]
    assert _successors(op, RegionBranchPoint(yield_op)) == [exit_]
    assert list(op.region_invocation_bounds([])) == [InvocationBounds(1, 1)]
    assert list(op.successor_inputs(exit_)) == list(op.results)
    assert list(op.successor_inputs(body)) == []
    assert not _trait(op).has_loop(op)


def test_index_switch_region_branch():
    module = _parse("""
    %arg = "test.op"() : () -> index
    %res = scf.index_switch %arg -> i32
    case 2 {
      %a = "test.op"() : () -> i32
      scf.yield %a : i32
    }
    case 5 {
      %b = "test.op"() : () -> i32
      scf.yield %b : i32
    }
    default {
      %c = "test.op"() : () -> i32
      scf.yield %c : i32
    }
    """)
    op = _find(module, IndexSwitchOp)
    default, case2, case5 = (RegionSuccessor(region) for region in op.regions)
    exit_ = RegionSuccessor(op)
    case_yield = op.case_regions[0].block.last_op
    assert isinstance(case_yield, YieldOp)
    assert _successors(op, RegionBranchPoint.PARENT) == [default, case2, case5]
    assert _successors(op, RegionBranchPoint(case_yield)) == [exit_]
    assert list(op.successor_inputs(exit_)) == list(op.results)
    assert list(op.successor_inputs(case2)) == []
    index = IndexType()
    assert list(op.entry_successor_regions([None])) == [default, case2, case5]
    assert list(op.entry_successor_regions([IntegerAttr(5, index)])) == [case5]
    assert list(op.entry_successor_regions([IntegerAttr(7, index)])) == [default]
    assert list(op.region_invocation_bounds([None])) == [InvocationBounds(0, 1)] * 3
    # The default region comes first, so case `k` is region `k + 1`.
    assert list(op.region_invocation_bounds([IntegerAttr(2, index)])) == [
        InvocationBounds(0, 0),
        InvocationBounds(0, 1),
        InvocationBounds(0, 0),
    ]
    assert list(op.region_invocation_bounds([IntegerAttr(5, index)])) == [
        InvocationBounds(0, 0),
        InvocationBounds(0, 0),
        InvocationBounds(0, 1),
    ]
    assert list(op.region_invocation_bounds([IntegerAttr(7, index)])) == [
        InvocationBounds(0, 1),
        InvocationBounds(0, 0),
        InvocationBounds(0, 0),
    ]


def test_parallel_region_branch():
    module = _parse("""
    %lb, %ub, %step, %init = "test.op"() : () -> (index, index, index, i32)
    %res = "scf.parallel"(%lb, %ub, %step, %init) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1>
    }> ({
    ^bb0(%i : index):
      %v = "test.op"() : () -> i32
      scf.reduce(%v : i32) {
      ^bb1(%x : i32, %y : i32):
        scf.reduce.return %x : i32
      }
    }) : (index, index, index, i32) -> i32
    """)
    op, reduce_op = _find(module, ParallelOp), _find(module, ReduceOp)
    body, exit_ = RegionSuccessor(op.body), RegionSuccessor(op)
    assert _successors(op, RegionBranchPoint.PARENT) == [body, exit_]
    assert _successors(op, RegionBranchPoint(reduce_op)) == [body, exit_]
    assert list(op.entry_successor_operands(body)) == []
    assert list(reduce_op.successor_operands(exit_)) == []
    terminator_trait = reduce_op.get_trait(RegionBranchTerminatorOpTrait)
    assert terminator_trait is not None
    assert list(terminator_trait.successor_regions(reduce_op, [None])) == [
        body,
        exit_,
    ]
