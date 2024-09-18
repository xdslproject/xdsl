from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, test
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    ModuleOp,
    f32,
    i32,
)
from xdsl.dialects.scf import For, Yield
from xdsl.ir import Block, Region
from xdsl.transforms.loop_invariant_code_motion import can_be_hoisted

index = IndexType()


def test_for_with_loop_invariant_verify():
    """Test for with loop-variant variables"""

    lower = test.TestOp(result_types=[i32])
    upper = test.TestOp(result_types=[i32])
    step = test.TestOp(result_types=[i32])
    carried = test.TestOp(result_types=[i32])

    a = test.TestOp(result_types=[i32])
    b = test.TestOp(result_types=[i32])
    block = Block(
        arg_types=(
            index,
            index,
        )
    )
    region = Region(block)
    with ImplicitBuilder(block) as (arg0, _):
        op1 = test.TestOp(result_types=[i32])
        op2 = test.TestOp(result_types=[i32])
        op3 = test.TestOp([op1, op2, arg0], result_types=[i32])
        f = Yield(op3)
    for_op = For(lower, upper, step, [carried], region)
    block0 = Block([a, b, for_op])
    _region_outer = Region(block0)

    assert can_be_hoisted(op1, region)
    assert can_be_hoisted(op2, region)
    assert not can_be_hoisted(op3, region)
    assert not can_be_hoisted(f, region)


def test_invariant_loop_dialect():
    """Test for with loop-variant variables"""
    # Create constants
    outer_region = Region([Block()])
    with ImplicitBuilder(outer_region):
        ci32, cf32 = test.TestOp(result_types=[i32, f32]).results
        outer_for = For(ci32, ci32, ci32, [], Region(Block(arg_types=[index, index])))
        with ImplicitBuilder(outer_for.body) as (_outer_i, _):
            inner_for = For(
                ci32, ci32, ci32, [], Region(Block(arg_types=[index, index]))
            )
            with ImplicitBuilder(inner_for.body) as (_inner_i, _):
                # Self contained Op
                no_args = test.TestOp(result_types=[i32])
                # Op with external dependency
                dep_on_outer = test.TestOp([ci32], result_types=[i32])
                args_in_inner = test.TestOp([no_args], result_types=[i32])
                no_results = test.TestOp([args_in_inner], result_types=[])
    func_type = FunctionType.from_lists([], [])
    function = func.FuncOp("invariant_loop_dialect", func_type, outer_region)
    # Wrap all in a ModuleOp
    _mod = ModuleOp([function])

    assert can_be_hoisted(no_args, inner_for.body)
    assert not can_be_hoisted(args_in_inner, inner_for.body)
    assert not can_be_hoisted(no_results, inner_for.body)
    assert can_be_hoisted(dep_on_outer, inner_for.body)
    assert can_be_hoisted(ci32.op, outer_for.body)
    assert can_be_hoisted(cf32.op, outer_for.body)
