from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, scf, test
from xdsl.dialects.arith import Addf
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

    ci0 = test.TestOp(result_types=[i32])
    cf7 = test.TestOp(result_types=[f32])
    cf8 = test.TestOp(result_types=[f32])
    ci10 = test.TestOp(result_types=[i32])
    ci1 = test.TestOp(result_types=[i32])
    co0 = test.TestOp(result_types=[i32])
    co1 = test.TestOp(result_types=[i32])
    coi = test.TestOp(result_types=[i32])
    block = Block(
        arg_types=(
            index,
            index,
        )
    )
    block1 = Block(
        arg_types=(
            index,
            index,
        )
    )
    region_inner = Region(block)
    with ImplicitBuilder(block) as (_arg0, _):
        op1 = test.TestOp(result_types=[i32])
        # Test operation inside the loop
        hello = test.TestOp([op1], result_types=[i32])
        test.TestOp([hello], result_types=[])
        # Floating-point addition inside the loop
        v0 = Addf(cf7, cf8)

    region = Region(block1)
    with ImplicitBuilder(block1) as (_arg1, _):
        scf.For(ci0, ci10, ci1, [], region_inner)

    for_op = For(co0, co1, coi, [], region)
    body0 = Block([cf7, cf8, for_op])
    region_outer = Region(body0)
    func_type = FunctionType.from_lists([], [])
    function = func.FuncOp("invariant_loop_dialect", func_type, region_outer)

    # Wrap all in a ModuleOp
    _mod = ModuleOp([function])
    assert can_be_hoisted(op1, region)
    assert not can_be_hoisted(hello, region)
    assert can_be_hoisted(cf7, region_outer)
    assert can_be_hoisted(cf8, region_outer)
    assert can_be_hoisted(v0, region)
