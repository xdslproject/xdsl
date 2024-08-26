from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import builtin, func, scf, test
from xdsl.dialects.arith import Addf
from xdsl.dialects.builtin import (
    FunctionType,
    IndexType,
    ModuleOp,
    f32,
    i32,
)
from xdsl.dialects.scf import For, Yield
from xdsl.ir import Block, BlockArgument, Region
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
    assert can_be_hoisted(op3, region)
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

    @Builder.implicit_region((builtin.IndexType(),))
    def inner_loop_body(args: tuple[BlockArgument, ...]):
        bla = test.TestOp(result_types=[i32])
        # Test operation inside the loop
        hello = test.TestOp([bla], result_types=[i32])
        test.TestOp([hello], result_types=[])
        # Floating-point addition inside the loop
        _v0 = Addf(cf7, cf8)

    @Builder.implicit_region((builtin.IndexType(),))
    def outer_loop_body(args: tuple[BlockArgument, ...]):
        scf.For(ci0, ci10, ci1, [], inner_loop_body)

    for_op = For(co0, co1, coi, [], outer_loop_body)
    body0 = Block([cf7, cf8, for_op])
    region0 = Region(body0)
    func_type = FunctionType.from_lists([], [])
    function = func.FuncOp("invariant_loop_dialect", func_type, region0)

    # Wrap all in a ModuleOp
    mod = ModuleOp([function])
    # Inner most for loop
    if any(isinstance(ha, scf.For) for ha in mod.body.walk()):
        return
    # Assertions to check loop invariant and hoisting
    for op in mod.body.walk():
        if isinstance(op, test.TestOp):
            assert not can_be_hoisted(op, mod.body)
        elif isinstance(op, Addf):
            region = op.parent_region()
            assert region is not None
            assert can_be_hoisted(op, region)
