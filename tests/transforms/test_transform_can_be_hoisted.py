from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import builtin, func, scf, test
from xdsl.dialects.arith import Addf, Constant
from xdsl.dialects.builtin import (
    FloatAttr,
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

    lower = Constant.from_int_and_width(0, IndexType())
    upper = Constant.from_int_and_width(42, IndexType())
    step = Constant.from_int_and_width(3, IndexType())
    carried = Constant.from_int_and_width(1, IndexType())

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
        _f = Yield(op3)
    for_op = For(lower, upper, step, [carried], region)
    block0 = Block([a, b, for_op])
    _region_outer = Region(block0)

    for region in for_op.regions:
        for op in region.block.walk():
            if not isinstance(op, scf.Yield):
                assert can_be_hoisted(op3, region)


def test_invariant_loop_dialect():
    """Test for with loop-variant variables"""
    # Create constants
    ci0 = Constant.from_int_and_width(0, IndexType())
    cf7 = Constant(FloatAttr(7.0, f32))
    cf8 = Constant(FloatAttr(8.0, f32))
    ci10 = Constant.from_int_and_width(10, IndexType())
    ci1 = Constant.from_int_and_width(1, IndexType())
    co0 = Constant.from_int_and_width(0, IndexType())
    co1 = Constant.from_int_and_width(15, IndexType())
    coi = Constant.from_int_and_width(1, IndexType())

    @Builder.implicit_region((builtin.IndexType(),))
    def inner_loop_body(args: tuple[BlockArgument, ...]):
        bla = Constant.from_int_and_width(5, i32)
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
    # Assertions to check loop invariant and hoisting
    for op in mod.body.walk():
        for region in mod.regions:
            if isinstance(op, test.TestOp):
                assert not can_be_hoisted(op, mod.body)
            elif isinstance(op, Addf):
                region = op.parent_region()
                assert region is not None
                assert can_be_hoisted(op, region)
