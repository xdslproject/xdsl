from xdsl.builder import Builder
from xdsl.dialects import builtin, func, scf, test
from xdsl.dialects.arith import Addf, Addi, Constant, DivUI, Muli
from xdsl.dialects.builtin import (
    FloatAttr,
    FunctionType,
    IndexType,
    IntegerAttr,
    ModuleOp,
    f32,
    i32,
)
from xdsl.dialects.scf import For, Yield
from xdsl.ir import Block, BlockArgument, Region
from xdsl.transforms.loop_invariant_code_motion import can_be_hoisted

index = IndexType()


def test_is_loop_dependent_no_dep():
    lb = Constant.from_int_and_width(0, i32)
    ub = Constant.from_int_and_width(42, i32)
    step = Constant.from_int_and_width(3, i32)

    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])

    for_op = For(lb, ub, step, [], Block([op1, op2], arg_types=[i32]))

    assert can_be_hoisted(op1, for_op.body)
    assert can_be_hoisted(op2, for_op.body)


def test_for_with_loop_invariant_verify():
    """Test for with loop-variant variables"""

    lower = Constant.from_int_and_width(0, IndexType())
    upper = Constant.from_int_and_width(42, IndexType())
    step = Constant.from_int_and_width(3, IndexType())
    carried = Constant.from_int_and_width(1, IndexType())

    @Builder.implicit_region((IndexType(), IndexType()))
    def body(_: tuple[BlockArgument, ...]) -> None:
        a = Constant(IntegerAttr.from_int_and_width(1, 32), i32)
        b = Constant(IntegerAttr.from_int_and_width(2, 32), i32)
        # Operations on these constants
        c = Addi(a, b)
        d = Muli(a, b)
        e = DivUI(c, d)
        Yield(e)

    for_op = For(lower, upper, step, [carried], body)

    for op in for_op.body.walk():
        for regoin in for_op.regions:
            # MLIR iterates the for loop's body and hoists operations that
            # can_be_hoisted. However, the DivUI operation cannot be hoisted immediately
            # because its operands are still within the loop's body. It can only be
            # hoisted once the instructions generating those operands have been hoisted.
            if not (isinstance(op, Yield) or isinstance(op, DivUI)):
                assert can_be_hoisted(op, regoin)


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
    # @Builder.implicit_region((IndexType(),))
    def outer_loop_body(args: tuple[BlockArgument, ...]):
        scf.For(ci0, ci10, ci1, [], inner_loop_body)

    for_op = For(co0, co1, coi, [], outer_loop_body)
    body0 = Block([cf7, cf8, for_op])
    region0 = Region(body0)
    # regoin1 = Region([body0, outer_loop_body.block])
    # # Create the function
    func_type = FunctionType.from_lists([], [])
    function = func.FuncOp("invariant_loop_dialect", func_type, region0)

    # Wrap all in a ModuleOp
    mod = ModuleOp([function])
    # Assertions to check loop invariant and hoisting
    for op in mod.body.walk():
        for regoin in mod.regions:
            if isinstance(op, test.TestOp):
                assert not can_be_hoisted(op, mod.body)
            elif isinstance(op, Addf):
                regoin = op.parent_region()
                assert regoin is not None
                assert can_be_hoisted(op, regoin)


def test_for_with_loop_invariant_verify1():
    """Test for with loop-invariant variables"""
    lb = Constant.from_int_and_width(0, i32)
    ub = Constant.from_int_and_width(42, i32)
    step = Constant.from_int_and_width(3, i32)

    op1 = test.TestOp(result_types=[i32])
    op2 = test.TestOp(result_types=[i32])
    op3 = test.TestOp([op2, op1], result_types=[i32])
    op4 = test.TestOp([op3, op2], result_types=[i32])
    bb0 = Block([op1, op2], arg_types=[i32])
    bb1 = Block([op3, op4], arg_types=[i32])

    for_op = For(lb, ub, step, [], bb1)
    bb3 = Block([for_op])
    regions = Region([bb0, bb3])

    # # Create the function
    func_type = FunctionType.from_lists([], [])
    function = func.FuncOp("invariant_loop_dialect", func_type, regions)

    for region in function.regions:
        assert can_be_hoisted(op3, region)
        # MLIR iterates the for loop's body and hoists operations that can_be_hoisted.
        # However, the DivUI operation cannot be hoisted immediately because its
        # operands are still within the loop's body. It can only be hoisted once the
        # instructions generating those operands have been hoisted.
        assert not can_be_hoisted(op4, region)
