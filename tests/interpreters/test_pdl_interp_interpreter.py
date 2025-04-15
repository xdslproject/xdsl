from typing import cast

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import pdl, pdl_interp, test
from xdsl.dialects.builtin import (
    FunctionType,
    ModuleOp,
    StringAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter, Successor
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, BlockArgument
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.test_value import create_ssa_value


def test_getters():
    # TODO: test negative cases returning None
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})
    op_res = op.results[0]

    assert interpreter.run_op(
        pdl_interp.GetOperandOp(1, create_ssa_value(pdl.OperationType())), (op,)
    ) == (c1,)

    assert interpreter.run_op(
        pdl_interp.GetResultOp(1, create_ssa_value(pdl.OperationType())), (op,)
    ) == (op.results[1],)

    assert (
        interpreter.run_op(
            pdl_interp.GetResultsOp(
                None,
                create_ssa_value(pdl.OperationType()),
                pdl.RangeType(pdl.ValueType()),
            ),
            (op,),
        )[0]
        == op.results
    )

    assert interpreter.run_op(
        pdl_interp.GetAttributeOp("myattr", create_ssa_value(pdl.OperationType())),
        (op,),
    ) == (myattr,)

    assert interpreter.run_op(
        pdl_interp.GetValueTypeOp(create_ssa_value(pdl.ValueType())), (c0,)
    ) == (i32,)

    assert interpreter.run_op(
        pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType())), (op_res,)
    ) == (op,)


def test_check_operation_name():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})

    trueresult = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperationNameOp(
            "test.op", create_ssa_value(pdl.OperationType()), truedest, falsedest
        ),
        (op,),
    )
    assert isinstance(trueresult.terminator_value, Successor)
    assert isinstance(trueresult.terminator_value.block, Block)
    assert trueresult.terminator_value.block == truedest

    falseresult = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperationNameOp(
            "test.other", create_ssa_value(pdl.OperationType()), truedest, falsedest
        ),
        (op,),
    )
    assert isinstance(falseresult.terminator_value, Successor)
    assert isinstance(falseresult.terminator_value.block, Block)
    assert falseresult.terminator_value.block == falsedest


def test_check_operand_count():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})

    # Test exact operand count
    exact_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperandCountOp(
            create_ssa_value(pdl.OperationType()),
            2,  # op has exactly 2 operands (c0, c1)
            truedest,
            falsedest,
            compareAtLeast=False,
        ),
        (op,),
    )
    assert isinstance(exact_result.terminator_value, Successor)
    assert exact_result.terminator_value.block == truedest

    # Test compareAtLeast=True
    at_least_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperandCountOp(
            create_ssa_value(pdl.OperationType()),
            1,  # op has 2 operands which is >= 1
            truedest,
            falsedest,
            compareAtLeast=True,
        ),
        (op,),
    )
    assert isinstance(at_least_result.terminator_value, Successor)
    assert at_least_result.terminator_value.block == truedest

    # Test failing cases
    fail_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckOperandCountOp(
            create_ssa_value(pdl.OperationType()),
            3,  # op has only 2 operands
            truedest,
            falsedest,
            compareAtLeast=False,
        ),
        (op,),
    )
    assert isinstance(fail_result.terminator_value, Successor)
    assert fail_result.terminator_value.block == falsedest


def test_check_result_count():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    myattr = StringAttr("hello")
    op = test.TestOp((c0, c1), (i32, i64), {"myattr": myattr})  # Has 2 results

    # Test exact result count
    exact_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckResultCountOp(
            create_ssa_value(pdl.OperationType()),
            2,  # op has exactly 2 results
            truedest,
            falsedest,
            compareAtLeast=False,
        ),
        (op,),
    )
    assert isinstance(exact_result.terminator_value, Successor)
    assert exact_result.terminator_value.block == truedest

    # Test compareAtLeast=True
    at_least_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckResultCountOp(
            create_ssa_value(pdl.OperationType()),
            1,  # op has 2 results which is >= 1
            truedest,
            falsedest,
            compareAtLeast=True,
        ),
        (op,),
    )
    assert isinstance(at_least_result.terminator_value, Successor)
    assert at_least_result.terminator_value.block == truedest

    # Test failing case
    fail_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckResultCountOp(
            create_ssa_value(pdl.OperationType()),
            3,  # op has only 2 results
            truedest,
            falsedest,
            compareAtLeast=False,
        ),
        (op,),
    )
    assert isinstance(fail_result.terminator_value, Successor)
    assert fail_result.terminator_value.block == falsedest


def test_check_attribute():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    # Test matching attribute
    match_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckAttributeOp(
            StringAttr("hello"),  # Expected value
            create_ssa_value(pdl.AttributeType()),  # Input attribute
            truedest,
            falsedest,
        ),
        (StringAttr("hello"),),  # Actual value that matches
    )
    assert isinstance(match_result.terminator_value, Successor)
    assert match_result.terminator_value.block == truedest

    # Test non-matching attribute
    nomatch_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.CheckAttributeOp(
            StringAttr("hello"),  # Expected value
            create_ssa_value(pdl.AttributeType()),  # Input attribute
            truedest,
            falsedest,
        ),
        (StringAttr("world"),),  # Different value
    )
    assert isinstance(nomatch_result.terminator_value, Successor)
    assert nomatch_result.terminator_value.block == falsedest


def test_is_not_null():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)

    # Test with non-null value
    notnull_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.IsNotNullOp(
            create_ssa_value(pdl.ValueType()),
            truedest,
            falsedest,
        ),
        (c0,),  # Non-null value
    )
    assert isinstance(notnull_result.terminator_value, Successor)
    assert notnull_result.terminator_value.block == truedest

    # Test with null value
    null_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.IsNotNullOp(
            create_ssa_value(pdl.ValueType()),
            truedest,
            falsedest,
        ),
        (None,),  # Null value
    )
    assert isinstance(null_result.terminator_value, Successor)
    assert null_result.terminator_value.block == falsedest


def test_are_equal():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    truedest = Block()
    falsedest = Block()

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)

    # Test with equal values
    equal_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.AreEqualOp(
            create_ssa_value(pdl.ValueType()),
            create_ssa_value(pdl.ValueType()),
            truedest,
            falsedest,
        ),
        (c0, c0),  # Same value
    )
    assert isinstance(equal_result.terminator_value, Successor)
    assert equal_result.terminator_value.block == truedest

    # Test with unequal values
    unequal_result = interpreter._run_op(  # pyright: ignore[reportPrivateUsage]
        pdl_interp.AreEqualOp(
            create_ssa_value(pdl.ValueType()),
            create_ssa_value(pdl.ValueType()),
            truedest,
            falsedest,
        ),
        (c0, c1),  # Different values
    )
    assert isinstance(unequal_result.terminator_value, Successor)
    assert unequal_result.terminator_value.block == falsedest


def test_replace():
    interpreter = Interpreter(ModuleOp([]))
    implementations = PDLInterpFunctions(Context())
    interpreter.register_implementations(implementations)

    @ModuleOp
    @Builder.implicit_region
    def _():
        # Create a test operation to be replaced
        c0 = create_ssa_value(i32)
        c1 = create_ssa_value(i32)
        original_op = test.TestOp((c0, c1), (i32,), {})
        original_op_result = original_op.results[0]

        implementations.rewriter = PatternRewriter(original_op)

        use_op = test.TestOp((original_op_result,), (i32,), {})

        # Create replacement values
        new_val = create_ssa_value(i32)

        # Test replace operation
        replace_op = pdl_interp.ReplaceOp(
            create_ssa_value(pdl.OperationType()),
            [create_ssa_value(pdl.ValueType())],
        )
        interpreter.run_op(replace_op, (original_op, new_val))

        assert use_op.ops[0] == new_val


def test_create_attribute():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLInterpFunctions(Context()))

    # Create test attribute
    test_attr = StringAttr("test")

    # Test create attribute operation
    create_attr_op = pdl_interp.CreateAttributeOp(test_attr)
    result = interpreter.run_op(create_attr_op, ())

    assert len(result) == 1
    assert result[0] == test_attr


def test_create_operation():
    interpreter = Interpreter(ModuleOp([]))
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    implementations = PDLInterpFunctions(ctx)
    interpreter.register_implementations(implementations)

    @ModuleOp
    @Builder.implicit_region
    def testmodule():
        root = test.TestOp()
        implementations.rewriter = PatternRewriter(root)

    # Create test values
    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    attr = StringAttr("test")

    # Test create operation
    create_op = pdl_interp.CreateOperationOp(
        name="test.op",
        inferred_result_types=UnitAttr(),
        input_attribute_names=[StringAttr("attr")],
        input_operands=[c0, c1],
        input_attributes=[create_ssa_value(pdl.AttributeType())],
        input_result_types=[create_ssa_value(pdl.TypeType())],
    )

    result = interpreter.run_op(create_op, (c0, c1, attr, i32))

    assert len(result) == 1
    assert isinstance(result[0], test.TestOp)
    created_op = result[0]
    assert len(created_op.operands) == 2
    assert created_op.ops == (c0, c1)
    assert created_op.attributes["attr"] == attr
    assert len(created_op.results) == 1
    assert created_op.results[0].type == i32
    # Verify that the operation was inserted:
    assert created_op.parent == testmodule.body.first_block


def test_func():
    # not really a unit test, just check if pdl_interp.func can be called.
    @ModuleOp
    @Builder.implicit_region
    def my_module():
        test.TestOp()

        @Builder.implicit_region((pdl.OperationType(),))
        def body(args: tuple[BlockArgument, ...]) -> None:
            pdl_interp.FinalizeOp()

        pdl_interp.FuncOp(
            "matcher",
            FunctionType.from_lists([pdl.OperationType()], []),
            None,
            None,
            body,
        )

    my_module.verify()
    op = cast(test.TestOp, cast(Block, my_module.body.first_block).first_op)

    interpreter = Interpreter(my_module)
    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    interpreter.register_implementations(PDLInterpFunctions(ctx))
    interpreter.call_op("matcher", (op,))


# def test_record_match():
