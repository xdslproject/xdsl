from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import pdl, pdl_interp, test
from xdsl.dialects.builtin import (
    ModuleOp,
    StringAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.test_value import create_ssa_value


def test_getters():
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

    # Negative cases

    # Test GetOperandOp with out-of-bounds index
    assert interpreter.run_op(
        pdl_interp.GetOperandOp(5, create_ssa_value(pdl.OperationType())), (op,)
    ) == (None,)

    # Test GetResultOp with out-of-bounds index
    assert interpreter.run_op(
        pdl_interp.GetResultOp(5, create_ssa_value(pdl.OperationType())), (op,)
    ) == (None,)

    # Test GetResultsOp with single-SSA type but multiple results
    single_result_type_op = pdl_interp.GetResultsOp(
        None,
        create_ssa_value(pdl.OperationType()),
        pdl.ValueType(),
    )
    assert interpreter.run_op(single_result_type_op, (op,)) == (None,)

    # Test GetAttributeOp with non-existent attribute
    assert interpreter.run_op(
        pdl_interp.GetAttributeOp(
            "non_existent", create_ssa_value(pdl.OperationType())
        ),
        (op,),
    ) == (None,)

    # Test GetDefiningOpOp with non-OpResult value
    block_arg = Block((), arg_types=(i32,)).args[0]
    assert interpreter.run_op(
        pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType())), (block_arg,)
    ) == (None,)

    # Test GetDefiningOpOp with None input
    assert interpreter.run_op(
        pdl_interp.GetDefiningOpOp(create_ssa_value(pdl.OperationType())), (None,)
    ) == (None,)


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
