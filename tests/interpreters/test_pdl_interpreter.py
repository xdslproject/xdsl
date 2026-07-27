from xdsl.builder import Builder, ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, pdl, test
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.pdl import PDLMatcher, PDLRewriteFunctions, PDLRewritePattern
from xdsl.ir import Attribute, Block
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.utils.test_value import create_ssa_value


def test_interpreter_functions():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(PDLRewriteFunctions(Context()))

    c0 = create_ssa_value(i32)
    c1 = create_ssa_value(i32)
    add = arith.AddiOp(c0, c1)
    add_res = add.result

    assert interpreter.run_op(
        pdl.ResultOp(0, create_ssa_value(pdl.OperationType())), (add,)
    ) == (add_res,)


def test_native_constraint():
    @ModuleOp
    @Builder.implicit_region
    def input_module_true():
        test.TestOp.create(properties={"attr": StringAttr("foo")})

    @ModuleOp
    @Builder.implicit_region
    def input_module_false():
        test.TestOp.create(properties={"attr": StringAttr("baar")})

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(42, None).body):
            attr = pdl.AttributeOp().output
            pdl.ApplyNativeConstraintOp("even_length_string", [attr], [])
            op = pdl.OperationOp(
                op_name=None,
                attribute_value_names=ArrayAttr([StringAttr("attr")]),
                attribute_values=[attr],
            ).op
            with ImplicitBuilder(pdl.RewriteOp(op).body):
                pdl.EraseOp(op)

    pdl_rewrite_op = next(
        op for op in pdl_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    def even_length_string(attr: Attribute) -> bool:
        return isinstance(attr, StringAttr) and len(attr.data) == 4

    ctx = Context()

    pattern_walker = PatternRewriteWalker(
        PDLRewritePattern(
            pdl_rewrite_op,
            ctx,
            native_constraints={"even_length_string": even_length_string},
        )
    )

    new_input_module_true = input_module_true.clone()
    pattern_walker.rewrite_module(new_input_module_true)

    new_input_module_false = input_module_false.clone()
    pattern_walker.rewrite_module(new_input_module_false)

    assert new_input_module_false.is_structurally_equivalent(ModuleOp([]))
    assert new_input_module_true.is_structurally_equivalent(input_module_true)


def test_match_type():
    matcher = PDLMatcher()

    pdl_op = pdl.TypeOp()
    ssa_value = pdl_op.result
    xdsl_value = StringAttr("a")

    # New value
    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Same value
    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Other value
    assert not matcher.match_type(ssa_value, pdl_op, StringAttr("b"))
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_match_fixed_type():
    matcher = PDLMatcher()

    pdl_op = pdl.TypeOp(IntegerType(32))
    xdsl_value = IntegerType(32)
    ssa_value = pdl_op.result

    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_not_match_fixed_type():
    matcher = PDLMatcher()

    pdl_op = pdl.TypeOp(IntegerType(64))
    xdsl_value = IntegerType(32)
    ssa_value = pdl_op.result

    assert not matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {}


def test_match_attribute():
    matcher = PDLMatcher()

    pdl_op = pdl.AttributeOp()
    ssa_value = pdl_op.output
    xdsl_value = StringAttr("test")

    # New value
    assert matcher.match_attribute(ssa_value, pdl_op, "attr", xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Same value
    assert matcher.match_attribute(ssa_value, pdl_op, "attr", xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Other value
    assert not matcher.match_attribute(
        ssa_value, pdl_op, "attr", StringAttr("different")
    )
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_match_fixed_attribute():
    matcher = PDLMatcher()

    pdl_op = pdl.AttributeOp(IntegerAttr(42, i32))
    ssa_value = pdl_op.output
    xdsl_value = IntegerAttr(42, i32)

    assert matcher.match_attribute(ssa_value, pdl_op, "attr", xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_not_match_fixed_attribute():
    matcher = PDLMatcher()

    pdl_op = pdl.AttributeOp(IntegerAttr(42, i32))
    ssa_value = pdl_op.output
    xdsl_value = IntegerAttr(24, i32)

    assert not matcher.match_attribute(ssa_value, pdl_op, "attr", xdsl_value)
    assert matcher.matching_context == {}


def test_match_attribute_with_type():
    matcher = PDLMatcher()

    type_op = pdl.TypeOp(i32)
    pdl_op = pdl.AttributeOp(type_op.result)
    ssa_value = pdl_op.output
    xdsl_value = IntegerAttr(42, i32)

    # Value with wrong type
    wrong_value = IntegerAttr(42, i64)
    assert not matcher.match_attribute(ssa_value, pdl_op, "attr", wrong_value)
    assert matcher.matching_context == {}

    # Value with matching type
    assert matcher.match_attribute(ssa_value, pdl_op, "attr", xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value, type_op.result: i32}


def test_match_operand():
    matcher = PDLMatcher()

    pdl_op = pdl.OperandOp()
    ssa_value = pdl_op.value
    xdsl_value = create_ssa_value(i32)

    # New value
    assert matcher.match_operand(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Same value
    assert matcher.match_operand(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Other value
    other_value = create_ssa_value(i32)
    assert not matcher.match_operand(ssa_value, pdl_op, other_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Wrong type
    type_op = pdl.TypeOp(i64)
    new_pdl_op = pdl.OperandOp(type_op.result)
    new_value = create_ssa_value(i32)
    assert not matcher.match_operand(new_pdl_op.value, new_pdl_op, new_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_match_result():
    matcher = PDLMatcher()

    type_op = pdl.TypeOp(IntegerType(32))
    operation_op = pdl.OperationOp(op_name=None, type_values=(type_op.result,))
    result_op = pdl.ResultOp(0, operation_op.op)
    xdsl_op = test.TestOp(result_types=(i32,))
    xdsl_value = xdsl_op.res[0]

    # New result
    # If the result of an operation has the expected type we should match
    assert matcher.match_result(result_op.val, result_op, xdsl_value)
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Same result
    # We should accept the same value given the same constraint
    assert matcher.match_result(result_op.val, result_op, xdsl_value)
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Other result
    # We should not match again with a different value, even if it has the correct type
    other_xdsl_op = test.TestOp(result_types=(i32,))
    other_xdsl_value = other_xdsl_op.res[0]

    assert not matcher.match_result(result_op.val, result_op, other_xdsl_value)
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Wrong type
    # Matching should fail if the result's type differs from the expected type
    wrong_type_op = pdl.TypeOp(i64)
    wrong_type_operation_op = pdl.OperationOp(
        op_name=None, type_values=(wrong_type_op.result,)
    )
    wrong_type_result_op = pdl.ResultOp(0, wrong_type_operation_op.op)

    assert not matcher.match_result(
        wrong_type_result_op.val, wrong_type_result_op, xdsl_value
    )
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Index out of range
    # If the operation has only one result, we should not match results at different
    # indices
    out_of_range_result_op = pdl.ResultOp(1, operation_op.op)
    assert not matcher.match_result(
        out_of_range_result_op.val, out_of_range_result_op, xdsl_value
    )
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Block argument
    # Result patterns should not match on block arguments
    block = Block(arg_types=(i32,))
    block_arg_result_op = pdl.ResultOp(1, operation_op.op)
    assert not matcher.match_result(
        block_arg_result_op.val, block_arg_result_op, block.args[0]
    )
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
    }

    # Index out of range for op with multiple results
    # If the operation has multiple results, we should not match results at indices
    # greater than the number of results
    n_results = 3
    invalid_index = 4
    multi_out_of_range_xdsl_op = test.TestOp(result_types=(i64,) * n_results)
    multi_out_of_range_type_op = pdl.TypeOp(i64)
    multi_out_of_range_operation_op = pdl.OperationOp(
        op_name=None, type_values=(multi_out_of_range_type_op.result,) * n_results
    )
    multi_out_of_range_result_op = pdl.ResultOp(
        index=invalid_index, parent=multi_out_of_range_operation_op.op
    )
    for xval in multi_out_of_range_xdsl_op.res:
        assert not matcher.match_result(
            multi_out_of_range_result_op.val,
            multi_out_of_range_result_op,
            xval,
        )
    assert matcher.matching_context == {
        result_op.val: xdsl_value,
        operation_op.op: xdsl_op,
        type_op.result: i32,
        # Updated keys for matched op and type
        multi_out_of_range_operation_op.op: multi_out_of_range_xdsl_op,
        multi_out_of_range_type_op.result: i64,
    }


def test_match_trivial_operation():
    matcher = PDLMatcher()

    # Create a trivial operation to match against
    trivial_op = test.TestOp()

    # Create PDL pattern to match an operation with wrong name
    wrong_name_operation_op = pdl.OperationOp(op_name="wrong.name")

    # Match should fail since operation names don't match
    assert not matcher.match_operation(
        wrong_name_operation_op.op, wrong_name_operation_op, trivial_op
    )
    assert matcher.matching_context == {}

    # Create PDL pattern to match an operation with required attribute
    operation_op_with_attr = pdl.OperationOp(
        op_name=None,
        attribute_value_names=[StringAttr("required_attr")],
        attribute_values=[pdl.AttributeOp().output],
    )

    # Match should fail since operation is missing required attribute
    assert not matcher.match_operation(
        operation_op_with_attr.op, operation_op_with_attr, trivial_op
    )
    assert matcher.matching_context == {}

    # Create PDL pattern to match an operation with results
    operation_op_with_results = pdl.OperationOp(
        op_name=None, type_values=[pdl.TypeOp(i32).result]
    )

    # Match should fail since operation has no results
    assert not matcher.match_operation(
        operation_op_with_results.op, operation_op_with_results, trivial_op
    )
    assert matcher.matching_context == {}

    # Create PDL pattern to match an operation with operands
    operand_op = pdl.OperandOp()
    operation_op_with_operands = pdl.OperationOp(
        op_name=None, operand_values=(operand_op.value,)
    )

    # Match should fail since operation has no operands
    assert not matcher.match_operation(
        operation_op_with_operands.op, operation_op_with_operands, trivial_op
    )
    assert matcher.matching_context == {}

    # Create PDL pattern to match an operation with no constraints
    operation_op = pdl.OperationOp(op_name=None)

    # Match should succeed and add operation to context
    assert matcher.match_operation(operation_op.op, operation_op, trivial_op)
    assert matcher.matching_context == {operation_op.op: trivial_op}

    # Match should succeed again with same operation
    assert matcher.match_operation(operation_op.op, operation_op, trivial_op)
    assert matcher.matching_context == {operation_op.op: trivial_op}


def test_match_operation_with_multiple_constraints():
    """Test matching an operation with multiple operands, results, and attributes."""
    matcher = PDLMatcher()

    # Create test operation with 2 operands, 2 results, and 2 attributes
    operand1 = create_ssa_value(i32)
    operand2 = create_ssa_value(i64)
    test_op = test.TestOp(
        operands=[operand1, operand2],
        result_types=[i32, i64],
        attributes={"attr1": StringAttr("test1"), "attr2": IntegerAttr(42, i32)},
    )

    pdl_type1 = pdl.TypeOp(i32).result
    pdl_type2 = pdl.TypeOp(i64).result
    pdl_attr1 = pdl.AttributeOp(StringAttr("test1")).output
    pdl_attr2 = pdl.AttributeOp(IntegerAttr(42, i32)).output
    pdl_operand1 = pdl.OperandOp(pdl_type1).value
    pdl_operand2 = pdl.OperandOp(pdl_type2).value

    # Create PDL pattern with wrong attribute type
    wrong_attr = pdl.AttributeOp(IntegerAttr(42, i64)).output  # i64 instead of i32
    operation_wrong_attr = pdl.OperationOp(
        op_name=None,
        operand_values=[pdl_operand1, pdl_operand2],
        type_values=[pdl_type1, pdl_type2],
        attribute_value_names=ArrayAttr([StringAttr("attr1"), StringAttr("attr2")]),
        attribute_values=[pdl_attr1, wrong_attr],
    )
    assert not matcher.match_operation(
        operation_wrong_attr.op, operation_wrong_attr, test_op
    )
    assert matcher.matching_context == {
        pdl_attr1: StringAttr("test1"),
    }

    # Create PDL pattern with wrong operand type
    wrong_operand_type = pdl.TypeOp(i32).result
    wrong_operand = pdl.OperandOp(wrong_operand_type).value
    operation_wrong_operand = pdl.OperationOp(
        op_name=None,
        operand_values=[pdl_operand1, wrong_operand],  # Both i32 instead of i32,i64
        type_values=[pdl_type1, pdl_type2],
        attribute_value_names=ArrayAttr([StringAttr("attr1"), StringAttr("attr2")]),
        attribute_values=[pdl_attr1, pdl_attr2],
    )
    assert not matcher.match_operation(
        operation_wrong_operand.op, operation_wrong_operand, test_op
    )
    assert matcher.matching_context == {
        pdl_attr1: StringAttr("test1"),
        pdl_attr2: IntegerAttr(42, i32),
        pdl_operand1: operand1,
        pdl_type1: i32,
    }

    # Create PDL pattern with wrong result type
    wrong_result_type = pdl.TypeOp(i32).result
    operation_wrong_result = pdl.OperationOp(
        op_name=None,
        operand_values=[pdl_operand1, pdl_operand2],
        type_values=[
            pdl_type1,
            wrong_result_type,
        ],  # Both i32 instead of i32,i64
        attribute_value_names=ArrayAttr([StringAttr("attr1"), StringAttr("attr2")]),
        attribute_values=[pdl_attr1, pdl_attr2],
    )
    assert not matcher.match_operation(
        operation_wrong_result.op, operation_wrong_result, test_op
    )
    assert matcher.matching_context == {
        pdl_type1: i32,
        pdl_type2: i64,
        pdl_attr1: StringAttr("test1"),
        pdl_attr2: IntegerAttr(42, i32),
        pdl_operand1: operand1,
        pdl_operand2: operand2,
    }

    # Create PDL pattern matching the operation

    operation_op = pdl.OperationOp(
        op_name=None,
        operand_values=[pdl_operand1, pdl_operand2],
        type_values=[pdl_type1, pdl_type2],
        attribute_value_names=ArrayAttr([StringAttr("attr1"), StringAttr("attr2")]),
        attribute_values=[pdl_attr1, pdl_attr2],
    )

    # Match should succeed and add all matched values to context
    assert matcher.match_operation(operation_op.op, operation_op, test_op)
    assert matcher.matching_context == {
        operation_op.op: test_op,
        pdl_type1: i32,
        pdl_type2: i64,
        pdl_attr1: StringAttr("test1"),
        pdl_attr2: IntegerAttr(42, i32),
        pdl_operand1: operand1,
        pdl_operand2: operand2,
    }


def test_native_constraint_constant_parameter():
    """
    Check that `pdl.apply_native_constraint` can take constant attribute parameters
    that are not otherwise matched.
    """

    @ModuleOp
    @Builder.implicit_region
    def input_module_true():
        test.TestOp.create(properties={"attr": StringAttr("foo")})

    @ModuleOp
    @Builder.implicit_region
    def input_module_false():
        test.TestOp.create(properties={"attr": StringAttr("baar")})

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(42, None).body):
            attr = pdl.AttributeOp().output
            four = pdl.AttributeOp(IntegerAttr(4, i32)).output
            pdl.ApplyNativeConstraintOp("length_string", [attr, four], [])
            op = pdl.OperationOp(
                op_name=None,
                attribute_value_names=ArrayAttr([StringAttr("attr")]),
                attribute_values=[attr],
            ).op
            with ImplicitBuilder(pdl.RewriteOp(op).body):
                pdl.EraseOp(op)

    pdl_rewrite_op = next(
        op for op in pdl_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    def length_string(attr: Attribute, size: Attribute) -> bool:
        return (
            isinstance(attr, StringAttr)
            and isinstance(size, IntegerAttr)
            and len(attr.data) == size.value.data
        )

    ctx = Context()

    pattern_walker = PatternRewriteWalker(
        PDLRewritePattern(
            pdl_rewrite_op, ctx, native_constraints={"length_string": length_string}
        )
    )

    new_input_module_true = input_module_true.clone()
    pattern_walker.rewrite_module(new_input_module_true)

    new_input_module_false = input_module_false.clone()
    pattern_walker.rewrite_module(new_input_module_false)

    assert new_input_module_false.is_structurally_equivalent(ModuleOp([]))
    assert new_input_module_true.is_structurally_equivalent(input_module_true)


def test_insert_operation_without_replace():
    """Test that operations created in a rewrite region are inserted by default."""

    @ModuleOp
    @Builder.implicit_region
    def input_module():
        test.TestOp.create(properties={"attr": StringAttr("original")})

    @ModuleOp
    @Builder.implicit_region
    def pdl_module():
        with ImplicitBuilder(pdl.PatternOp(1, None).body):
            attr = pdl.AttributeOp(StringAttr("original")).output
            op = pdl.OperationOp(
                op_name="test.op",
                attribute_value_names=ArrayAttr([StringAttr("attr")]),
                attribute_values=[attr],
            ).op
            with ImplicitBuilder(pdl.RewriteOp(op).body):
                # Create a new operation but don't use it in a replace
                pdl.OperationOp(op_name="test.op", type_values=())
                pdl.EraseOp(op)

    pdl_rewrite_op = next(
        op for op in pdl_module.walk() if isinstance(op, pdl.RewriteOp)
    )

    ctx = Context()
    ctx.register_dialect("test", lambda: test.Test)
    pattern_walker = PatternRewriteWalker(PDLRewritePattern(pdl_rewrite_op, ctx))

    # Apply the pattern
    new_module = input_module.clone()
    pattern_walker.rewrite_module(new_module)

    # The original op should be erased and the new op should be inserted
    expected_module = ModuleOp([test.TestOp()])
    assert new_module.is_structurally_equivalent(expected_module)
