from xdsl.dialects import pdl, test
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    StringAttr,
    i32,
    i64,
)
from xdsl.interpreters.eqsat_pdl import EqsatPDLMatcher
from xdsl.ir import Block
from xdsl.utils.test_value import TestSSAValue


def test_match_type():
    matcher = EqsatPDLMatcher()

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
    matcher = EqsatPDLMatcher()

    pdl_op = pdl.TypeOp(IntegerType(32))
    xdsl_value = IntegerType(32)
    ssa_value = pdl_op.result

    assert matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_not_match_fixed_type():
    matcher = EqsatPDLMatcher()

    pdl_op = pdl.TypeOp(IntegerType(64))
    xdsl_value = IntegerType(32)
    ssa_value = pdl_op.result

    assert not matcher.match_type(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {}


def test_match_attribute():
    matcher = EqsatPDLMatcher()

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
    matcher = EqsatPDLMatcher()

    pdl_op = pdl.AttributeOp(IntegerAttr(42, i32))
    ssa_value = pdl_op.output
    xdsl_value = IntegerAttr(42, i32)

    assert matcher.match_attribute(ssa_value, pdl_op, "attr", xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_not_match_fixed_attribute():
    matcher = EqsatPDLMatcher()

    pdl_op = pdl.AttributeOp(IntegerAttr(42, i32))
    ssa_value = pdl_op.output
    xdsl_value = IntegerAttr(24, i32)

    assert not matcher.match_attribute(ssa_value, pdl_op, "attr", xdsl_value)
    assert matcher.matching_context == {}


def test_match_attribute_with_type():
    matcher = EqsatPDLMatcher()

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
    matcher = EqsatPDLMatcher()

    pdl_op = pdl.OperandOp()
    ssa_value = pdl_op.value
    xdsl_value = TestSSAValue(i32)

    # New value
    assert matcher.match_operand(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Same value
    assert matcher.match_operand(ssa_value, pdl_op, xdsl_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Other value
    other_value = TestSSAValue(i32)
    assert not matcher.match_operand(ssa_value, pdl_op, other_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}

    # Wrong type
    type_op = pdl.TypeOp(i64)
    new_pdl_op = pdl.OperandOp(type_op.result)
    new_value = TestSSAValue(i32)
    assert not matcher.match_operand(new_pdl_op.value, new_pdl_op, new_value)
    assert matcher.matching_context == {ssa_value: xdsl_value}


def test_match_result():
    matcher = EqsatPDLMatcher()

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


def test_match_trivial_operation():
    matcher = EqsatPDLMatcher()

    # Create a trivial operation to match against
    trivial_op = test.TestOp()

    # Create EqsatPDL pattern to match an operation with wrong name
    wrong_name_operation_op = pdl.OperationOp(op_name="wrong.name")

    # Match should fail since operation names don't match
    assert not matcher.match_operation(
        wrong_name_operation_op.op, wrong_name_operation_op, trivial_op
    )
    assert matcher.matching_context == {}

    # Create EqsatPDL pattern to match an operation with required attribute
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

    # Create EqsatPDL pattern to match an operation with results
    operation_op_with_results = pdl.OperationOp(
        op_name=None, type_values=[pdl.TypeOp(i32).result]
    )

    # Match should fail since operation has no results
    assert not matcher.match_operation(
        operation_op_with_results.op, operation_op_with_results, trivial_op
    )
    assert matcher.matching_context == {}

    # Create EqsatPDL pattern to match an operation with operands
    operand_op = pdl.OperandOp()
    operation_op_with_operands = pdl.OperationOp(
        op_name=None, operand_values=(operand_op.value,)
    )

    # Match should fail since operation has no operands
    assert not matcher.match_operation(
        operation_op_with_operands.op, operation_op_with_operands, trivial_op
    )
    assert matcher.matching_context == {}

    # Create EqsatPDL pattern to match an operation with no constraints
    operation_op = pdl.OperationOp(op_name=None)

    # Match should succeed and add operation to context
    assert matcher.match_operation(operation_op.op, operation_op, trivial_op)
    assert matcher.matching_context == {operation_op.op: trivial_op}

    # Match should succeed again with same operation
    assert matcher.match_operation(operation_op.op, operation_op, trivial_op)
    assert matcher.matching_context == {operation_op.op: trivial_op}


def test_match_operation_with_multiple_constraints():
    """Test matching an operation with multiple operands, results, and attributes."""
    matcher = EqsatPDLMatcher()

    # Create test operation with 2 operands, 2 results, and 2 attributes
    operand1 = TestSSAValue(i32)
    operand2 = TestSSAValue(i64)
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

    # Create EqsatPDL pattern with wrong attribute type
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

    # Create EqsatPDL pattern with wrong operand type
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

    # Create EqsatPDL pattern with wrong result type
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

    # Create EqsatPDL pattern matching the operation

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
