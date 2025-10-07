from typing import cast

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import pdl
from xdsl.dialects.builtin import IntegerType, StringAttr, f32, i32
from xdsl.ir import Block, Region, SSAValue
from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
    PatternAnalyzer,
    PredicateTreeBuilder,
)
from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
    AttributeAnswer,
    AttributeConstraintQuestion,
    AttributeLiteralPosition,
    AttributePosition,
    ConstraintPosition,
    ConstraintQuestion,
    EqualToQuestion,
    IsNotNullQuestion,
    OperandCountAtLeastQuestion,
    OperandCountQuestion,
    OperationNameQuestion,
    OperationPosition,
    Position,
    PositionalPredicate,
    Predicate,
    ResultCountAtLeastQuestion,
    ResultCountQuestion,
    ResultPosition,
    StringAnswer,
    TrueAnswer,
    TypeAnswer,
    TypeConstraintQuestion,
    TypeLiteralPosition,
    TypePosition,
    UnsignedAnswer,
)


def test_get_operation_depth():
    pos = (base := OperationPosition(None, depth=0)).get_operand(2).get_type()
    assert pos.get_base_operation() is base
    assert pos.get_operation_depth() == 0

    pos = (
        base := OperationPosition(None, depth=0).get_operand(1).get_defining_op()
    ).get_attribute("my_attr")
    assert pos.get_base_operation() is base
    assert pos.get_operation_depth() == 1


def test_detect_roots():
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type = pdl.TypeOp().result
        op1 = pdl.OperationOp("op1", type_values=(type,)).op
        op2 = pdl.OperationOp("op2").op
        op1_res = pdl.ResultOp(0, op1).val
        op3 = pdl.OperationOp("op3", operand_values=(op1_res,)).op

        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    roots = PatternAnalyzer().detect_roots(pattern)
    assert roots == [op2, op3]


def test_detect_roots_with_rewrite_root_exclusion():
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type = pdl.TypeOp().result
        op1 = pdl.OperationOp("op1", type_values=(type,)).op
        op2 = pdl.OperationOp("op2").op
        op1_res = pdl.ResultOp(0, op1).val
        op3 = pdl.OperationOp("op3", operand_values=(op1_res,)).op

        # Rewrite operation specifies op1 as root, even though it's used by op3
        pdl.RewriteOp(op1, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    roots = PatternAnalyzer().detect_roots(pattern)
    # op1 should be included as a root despite being used by op3, because it's specified as the rewrite root
    assert roots == [op1, op2, op3]


def test_extract_tree_predicates():
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type = pdl.TypeOp(f32).result
        root = pdl.OperationOp("op1", type_values=(type,)).op

        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)

    predicates = p.extract_tree_predicates(root, root_pos, {})

    assert predicates[0] == PositionalPredicate(
        OperationNameQuestion(),
        StringAnswer("op1"),
        root_pos := OperationPosition(None, depth=0),
    )
    assert predicates[1] == PositionalPredicate(
        OperandCountQuestion(),
        UnsignedAnswer(0),
        root_pos,
    )
    assert predicates[2] == PositionalPredicate(
        ResultCountQuestion(),
        UnsignedAnswer(1),
        root_pos,
    )
    assert predicates[3] == PositionalPredicate(
        IsNotNullQuestion(),
        TrueAnswer(),
        result_pos := ResultPosition(root_pos, result_number=0),
    )
    assert predicates[4] == PositionalPredicate(
        TypeConstraintQuestion(),
        TypeAnswer(f32),
        TypePosition(result_pos),
    )
    assert len(predicates) == 5


def test_extract_type_predicates():
    """Test _extract_type_predicates method"""
    analyzer = PatternAnalyzer()

    # Test case 1: TypeOp with constant type - should create predicate
    type_op_with_const = pdl.TypeOp(f32)
    type_pos = TypePosition(OperationPosition(None, depth=0).get_result(0))

    predicates = analyzer._extract_type_predicates(  # pyright: ignore[reportPrivateUsage]
        type_op_with_const, type_pos, {}
    )

    assert len(predicates) == 1
    expected_predicate = Predicate.get_type_constraint(f32)
    assert predicates[0] == PositionalPredicate(
        q=expected_predicate.q, a=expected_predicate.a, position=type_pos
    )

    # Test case 2: TypeOp without constant type - should not create predicates
    type_op_without_const = pdl.TypeOp()
    predicates = analyzer._extract_type_predicates(  # pyright: ignore[reportPrivateUsage]
        type_op_without_const, type_pos, {}
    )
    assert len(predicates) == 0

    # Test case 3: TypesOp with constant types - should create predicate
    from xdsl.dialects.builtin import ArrayAttr

    constant_types = ArrayAttr([f32, i32])
    types_op_with_const = pdl.TypesOp(constant_types)

    predicates = analyzer._extract_type_predicates(  # pyright: ignore[reportPrivateUsage]
        types_op_with_const, type_pos, {}
    )

    assert len(predicates) == 1
    expected_predicate = Predicate.get_type_constraint(constant_types)
    assert predicates[0] == PositionalPredicate(
        q=expected_predicate.q, a=expected_predicate.a, position=type_pos
    )

    # Test case 4: TypesOp without constant types - should not create predicates
    types_op_without_const = pdl.TypesOp()
    predicates = analyzer._extract_type_predicates(  # pyright: ignore[reportPrivateUsage]
        types_op_without_const, type_pos, {}
    )
    assert len(predicates) == 0

    # Test case 5: SSAValue input (should extract from owner)
    type_op_with_const = pdl.TypeOp(i32)
    type_value = type_op_with_const.result

    predicates = analyzer._extract_type_predicates(  # pyright: ignore[reportPrivateUsage]
        type_value.owner, type_pos, {}
    )

    assert len(predicates) == 1
    expected_predicate = Predicate.get_type_constraint(i32)
    assert predicates[0] == PositionalPredicate(
        q=expected_predicate.q, a=expected_predicate.a, position=type_pos
    )


def test_operation_with_named_attributes():
    """Test operation with named attributes"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        attr_type = pdl.TypeOp(f32).result
        attr = pdl.AttributeOp(attr_type).output
        root = pdl.OperationOp(
            "op1",
            attribute_value_names=[StringAttr("my_attr")],
            attribute_values=(attr,),
        ).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    assert len(predicates) == 5
    assert (
        PositionalPredicate(
            IsNotNullQuestion(),
            TrueAnswer(),
            AttributePosition(
                OperationPosition(None, depth=0), attribute_name="my_attr"
            ),
        )
        in predicates
    )


def test_operation_with_constant_attribute():
    """Test operation with constant attribute value"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        from xdsl.dialects.builtin import IntegerAttr

        const_attr = pdl.AttributeOp(IntegerAttr(42, i32)).output
        root = pdl.OperationOp(
            "op1",
            attribute_value_names=[StringAttr("value")],
            attribute_values=(const_attr,),
        ).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    assert len(predicates) == 5
    assert (
        PositionalPredicate(
            AttributeConstraintQuestion(),
            AttributeAnswer(IntegerAttr(42, i32)),
            AttributePosition(OperationPosition(None, depth=0), attribute_name="value"),
        )
        in predicates
    )


def test_operation_with_multiple_results():
    """Test operation with multiple non-variadic results"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type1 = pdl.TypeOp(i32).result
        type2 = pdl.TypeOp(f32).result
        root = pdl.OperationOp("op1", type_values=(type1, type2)).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    assert len(predicates) == 7
    assert (
        PositionalPredicate(
            ResultCountQuestion(),
            UnsignedAnswer(2),
            root_pos := OperationPosition(None, depth=0),
        )
        in predicates
    )
    assert (
        PositionalPredicate(
            TypeConstraintQuestion(),
            TypeAnswer(IntegerType(32)),
            TypePosition(ResultPosition(root_pos, result_number=0)),
        )
        in predicates
    )
    assert (
        PositionalPredicate(
            TypeConstraintQuestion(),
            TypeAnswer(f32),
            TypePosition(ResultPosition(root_pos, result_number=1)),
        )
        in predicates
    )


def test_nested_operations_via_result():
    """Test operations connected via ResultOp"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type1 = pdl.TypeOp(i32).result
        op1 = pdl.OperationOp("producer", type_values=(type1,)).op
        op1_result = pdl.ResultOp(0, op1).val
        root = pdl.OperationOp("consumer", operand_values=(op1_result,)).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    operand_pos = root_pos.get_operand(0)
    defining_op_pos = operand_pos.get_defining_op()
    result_pos = defining_op_pos.get_result(0)
    type_pos = result_pos.get_type()

    assert predicates[0] == PositionalPredicate(
        OperationNameQuestion(),
        StringAnswer("consumer"),
        root_pos,
    )

    assert predicates[1] == PositionalPredicate(
        OperandCountQuestion(),
        UnsignedAnswer(1),
        root_pos,
    )

    assert predicates[2] == PositionalPredicate(
        ResultCountQuestion(),
        UnsignedAnswer(0),
        root_pos,
    )

    assert predicates[3] == PositionalPredicate(
        IsNotNullQuestion(),
        TrueAnswer(),
        operand_pos,
    )

    assert predicates[4] == PositionalPredicate(
        IsNotNullQuestion(),
        TrueAnswer(),
        defining_op_pos,
    )

    assert predicates[5] == PositionalPredicate(
        EqualToQuestion(operand_pos),
        TrueAnswer(),
        result_pos,
    )

    assert predicates[6] == PositionalPredicate(
        IsNotNullQuestion(),
        TrueAnswer(),
        defining_op_pos,
    )

    assert predicates[7] == PositionalPredicate(
        OperationNameQuestion(),
        StringAnswer("producer"),
        defining_op_pos,
    )

    assert predicates[8] == PositionalPredicate(
        OperandCountQuestion(),
        UnsignedAnswer(0),
        defining_op_pos,
    )

    assert predicates[9] == PositionalPredicate(
        ResultCountQuestion(),
        UnsignedAnswer(1),
        defining_op_pos,
    )

    assert predicates[10] == PositionalPredicate(
        IsNotNullQuestion(),
        TrueAnswer(),
        result_pos,
    )

    assert predicates[11] == PositionalPredicate(
        TypeConstraintQuestion(),
        TypeAnswer(IntegerType(32)),
        type_pos,
    )

    assert len(predicates) == 12


def test_apply_native_constraint():
    """Test ApplyNativeConstraintOp"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type1 = pdl.TypeOp(i32).result
        root = op1 = pdl.OperationOp("op1", type_values=(type1,)).op
        pdl.ApplyNativeConstraintOp("my_constraint", [op1], [i32]).res[0]
        # Use constraint_result somehow to make it binding
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {}

    # populate inputs:
    p.extract_tree_predicates(root, root_pos, inputs)

    predicates = p.extract_non_tree_predicates(pattern, inputs)
    assert predicates == [
        PositionalPredicate(
            ConstraintQuestion(
                "my_constraint",
                (OperationPosition(None, depth=0),),
                (IntegerType(32),),
                False,
            ),
            TrueAnswer(),
            OperationPosition(None, depth=0),
        )
    ]


def test_operation_without_name():
    """Test operation without an operation name"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp(None).op  # No operation name
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    assert len(predicates) == 2  # Only operand count and result count
    # Should not include OperationNameQuestion predicate
    assert isinstance(predicates[0].q, OperandCountQuestion)
    assert isinstance(predicates[1].q, ResultCountQuestion)


def test_variadic_operands():
    """Test operation with variadic operands"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Create some operands
        operand1 = pdl.OperandOp().value
        operand2 = pdl.OperandOp().value
        operands = pdl.OperandsOp(None).value  # Variadic operands

        # Operation with both individual and variadic operands
        root = pdl.OperationOp("op1", operand_values=(operand1, operand2, operands)).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    # Should create operand count at least predicate instead of exact count
    operand_count_predicates = [
        p for p in predicates if isinstance(p.q, OperandCountAtLeastQuestion)
    ]
    assert len(operand_count_predicates) == 1
    # Should be at least 2 (non-variadic operands)
    assert operand_count_predicates[0].a == UnsignedAnswer(2)


def test_single_variadic_operand():
    """Test operation with single variadic operand"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        operands = pdl.OperandsOp(
            None
        ).value  # Single variadic operand represents all operands
        root = pdl.OperationOp("op1", operand_values=(operands,)).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    # Should not create individual operand predicates, but process all operands as group
    assert len(predicates) == 2  # At least operation name, result count


def test_variadic_results():
    """Test operation with variadic results"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type1 = pdl.TypeOp(i32).result
        type2 = pdl.TypeOp(f32).result
        types = pdl.TypesOp().result  # Variadic types

        root = pdl.OperationOp("op1", type_values=(type1, type2, types)).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    # Should create result count at least predicate for variadic results
    result_count_predicates = [
        p for p in predicates if isinstance(p.q, ResultCountAtLeastQuestion)
    ]
    assert len(result_count_predicates) == 1
    # Should be at least 2 (non-variadic operands)
    assert result_count_predicates[0].a == UnsignedAnswer(2)

    assert len(result_count_predicates) >= 1


def test_single_variadic_result():
    """Test operation with single variadic result"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        types = pdl.TypesOp().result  # Single variadic result represents all results
        root = pdl.OperationOp("op1", type_values=(types,)).op
        pdl.RewriteOp(root, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})
    assert len(predicates) == 2  # No ResultCountQuestion


def test_existing_value_reuse():
    """Test case where a value is reused (has existing position)"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        shared_type = pdl.TypeOp(i32).result
        op1 = pdl.OperationOp("op1", type_values=(shared_type,)).op
        op2 = pdl.OperationOp("op2", type_values=(shared_type,)).op  # Reuse shared_type
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    # Process first operation
    op1_pos = OperationPosition(depth=0)
    p.extract_tree_predicates(op1, op1_pos, inputs)

    # Process second operation - should create equality predicate for shared type
    op2_pos = OperationPosition(op1_pos, depth=1)
    predicates2 = p.extract_tree_predicates(op2, op2_pos, inputs)

    # Should have equality predicates for the shared type
    equal_predicates = [p for p in predicates2 if isinstance(p.q, EqualToQuestion)]
    assert len(equal_predicates) == 1
    pred = equal_predicates[0]
    assert isinstance(pred.q, EqualToQuestion)
    assert pred.position == TypePosition(ResultPosition(op2_pos, result_number=0))
    assert pred.q.other_position == TypePosition(
        ResultPosition(op1_pos, result_number=0)
    )


def test_attribute_with_type():
    """Test attribute with type constraint"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        attr_type = pdl.TypeOp(i32).result
        attr = pdl.AttributeOp(attr_type).output
        root = pdl.OperationOp(
            "op1",
            attribute_value_names=[StringAttr("typed_attr")],
            attribute_values=(attr,),
        ).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    # Should have type constraint for attribute type
    type_constraint_predicates = [
        p for p in predicates if isinstance(p.q, TypeConstraintQuestion)
    ]

    # Should have exactly one type constraint for the attribute type
    assert len(type_constraint_predicates) == 1
    # Check it's for the correct type
    assert type_constraint_predicates[0].a == TypeAnswer(i32)
    # Check it's at the right position (attribute type position)
    assert isinstance(type_constraint_predicates[0].position, TypePosition)
    assert isinstance(type_constraint_predicates[0].position.parent, AttributePosition)
    assert type_constraint_predicates[0].position.parent.attribute_name == "typed_attr"


def test_results_op():
    """Test ResultsOp (multiple results)"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        op1 = pdl.OperationOp("producer", type_values=()).op
        results = pdl.ResultsOp(op1, 0).val  # Get all results
        root = pdl.OperationOp("consumer", operand_values=(results,)).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    # Should process the results connection - expect 9 predicates total
    assert len(predicates) == 9

    # Check we have the root consumer operation predicates
    assert predicates[0].q == OperationNameQuestion()
    assert predicates[0].a == StringAnswer("consumer")

    # Check we have the producer operation predicates
    producer_name_predicates = [
        p
        for p in predicates
        if isinstance(p.q, OperationNameQuestion) and p.a == StringAnswer("producer")
    ]
    assert len(producer_name_predicates) == 1

    # Check we have the EqualToQuestion for linking results to operands
    equal_predicates = [p for p in predicates if isinstance(p.q, EqualToQuestion)]
    assert len(equal_predicates) == 1


def test_operands_op():
    """Test OperandsOp (multiple operands)"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        pdl.OperationOp("producer").op
        operands = pdl.OperandsOp(None).value  # Get all operands
        root = pdl.OperationOp("consumer", operand_values=(operands,)).op
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    root_pos = OperationPosition(depth=0)
    predicates = p.extract_tree_predicates(root, root_pos, {})

    # Should have exactly 2 predicates for basic consumer operation
    assert len(predicates) == 2
    assert predicates[0].q == OperationNameQuestion()
    assert predicates[0].a == StringAnswer("consumer")
    assert predicates[1].q == ResultCountQuestion()
    assert predicates[1].a == UnsignedAnswer(0)


def test_non_tree_predicates_type_literal():
    """Test non-tree predicates for TypeOp with constant"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        const_type = pdl.TypeOp(i32)  # Constant type not connected to anything
        root = pdl.OperationOp("op1").op
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    # Process tree predicates first
    root_pos = OperationPosition(depth=0)
    p.extract_tree_predicates(root, root_pos, inputs)

    # Extract non-tree predicates should handle unconnected constant type
    p.extract_non_tree_predicates(pattern, inputs)

    # Should have processed the constant type
    assert const_type.result in inputs
    # Check it's at a TypeLiteralPosition with the correct value
    assert isinstance(type_pos := inputs[const_type.result], TypeLiteralPosition)
    assert type_pos.value == i32


def test_non_tree_predicates_types_literal():
    """Test non-tree predicates for TypesOp with constant"""
    from xdsl.dialects.builtin import ArrayAttr

    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        const_types = pdl.TypesOp(ArrayAttr([i32, f32]))  # Constant types
        root = pdl.OperationOp("op1").op
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    root_pos = OperationPosition(depth=0)
    p.extract_tree_predicates(root, root_pos, inputs)

    p.extract_non_tree_predicates(pattern, inputs)

    # Should have processed the constant types
    assert const_types.result in inputs
    # Check it's at a TypeLiteralPosition with the correct array value
    assert isinstance(type_pos := inputs[const_types.result], TypeLiteralPosition)
    # The ArrayAttr should contain both i32 and f32
    from xdsl.dialects.builtin import ArrayAttr

    expected_attr = ArrayAttr([i32, f32])
    assert type_pos.value == expected_attr


def test_non_tree_predicates_attribute_literal():
    """Test non-tree predicates for AttributeOp with constant value"""
    from xdsl.dialects.builtin import IntegerAttr

    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        const_attr = pdl.AttributeOp(IntegerAttr(42, i32))  # Constant attribute
        root = pdl.OperationOp("op1").op
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    root_pos = OperationPosition(depth=0)
    p.extract_tree_predicates(root, root_pos, inputs)

    p.extract_non_tree_predicates(pattern, inputs)

    # Should have processed the constant attribute
    assert const_attr.output in inputs
    # Check it's at an AttributeLiteralPosition with the correct value
    assert isinstance(attr_pos := inputs[const_attr.output], AttributeLiteralPosition)
    from xdsl.dialects.builtin import IntegerAttr

    expected_attr = IntegerAttr(42, i32)
    assert attr_pos.value == expected_attr


@pytest.mark.parametrize("is_negated", [True, False])
def test_constraint_simple(is_negated: bool):
    """Test ApplyNativeConstraintOp basic functionality"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("op1").op
        # Simple constraint that doesn't create conflicts
        pdl.ApplyNativeConstraintOp(
            "simple_constraint", [root], [], is_negated=is_negated
        ).res
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    # Process tree predicates to populate inputs
    root_pos = OperationPosition(depth=0)
    p.extract_tree_predicates(root, root_pos, inputs)

    # Extract non-tree predicates should handle the constraint
    predicates = p.extract_non_tree_predicates(pattern, inputs)

    # Should have constraint predicate
    constraint_predicates = [
        p for p in predicates if isinstance(p.q, ConstraintQuestion)
    ]

    # Should have exactly one constraint predicate
    assert len(constraint_predicates) == 1
    # Check the constraint details
    constraint_pred = constraint_predicates[0]
    assert (
        q := cast(ConstraintQuestion, constraint_pred.q)
    ).name == "simple_constraint"
    assert constraint_pred.a == TrueAnswer()
    assert len(q.arg_positions) == 1  # Should have the root operation as argument
    assert q.is_negated is is_negated


def test_result_op_non_tree():
    """Test ResultOp handling in non-tree predicates"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        producer = pdl.OperationOp("producer", type_values=[pdl.TypeOp(i32).result]).op
        pdl.ResultOp(0, producer)  # Standalone result op
        pdl.OperationOp("consumer").op
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    # Add producer to inputs
    producer_pos = OperationPosition(depth=0)
    inputs[producer] = producer_pos

    predicates = p.extract_non_tree_predicates(pattern, inputs)

    # Should handle the standalone result
    is_not_null_predicates = [
        p for p in predicates if isinstance(p.q, IsNotNullQuestion)
    ]

    # Should have exactly one IsNotNull predicate for the result
    assert len(is_not_null_predicates) == 1
    # Check it's for a result position
    assert isinstance(is_not_null_predicates[0].position, ResultPosition)
    assert is_not_null_predicates[0].position.result_number == 0


def test_results_op_non_tree():
    """Test ResultsOp handling in non-tree predicates"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        producer = pdl.OperationOp("producer").op
        pdl.ResultsOp(producer)  # Standalone results op
        pdl.OperationOp("consumer").op
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    # Add producer to inputs
    producer_pos = OperationPosition(depth=0)
    inputs[producer] = producer_pos

    predicates = p.extract_non_tree_predicates(pattern, inputs)

    # Should handle the standalone results - no predicates expected for unused results
    assert len(predicates) == 0


def test_extract_operation_predicates_non_operation_op():
    """Test that _extract_operation_predicates returns empty list when op_op is not an OperationOp"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Create a non-OperationOp (e.g., TypeOp)
        type_op = pdl.TypeOp(f32)
        op_op = pdl.OperationOp(None, type_values=(type_op.result,))
        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    op_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {}

    # Call _extract_operation_predicates with a non-OperationOp
    predicates = p._extract_operation_predicates(type_op, op_pos, inputs)  # pyright: ignore[reportPrivateUsage]

    # Should return no predicates
    assert len(predicates) == 0  # Root position, so no is_not_null predicate

    # Test with non-root position
    non_root_pos = OperationPosition(depth=1)
    predicates = p._extract_operation_predicates(op_op, non_root_pos, inputs)  # pyright: ignore[reportPrivateUsage]

    # Should return only the is_not_null predicate
    assert len(predicates) == 5
    assert predicates[0].q == IsNotNullQuestion()
    assert predicates[0].a == TrueAnswer()
    assert predicates[0].position == non_root_pos


def test_extract_operation_predicates_with_ignore_operand():
    """Test extract_tree_predicates with ignore_operand parameter"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Create operands with types to ensure they generate predicates
        operand_type = pdl.TypeOp(i32)
        operand1 = pdl.OperandOp(value_type=operand_type.result)
        operand2 = pdl.OperandOp(value_type=operand_type.result)
        operand3 = pdl.OperandOp(value_type=operand_type.result)

        # Create operation with multiple operands
        op = pdl.OperationOp(
            "test_op",
            operand_values=(operand1.value, operand2.value, operand3.value),
        )

        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    op_pos = OperationPosition(depth=0)

    # Extract predicates without ignore_operand
    predicates_all = p.extract_tree_predicates(op.op, op_pos, {})

    # Extract predicates with ignore_operand=1 (ignore second operand)
    predicates_ignore = p.extract_tree_predicates(op.op, op_pos, {}, ignore_operand=1)

    # Should have fewer predicates when ignoring an operand
    # The ignore_operand parameter should result in different predicate generation
    assert len(predicates_ignore) < len(predicates_all)

    ignored = set((p.q, p.a, p.position) for p in predicates_all).difference(
        set((p.q, p.a, p.position) for p in predicates_ignore)
    )
    assert (
        EqualToQuestion(TypePosition(op_pos.get_operand(1))),
        TrueAnswer(),
        TypePosition(op_pos.get_operand(0)),
    ) in ignored
    assert (IsNotNullQuestion(), TrueAnswer(), op_pos.get_operand(1)) in ignored


def test_extract_operand_tree_predicates_with_value_type():
    """Test _extract_operand_tree_predicates where defining_op has value_type"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Create a type for the operand
        operand_type = pdl.TypeOp(i32)

        # Create an OperandOp with a value_type
        operand_op = pdl.OperandOp(value_type=operand_type.result)

        pdl.RewriteOp(None, name="rewrite")

    p = PatternAnalyzer()
    operand_pos = OperationPosition(depth=0).get_operand(0)
    inputs: dict[SSAValue, Position] = {}

    # Extract operand tree predicates
    predicates = p._extract_operand_tree_predicates(  # pyright: ignore[reportPrivateUsage]
        operand_op.value, operand_pos, inputs
    )

    # Should have is_not_null predicate and type constraint predicate
    assert len(predicates) == 2

    # First predicate should be is_not_null for the operand
    assert predicates[0].q == IsNotNullQuestion()
    assert predicates[0].a == TrueAnswer()
    assert predicates[0].position == operand_pos

    # Second predicate should be type constraint for the operand's type
    assert predicates[1].q == TypeConstraintQuestion()
    assert predicates[1].a == TypeAnswer(i32)
    assert isinstance(predicates[1].position, TypePosition)


def test_extract_non_tree_predicates_existing_constraint_result():
    """Test extract_non_tree_predicates when constraint result already exists in inputs

    Note: This test documents a limitation where the existing constraint result
    handling has issues with ConstraintPosition.get_operation_depth().
    The current implementation assumes ConstraintPosition can determine operation depth
    but it has parent=None, causing assertion failures in get_base_operation().
    """
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("op1").op
        # Create a constraint that returns a result
        constraint_result = pdl.ApplyNativeConstraintOp(
            "my_constraint", [root], [i32]
        ).res[0]
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    # Add the root operation to inputs
    root_pos = OperationPosition(depth=0)
    inputs[root] = root_pos

    # Pre-add the constraint result to inputs to simulate the "existing" case
    # Use a simple result position that has proper operation depth support
    result_pos = root_pos.get_result(0)
    inputs[constraint_result] = result_pos

    predicates = p.extract_non_tree_predicates(pattern, inputs)

    assert len(predicates) == 2
    assert predicates[0] == PositionalPredicate(
        EqualToQuestion(
            ConstraintPosition(
                None,
                constraint=ConstraintQuestion(
                    "my_constraint",
                    arg_positions=(root_pos,),
                    result_types=(i32,),
                    is_negated=False,
                ),
                result_index=0,
            ),
        ),
        TrueAnswer(),
        result_pos,
    )
    assert isinstance(predicates[1].q, ConstraintQuestion)


def test_extract_non_tree_predicates_results_op_with_index():
    """Test extract_non_tree_predicates with ResultsOp where index is not None"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Create an operation with multiple results
        type1 = pdl.TypeOp(i32).result
        type2 = pdl.TypeOp(f32).result
        producer = pdl.OperationOp("producer", type_values=(type1, type2)).op

        # Create a ResultsOp with explicit index
        pdl.ResultsOp(producer, index=1)  # index=1 means it's not None

        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    p = PatternAnalyzer()
    inputs: dict[SSAValue, Position] = {}

    # Add producer to inputs
    producer_pos = OperationPosition(depth=0)
    inputs[producer] = producer_pos

    predicates = p.extract_non_tree_predicates(pattern, inputs)

    # Should have exactly one IsNotNull predicate for the ResultsOp with index
    is_not_null_preds = [p for p in predicates if isinstance(p.q, IsNotNullQuestion)]
    assert len(is_not_null_preds) == 1

    # Verify it's for a result group position
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import ResultGroupPosition

    assert isinstance(is_not_null_preds[0].position, ResultGroupPosition)


def test_extract_pattern_predicates_multiple_roots():
    """Test that an error is thrown when multiple roots are present"""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        pdl.OperationOp("op1").op
        pdl.OperationOp("op2").op
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    builder = PredicateTreeBuilder()

    with pytest.raises(ValueError, match="Multi-root patterns are not yet supported."):
        builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]


def test_extract_pattern_predicates_regular_flow():
    """Test a regular flow where there are some regular and non-tree predicates."""
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type1 = pdl.TypeOp(i32).result
        op1 = pdl.OperationOp("op1", type_values=(type1,)).op
        pdl.ApplyNativeConstraintOp("my_constraint", [op1], [])
        pdl.RewriteOp(op1, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)
    builder = PredicateTreeBuilder()

    predicates, best_root, inputs = builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]

    assert best_root is op1

    # Check for some tree predicates
    root_pos = OperationPosition(None, depth=0)
    assert (
        PositionalPredicate(
            OperationNameQuestion(),
            StringAnswer("op1"),
            root_pos,
        )
        in predicates
    )
    assert (
        PositionalPredicate(
            ResultCountQuestion(),
            UnsignedAnswer(1),
            root_pos,
        )
        in predicates
    )

    # Check for non-tree predicate
    assert (
        PositionalPredicate(
            ConstraintQuestion(
                "my_constraint",
                (root_pos,),
                (),
                False,
            ),
            TrueAnswer(),
            root_pos,
        )
        in predicates
    )

    # check that inputs are populated
    assert op1 in inputs
    assert inputs[op1] == root_pos


def test_create_ordered_predicates():
    """Test the _create_ordered_predicates method for frequency analysis."""
    builder = PredicateTreeBuilder()

    # Create mock patterns
    body1 = Region([Block()])
    with ImplicitBuilder(body1.first_block):
        pdl.RewriteOp(None, name="rewrite1")
    pattern1 = pdl.PatternOp(1, "pattern1", body1)

    body2 = Region([Block()])
    with ImplicitBuilder(body2.first_block):
        pdl.RewriteOp(None, name="rewrite2")
    pattern2 = pdl.PatternOp(2, "pattern2", body2)

    # Create mock predicates
    pos1 = OperationPosition(None, depth=0)
    q1 = OperationNameQuestion()
    a1 = StringAnswer("op1")
    a2 = StringAnswer("op2")

    pos2 = OperationPosition(None, depth=1)
    q2 = ResultCountQuestion()
    a3 = UnsignedAnswer(1)

    # A predicate that is present in both patterns
    pred1_p1 = PositionalPredicate(q1, a1, pos1)
    pred1_p2 = PositionalPredicate(q1, a2, pos1)

    # A predicate that is unique to pattern1
    pred2_p1 = PositionalPredicate(q2, a3, pos2)

    # A predicate that is unique to pattern2
    pred3_p2 = PositionalPredicate(q2, a3, pos1)

    all_pattern_predicates = [
        (pattern1, [pred1_p1, pred2_p1, pred1_p1]),  # pred1_p1 is duplicated
        (pattern2, [pred1_p2, pred3_p2]),
    ]

    # Run the method under test
    predicate_map = builder._create_ordered_predicates(  # pyright: ignore[reportPrivateUsage]
        all_pattern_predicates
    )

    # Verification
    assert len(predicate_map) == 3  # Three unique predicates

    key1 = (pos1, q1)
    key2 = (pos2, q2)
    key3 = (pos1, q2)

    assert key1 in predicate_map
    assert key2 in predicate_map
    assert key3 in predicate_map

    # Check ordered predicate for key1
    ordered_pred1 = predicate_map[key1]
    assert ordered_pred1.position == pos1
    assert ordered_pred1.question == q1
    assert ordered_pred1.primary_score == 3  # Occurs twice in p1, once in p2
    assert ordered_pred1.pattern_answers == {pattern1: a1, pattern2: a2}
    assert ordered_pred1.tie_breaker == 0
    # p1_sum = 3**2 + 1**2 = 10. p2_sum = 3**2 + 1**2 = 10. Total = 10 + 10 = 20
    assert ordered_pred1.secondary_score == 20

    # Check ordered predicate for key2
    ordered_pred2 = predicate_map[key2]
    assert ordered_pred2.position == pos2
    assert ordered_pred2.question == q2
    assert ordered_pred2.primary_score == 1
    assert ordered_pred2.pattern_answers == {pattern1: a3}
    assert ordered_pred2.tie_breaker == 1
    # p1_sum = 3**2 + 1**2 = 10.
    assert ordered_pred2.secondary_score == 10

    # Check ordered predicate for key3
    ordered_pred3 = predicate_map[key3]
    assert ordered_pred3.position == pos1
    assert ordered_pred3.question == q2
    assert ordered_pred3.primary_score == 1
    assert ordered_pred3.pattern_answers == {pattern2: a3}
    assert ordered_pred3.tie_breaker == 2
    # p2_sum = 3**2 + 1**2 = 10.
    assert ordered_pred3.secondary_score == 10
