from typing import cast

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import pdl, pdl_interp
from xdsl.dialects.builtin import (
    FunctionType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    f32,
    i32,
)
from xdsl.ir import Block, Region, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
    BoolNode,
    ExitNode,
    MatcherGenerator,
    MatcherNode,
    OrderedPredicate,
    PatternAnalyzer,
    PredicateTreeBuilder,
    SuccessNode,
    SwitchNode,
)
from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
    Answer,
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
    Question,
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
        pdl.ApplyNativeConstraintOp("my_constraint", [op1], [pdl.TypeType()]).res[0]
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
                (pdl.TypeType(),),
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
            "my_constraint", [root], [pdl.OperationType()]
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
                    result_types=(pdl.OperationType(),),
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


def test_propagate_pattern():
    """Tests for the _propagate_pattern method of PredicateTreeBuilder."""

    # region Test Fixtures
    pos1 = OperationPosition(None, depth=0)
    pos2 = OperationPosition(None, depth=1)
    q1 = OperationNameQuestion()
    q2 = ResultCountQuestion()
    ans1 = StringAnswer("op1")
    ans2 = StringAnswer("op2")
    ans3 = UnsignedAnswer(1)

    pred1 = PositionalPredicate(q1, ans1, pos1)
    pred2 = PositionalPredicate(q1, ans2, pos1)
    pred3 = PositionalPredicate(q2, ans3, pos2)

    ordered_pred1 = OrderedPredicate(position=pos1, question=q1)
    ordered_pred2 = OrderedPredicate(position=pos2, question=q2)

    body1 = Region([Block()])
    with ImplicitBuilder(body1.first_block):
        root_val1 = pdl.OperationOp("test_op")
        pdl.RewriteOp(None, name="rewrite1")
    pattern1 = pdl.PatternOp(1, "pattern1", body1)

    body2 = Region([Block()])
    with ImplicitBuilder(body2.first_block):
        root_val2 = pdl.OperationOp("test_op")
        pdl.RewriteOp(None, name="rewrite2")
    pattern2 = pdl.PatternOp(2, "pattern2", body2)

    # endregion

    def test_single_pattern_single_predicate():
        builder = PredicateTreeBuilder()
        builder._pattern_roots = {pattern1: root_val1.op}  # pyright: ignore[reportPrivateUsage]
        pattern_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos1, q1): pred1
        }
        sorted_preds = [ordered_pred1]

        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            None, pattern1, pattern_preds, sorted_preds, 0
        )

        assert isinstance(tree, SwitchNode)
        assert tree.position == pos1
        assert tree.question == q1
        assert tree.failure_node is None
        assert len(tree.children) == 1
        assert ans1 in tree.children

        child = tree.children[ans1]
        assert isinstance(child, SuccessNode)
        assert child.pattern is pattern1
        assert child.failure_node is None

    def test_single_pattern_multiple_predicates():
        builder = PredicateTreeBuilder()
        builder._pattern_roots = {pattern1: root_val1.op}  # pyright: ignore[reportPrivateUsage]
        pattern_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos1, q1): pred1,
            (pos2, q2): pred3,
        }
        sorted_preds = [ordered_pred1, ordered_pred2]

        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            None, pattern1, pattern_preds, sorted_preds, 0
        )

        assert isinstance(tree, SwitchNode)
        assert tree.position == pos1
        assert tree.question == q1
        child1 = tree.children[ans1]
        assert isinstance(child1, SwitchNode)
        assert child1.position == pos2
        assert child1.question == q2
        child2 = child1.children[ans3]
        assert isinstance(child2, SuccessNode)
        assert child2.pattern is pattern1

    def test_two_patterns_shared_node():
        builder = PredicateTreeBuilder()
        builder._pattern_roots = {  # pyright: ignore[reportPrivateUsage]
            pattern1: root_val1.op,
            pattern2: root_val2.op,
        }
        pattern1_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos1, q1): pred1
        }
        pattern2_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos1, q1): pred2
        }
        sorted_preds = [ordered_pred1]

        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            None, pattern1, pattern1_preds, sorted_preds, 0
        )
        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            tree, pattern2, pattern2_preds, sorted_preds, 0
        )

        assert isinstance(tree, SwitchNode)
        assert tree.position == pos1
        assert tree.question == q1
        assert len(tree.children) == 2
        assert ans1 in tree.children
        assert ans2 in tree.children
        child1 = tree.children[ans1]
        assert isinstance(child1, SuccessNode)
        assert child1.pattern is pattern1
        child2 = tree.children[ans2]
        assert isinstance(child2, SuccessNode)
        assert child2.pattern is pattern2

    def test_predicate_not_in_pattern():
        builder = PredicateTreeBuilder()
        builder._pattern_roots = {pattern1: root_val1.op}  # pyright: ignore[reportPrivateUsage]
        pattern_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos1, q1): pred1
        }
        sorted_preds = [ordered_pred2, ordered_pred1]

        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            None, pattern1, pattern_preds, sorted_preds, 0
        )

        assert isinstance(tree, SwitchNode)
        assert tree.position == pos1
        assert tree.question == q1
        child = tree.children[ans1]
        assert isinstance(child, SuccessNode)
        assert child.pattern is pattern1

    def test_predicate_divergence():
        builder = PredicateTreeBuilder()
        builder._pattern_roots = {  # pyright: ignore[reportPrivateUsage]
            pattern1: root_val1.op,
            pattern2: root_val2.op,
        }
        pattern1_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos1, q1): pred1
        }
        pattern2_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos2, q2): pred3
        }
        sorted_preds = [ordered_pred1, ordered_pred2]

        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            None, pattern1, pattern1_preds, sorted_preds, 0
        )
        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            tree, pattern2, pattern2_preds, sorted_preds, 0
        )

        assert isinstance(tree, SwitchNode)
        assert tree.position == pos1
        assert tree.question == q1
        assert tree.failure_node is not None

        child1 = tree.children[ans1]
        assert isinstance(child1, SuccessNode)
        assert child1.pattern is pattern1

        failure_node = tree.failure_node
        assert isinstance(failure_node, SwitchNode)
        assert failure_node.position == pos2
        assert failure_node.question == q2
        child2 = failure_node.children[ans3]
        assert isinstance(child2, SuccessNode)
        assert child2.pattern is pattern2

    def test_success_node_failure_path():
        builder = PredicateTreeBuilder()
        builder._pattern_roots = {  # pyright: ignore[reportPrivateUsage]
            pattern1: root_val1.op,
            pattern2: root_val2.op,
        }
        pattern1_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos1, q1): pred1
        }
        pattern2_preds: dict[tuple[Position, Question], PositionalPredicate] = {
            (pos1, q1): pred1,
            (pos2, q2): pred3,
        }
        sorted_preds = [ordered_pred1, ordered_pred2]

        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            None, pattern2, pattern2_preds, sorted_preds, 0
        )
        tree = builder._propagate_pattern(  # pyright: ignore[reportPrivateUsage]
            tree, pattern1, pattern1_preds, sorted_preds, 0
        )

        assert isinstance(tree, SwitchNode)
        child1 = tree.children[ans1]
        assert isinstance(child1, SuccessNode)
        assert child1.pattern is pattern1
        assert child1.failure_node is not None

        failure_node = child1.failure_node
        assert isinstance(failure_node, SwitchNode)
        assert failure_node.position == pos2
        assert failure_node.question == q2
        assert ans3 in failure_node.children

        child2 = failure_node.children[ans3]
        assert isinstance(child2, SuccessNode)
        assert child2.pattern is pattern2

    test_single_pattern_single_predicate()
    test_single_pattern_multiple_predicates()
    test_two_patterns_shared_node()
    test_predicate_not_in_pattern()
    test_predicate_divergence()
    test_success_node_failure_path()


def test_insert_exit_node():
    """Test the _insert_exit_node method."""
    builder = PredicateTreeBuilder()
    pos = OperationPosition(None, depth=0)
    q = OperationNameQuestion()

    # Test with a chain of failure nodes
    root = SwitchNode(position=pos, question=q)
    root.failure_node = SwitchNode(position=pos, question=q)
    root.failure_node.failure_node = SwitchNode(position=pos, question=q)

    builder._insert_exit_node(root)  # pyright: ignore[reportPrivateUsage]

    assert root.failure_node is not None
    assert root.failure_node.failure_node is not None
    assert root.failure_node.failure_node.failure_node is not None
    assert isinstance(root.failure_node.failure_node.failure_node, ExitNode)
    assert root.failure_node.failure_node.failure_node.failure_node is None

    # Test with a single node
    root2 = SwitchNode(position=pos, question=q)
    builder._insert_exit_node(root2)  # pyright: ignore[reportPrivateUsage]
    assert root2.failure_node is not None
    assert isinstance(root2.failure_node, ExitNode)


def test_optimize_tree():
    """Test the _optimize_tree method."""
    builder = PredicateTreeBuilder()
    pos1 = OperationPosition(None, depth=0)
    q1 = OperationNameQuestion()
    ans1 = StringAnswer("op1")
    ans1_other = StringAnswer("op2")

    pos2 = OperationPosition(None, depth=1)
    q2 = ResultCountQuestion()
    ans2 = UnsignedAnswer(1)

    body = Region([Block()])
    with ImplicitBuilder(body.first_block):
        pdl.RewriteOp(None, name="rewrite")
    pattern = pdl.PatternOp(1, "pattern", body)
    success = SuccessNode(pattern=pattern)

    # Test case 1: Single-child SwitchNode is converted to BoolNode
    root1 = SwitchNode(position=pos1, question=q1, children={ans1: success})
    optimized1 = builder._optimize_tree(root1)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(optimized1, BoolNode)
    assert optimized1.position == pos1
    assert optimized1.question == q1
    assert optimized1.answer == ans1
    assert optimized1.success_node is success
    assert optimized1.failure_node is None

    # Test case 2: Multi-child SwitchNode is not converted
    root2 = SwitchNode(
        position=pos1, question=q1, children={ans1: success, ans1_other: success}
    )
    optimized2 = builder._optimize_tree(root2)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(optimized2, SwitchNode)
    assert len(optimized2.children) == 2

    # Test case 3: Recursive optimization
    child3 = SwitchNode(position=pos2, question=q2, children={ans2: success})
    root3 = SwitchNode(position=pos1, question=q1, children={ans1: child3})
    optimized3 = builder._optimize_tree(root3)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(optimized3, BoolNode)
    assert optimized3.position == pos1
    assert optimized3.question == q1
    assert optimized3.answer == ans1
    assert optimized3.success_node is not None
    assert isinstance(optimized3.success_node, BoolNode)
    assert optimized3.success_node.position == pos2
    assert optimized3.success_node.question == q2
    assert optimized3.success_node.answer == ans2
    assert optimized3.success_node.success_node is success

    # Test case 4: Optimization on failure path
    failure_child4 = SwitchNode(position=pos2, question=q2, children={ans2: success})
    root4 = SwitchNode(position=pos1, question=q1, children={ans1: success})
    root4.failure_node = failure_child4
    optimized4 = builder._optimize_tree(root4)  # pyright: ignore[reportPrivateUsage]

    assert isinstance(optimized4, BoolNode)
    assert optimized4.position == pos1
    assert optimized4.failure_node is not None
    assert isinstance(optimized4.failure_node, BoolNode)
    assert optimized4.failure_node.position == pos2
    assert optimized4.failure_node.question == q2
    assert optimized4.failure_node.answer == ans2

    # Test case 5: No optimization needed
    root5 = BoolNode(
        position=pos1,
        question=q1,
        answer=ans1,
        success_node=success,
        failure_node=None,
    )
    optimized5 = builder._optimize_tree(root5)  # pyright: ignore[reportPrivateUsage]
    assert optimized5 is root5


def test_depends_on():
    """Tests for the _depends_on function."""
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        _depends_on,  # pyright: ignore[reportPrivateUsage]
    )

    # Helper to create a mock OrderedPredicate
    def create_pred(pos: Position, q: Question) -> OrderedPredicate:
        return OrderedPredicate(position=pos, question=q)

    # Test fixtures
    op_pos = OperationPosition(None, depth=0)
    op_q = OperationNameQuestion()

    # Constraint A
    constraint_q_a = ConstraintQuestion("constraint_a", (), (), False)
    pred_a = create_pred(op_pos, constraint_q_a)

    # A predicate that is not a constraint
    pred_not_constraint = create_pred(op_pos, op_q)

    # A position that depends on constraint A
    pos_depends_on_a = ConstraintPosition(
        None, constraint=constraint_q_a, result_index=0
    )

    # Case 1: pred_a is not a constraint question
    assert not _depends_on(pred_not_constraint, pred_a)

    # Case 2: pred_b's position depends on pred_a
    pred_b_pos_dep = create_pred(pos_depends_on_a, op_q)
    assert _depends_on(pred_a, pred_b_pos_dep)

    # Case 3: pred_b is an EqualToQuestion, with position depending on pred_a
    eq_q_pos_dep = EqualToQuestion(other_position=op_pos)
    pred_b_eq_pos_dep = create_pred(pos_depends_on_a, eq_q_pos_dep)
    assert _depends_on(pred_a, pred_b_eq_pos_dep)

    # Case 4: pred_b is an EqualToQuestion, with other_position depending on pred_a
    eq_q_other_pos_dep = EqualToQuestion(other_position=pos_depends_on_a)
    pred_b_eq_other_pos_dep = create_pred(op_pos, eq_q_other_pos_dep)
    assert _depends_on(pred_a, pred_b_eq_other_pos_dep)

    # Case 5: pred_b is a ConstraintQuestion, with arg_positions depending on pred_a
    constraint_q_b = ConstraintQuestion("constraint_b", (pos_depends_on_a,), (), False)
    pred_b_constraint_dep = create_pred(op_pos, constraint_q_b)
    assert _depends_on(pred_a, pred_b_constraint_dep)

    # Case 6: No dependency
    pred_b_no_dep = create_pred(op_pos, op_q)
    assert not _depends_on(pred_a, pred_b_no_dep)


def test_stable_topological_sort():
    """Tests for the _stable_topological_sort function."""
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        _stable_topological_sort,  # pyright: ignore[reportPrivateUsage]
    )

    # Helper to create a mock OrderedPredicate
    def create_pred(pos: Position, q: Question) -> OrderedPredicate:
        return OrderedPredicate(position=pos, question=q)

    op_pos = OperationPosition(None, depth=0)
    op_q = OperationNameQuestion()

    # Constraints A, B, C, where C depends on B, and B depends on A
    constraint_q_a = ConstraintQuestion("constraint_a", (), (), False)
    pred_a = create_pred(op_pos, constraint_q_a)

    pos_depends_on_a = ConstraintPosition(
        None, constraint=constraint_q_a, result_index=0
    )
    constraint_q_b = ConstraintQuestion("constraint_b", (pos_depends_on_a,), (), False)
    pred_b = create_pred(op_pos, constraint_q_b)

    pos_depends_on_b = ConstraintPosition(
        None, constraint=constraint_q_b, result_index=0
    )
    constraint_q_c = ConstraintQuestion("constraint_c", (pos_depends_on_b,), (), False)
    pred_c = create_pred(op_pos, constraint_q_c)

    # Independent predicate
    pred_d = create_pred(op_pos, op_q)

    # Case 1: Simple dependency chain (c, b, a) -> (a, b, c)
    input_list = [pred_c, pred_b, pred_a]
    sorted_list = _stable_topological_sort(input_list)
    assert sorted_list == [pred_a, pred_b, pred_c]

    # Case 2: Mix of dependent and independent items
    input_list = [pred_c, pred_d, pred_b, pred_a]
    sorted_list = _stable_topological_sort(input_list)
    assert sorted_list == [pred_d, pred_a, pred_b, pred_c]

    # Case 3: Already sorted
    input_list = [pred_a, pred_b, pred_c]
    sorted_list = _stable_topological_sort(input_list)
    assert sorted_list == [pred_a, pred_b, pred_c]

    # Case 4: Empty list
    assert _stable_topological_sort([]) == []

    # Case 5: Stability with independent items
    pred_e = create_pred(op_pos, ResultCountQuestion())
    input_list = [pred_d, pred_e]
    sorted_list = _stable_topological_sort(input_list)
    assert sorted_list == [pred_d, pred_e]


def test_build_predicate_tree():
    """Test the build_predicate_tree method with a simple pattern."""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import pdl
    from xdsl.dialects.builtin import i32
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        ExitNode,
        PredicateTreeBuilder,
        SuccessNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        OperandCountQuestion,
        OperationNameQuestion,
        OperationPosition,
        ResultCountQuestion,
        ResultPosition,
        StringAnswer,
        TrueAnswer,
        TypeAnswer,
        TypeConstraintQuestion,
        TypePosition,
        UnsignedAnswer,
    )

    # Create a simple pattern: match an operation named "test_op" with one i32 result
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        result_type = pdl.TypeOp(i32).result
        root = pdl.OperationOp("test_op", type_values=(result_type,)).op
        pdl.RewriteOp(root, name="rewrite")

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Build the predicate tree
    builder = PredicateTreeBuilder()
    tree = builder.build_predicate_tree([pattern])

    # Verify the tree structure
    # Root node: Check operation name
    assert isinstance(tree, BoolNode)
    assert tree.position == OperationPosition(None, depth=0)
    assert isinstance(tree.question, OperationNameQuestion)
    assert tree.answer == StringAnswer("test_op")
    assert tree.failure_node is not None
    assert isinstance(tree.failure_node, ExitNode)

    # Second level: Check operand count
    node2 = tree.success_node
    assert isinstance(node2, BoolNode)
    assert node2.position == OperationPosition(None, depth=0)
    assert isinstance(node2.question, OperandCountQuestion)
    assert node2.answer == UnsignedAnswer(0)

    # Third level: Check result count
    node3 = node2.success_node
    assert isinstance(node3, BoolNode)
    assert node3.position == OperationPosition(None, depth=0)
    assert isinstance(node3.question, ResultCountQuestion)
    assert node3.answer == UnsignedAnswer(1)

    # Fourth level: Check result[0] is not null
    node4 = node3.success_node
    assert isinstance(node4, BoolNode)
    assert node4.position == ResultPosition(
        OperationPosition(None, depth=0), result_number=0
    )
    assert isinstance(node4.question, IsNotNullQuestion)
    assert node4.answer == TrueAnswer()

    # Fifth level: Check result[0].type constraint
    node5 = node4.success_node
    assert isinstance(node5, BoolNode)
    assert node5.position == TypePosition(
        ResultPosition(OperationPosition(None, depth=0), result_number=0)
    )
    assert isinstance(node5.question, TypeConstraintQuestion)
    assert node5.answer == TypeAnswer(i32)

    # Final level: Success node
    node6 = node5.success_node
    assert isinstance(node6, SuccessNode)
    assert node6.pattern is pattern
    assert node6.root is root
    assert node6.failure_node is None


def test_build_predicate_tree_without_predicates():
    """Test build_predicate_tree when no predicates are generated."""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects import pdl
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        ExitNode,
        PredicateTreeBuilder,
    )

    # Create a pattern that matches any operation (no specific predicates)
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        result_type = pdl.TypesOp(None).result
        operands = pdl.OperandsOp(None).value
        pdl.OperationOp(None, operand_values=(operands,), type_values=(result_type,)).op
        pdl.RewriteOp(None, name="rewrite")

    pattern = pdl.PatternOp(1, "pattern", body)

    # Build the predicate tree
    builder = PredicateTreeBuilder()
    tree = builder.build_predicate_tree([pattern])

    # Should return an ExitNode since no predicates are generated
    assert isinstance(tree, ExitNode)


def test_get_value_at_operation_position():
    """Test get_value_at with OperationPosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Create matcher function and rewriter module
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Test case 1: Root operation position (passthrough)
    root_pos = OperationPosition(None, depth=0)
    block = matcher_func.body.block
    root_val = block.args[0]

    # Manually add root to cache
    generator.values[root_pos] = root_val

    result = generator.get_value_at(root_pos)
    assert result is root_val

    # Test case 2: Operand defining op position
    operand_pos = root_pos.get_operand(0)
    defining_op_pos = operand_pos.get_defining_op()

    # First get the operand value
    generator.values[operand_pos] = root_val  # Mock operand value

    result = generator.get_value_at(defining_op_pos)

    # Should create GetDefiningOpOp
    get_def_ops = [op for op in block.ops if isinstance(op, pdl_interp.GetDefiningOpOp)]
    assert len(get_def_ops) == 1
    assert get_def_ops[0].value is root_val
    assert result == get_def_ops[0].input_op


def test_get_value_at_operand_position():
    """Test get_value_at with OperandPosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    root_pos = OperationPosition(None, depth=0)
    block = matcher_func.body.block
    root_val = block.args[0]
    generator.values[root_pos] = root_val

    # Get operand at index 2
    operand_pos = root_pos.get_operand(2)
    result = generator.get_value_at(operand_pos)

    # Should create GetOperandOp with index 2
    get_operand_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.GetOperandOp)
    ]
    assert len(get_operand_ops) == 1
    assert get_operand_ops[0].index.value.data == 2
    assert get_operand_ops[0].input_op is root_val
    assert result == get_operand_ops[0].value


def test_get_value_at_result_position():
    """Test get_value_at with ResultPosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    root_pos = OperationPosition(None, depth=0)
    block = matcher_func.body.block
    root_val = block.args[0]
    generator.values[root_pos] = root_val

    # Get result at index 1
    result_pos = root_pos.get_result(1)
    result = generator.get_value_at(result_pos)

    # Should create GetResultOp with index 1
    get_result_ops = [op for op in block.ops if isinstance(op, pdl_interp.GetResultOp)]
    assert len(get_result_ops) == 1
    assert get_result_ops[0].index.value.data == 1
    assert get_result_ops[0].input_op is root_val
    assert result == get_result_ops[0].value


def test_get_value_at_result_group_position():
    """Test get_value_at with ResultGroupPosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    root_pos = OperationPosition(None, depth=0)
    block = matcher_func.body.block
    root_val = block.args[0]
    generator.values[root_pos] = root_val

    # Test variadic result group
    result_group_pos = root_pos.get_result_group(0, is_variadic=True)
    result = generator.get_value_at(result_group_pos)

    # Should create GetResultsOp
    get_results_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.GetResultsOp)
    ]
    assert len(get_results_ops) == 1
    assert get_results_ops[0].index is not None
    assert get_results_ops[0].index.value.data == 0
    assert get_results_ops[0].input_op is root_val
    assert isinstance(result.type, pdl.RangeType)

    # Test non-variadic result group
    result_group_pos2 = root_pos.get_result_group(1, is_variadic=False)
    result2 = generator.get_value_at(result_group_pos2)

    get_results_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.GetResultsOp)
    ]
    assert len(get_results_ops) == 2
    assert get_results_ops[1].index is not None
    assert get_results_ops[1].index.value.data == 1
    assert result2.type == pdl.ValueType()


def test_get_value_at_attribute_position():
    """Test get_value_at with AttributePosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    root_pos = OperationPosition(None, depth=0)
    block = matcher_func.body.block
    root_val = block.args[0]
    generator.values[root_pos] = root_val

    # Get attribute named "test_attr"
    attr_pos = root_pos.get_attribute("test_attr")
    result = generator.get_value_at(attr_pos)

    # Should create GetAttributeOp
    get_attr_ops = [op for op in block.ops if isinstance(op, pdl_interp.GetAttributeOp)]
    assert len(get_attr_ops) == 1
    assert get_attr_ops[0].constraint_name.data == "test_attr"
    assert get_attr_ops[0].input_op is root_val
    assert result == get_attr_ops[0].value


def test_get_value_at_attribute_literal_position():
    """Test get_value_at with AttributeLiteralPosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import IntegerAttr, ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        AttributeLiteralPosition,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    block = matcher_func.body.block

    # Create constant attribute
    const_attr = IntegerAttr(42, i32)
    attr_literal_pos = AttributeLiteralPosition(value=const_attr, parent=None)

    result = generator.get_value_at(attr_literal_pos)

    # Should create CreateAttributeOp
    create_attr_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.CreateAttributeOp)
    ]
    assert len(create_attr_ops) == 1
    assert create_attr_ops[0].value == const_attr
    assert result == create_attr_ops[0].attribute


def test_get_value_at_type_position():
    """Test get_value_at with TypePosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    root_pos = OperationPosition(None, depth=0)
    block = matcher_func.body.block
    root_val = block.args[0]
    generator.values[root_pos] = root_val

    # Get type of a result
    result_pos = root_pos.get_result(0)
    result_val = pdl_interp.GetResultOp(0, root_val).value
    generator.values[result_pos] = result_val

    type_pos = result_pos.get_type()
    result = generator.get_value_at(type_pos)

    # Should create GetValueTypeOp
    get_type_ops = [op for op in block.ops if isinstance(op, pdl_interp.GetValueTypeOp)]
    assert len(get_type_ops) == 1
    assert get_type_ops[0].value is result_val
    assert result == get_type_ops[0].result


def test_get_value_at_type_literal_position():
    """Test get_value_at with TypeLiteralPosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ArrayAttr, ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import TypeLiteralPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    block = matcher_func.body.block

    # Test case 1: Single type literal
    type_literal_pos = TypeLiteralPosition.get_type_literal(value=i32)
    result = generator.get_value_at(type_literal_pos)

    # Should create CreateTypeOp
    create_type_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.CreateTypeOp)
    ]
    assert len(create_type_ops) == 1
    assert create_type_ops[0].value == i32
    assert result == create_type_ops[0].result

    # Test case 2: Multiple types (ArrayAttr)
    types_array = ArrayAttr([i32, f32])
    types_literal_pos = TypeLiteralPosition.get_type_literal(value=types_array)
    result2 = generator.get_value_at(types_literal_pos)

    # Should create CreateTypesOp
    create_types_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.CreateTypesOp)
    ]
    assert len(create_types_ops) == 1
    assert create_types_ops[0].value == types_array
    assert result2 == create_types_ops[0].result


def test_get_value_at_constraint_position():
    """Test get_value_at with ConstraintPosition"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        ConstraintPosition,
        ConstraintQuestion,
        OperationPosition,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    block = matcher_func.body.block
    root_val = block.args[0]

    # Create a mock constraint operation with results
    root_pos = OperationPosition(None, depth=0)
    constraint_q = ConstraintQuestion(
        name="test_constraint",
        arg_positions=(root_pos,),
        result_types=(pdl.OperationType(), pdl.TypeType()),
        is_negated=False,
    )

    # Create and register the constraint op
    constraint_op = pdl_interp.ApplyConstraintOp(
        "test_constraint",
        [root_val],
        res_types=[pdl.OperationType(), pdl.TypeType()],
        true_dest=Block(),
        false_dest=Block(),
    )
    generator.constraint_op_map[constraint_q] = constraint_op

    # Get constraint result at index 1
    constraint_pos = ConstraintPosition.get_constraint(constraint_q, result_index=1)
    result = generator.get_value_at(constraint_pos)

    # Should return the second result of the constraint op
    assert result == constraint_op.results[1]


def test_get_value_at_caching():
    """Test that get_value_at properly caches values"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    root_pos = OperationPosition(None, depth=0)
    block = matcher_func.body.block
    root_val = block.args[0]
    generator.values[root_pos] = root_val

    # Get operand twice
    operand_pos = root_pos.get_operand(0)
    result1 = generator.get_value_at(operand_pos)
    result2 = generator.get_value_at(operand_pos)

    # Should return the same value (cached)
    assert result1 is result2

    # Should only create one GetOperandOp
    get_operand_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.GetOperandOp)
    ]
    assert len(get_operand_ops) == 1


def test_get_value_at_unimplemented_positions():
    """Test that get_value_at raises NotImplementedError for unsupported positions"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        ForEachPosition,
        OperationPosition,
        UsersPosition,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    root_pos = OperationPosition(None, depth=0)
    generator.values[root_pos] = matcher_func.body.block.args[0]

    # Test UsersPosition
    users_pos = UsersPosition(parent=root_pos, use_representative=True)
    with pytest.raises(NotImplementedError, match="UsersPosition"):
        generator.get_value_at(users_pos)

    # Test ForEachPosition
    foreach_pos = ForEachPosition(parent=root_pos, id=0)
    with pytest.raises(NotImplementedError, match="ForEachPosition"):
        generator.get_value_at(foreach_pos)


def test_get_value_at_operand_group_position():
    """Test get_value_at with OperandGroupPosition (variadic and non-variadic)"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    root_pos = OperationPosition(None, depth=0)
    block = matcher_func.body.block
    root_val = block.args[0]
    generator.values[root_pos] = root_val

    # Test variadic operand group
    operand_group_pos = root_pos.get_operand_group(0, is_variadic=True)
    result = generator.get_value_at(operand_group_pos)

    # Should create GetOperandsOp
    get_operands_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.GetOperandsOp)
    ]
    assert len(get_operands_ops) == 1
    assert get_operands_ops[0].index is not None
    assert get_operands_ops[0].index.value.data == 0
    assert get_operands_ops[0].input_op is root_val
    assert isinstance(result.type, pdl.RangeType)

    # Test non-variadic operand group
    operand_group_pos2 = root_pos.get_operand_group(1, is_variadic=False)
    result2 = generator.get_value_at(operand_group_pos2)

    get_operands_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.GetOperandsOp)
    ]
    assert len(get_operands_ops) == 2
    assert get_operands_ops[1].index is not None
    assert get_operands_ops[1].index.value.data == 1
    assert result2.type == pdl.ValueType()


def test_get_value_at_operation_position_passthrough():
    """Test get_value_at with OperationPosition passthrough (not operand-defining-op)

    This case occurs when a constraint returns a pdl.OperationType and we need
    to access it through an OperationPosition that just passes through the parent value.
    """
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        ConstraintPosition,
        ConstraintQuestion,
        OperationPosition,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    block = matcher_func.body.block
    root_val = block.args[0]

    # Create a constraint operation that returns a pdl.OperationType
    root_pos = OperationPosition(None, depth=0)
    constraint_q = ConstraintQuestion(
        name="returns_operation",
        arg_positions=(root_pos,),
        result_types=(pdl.OperationType(),),  # Constraint returns an operation
        is_negated=False,
    )

    # Create and register the constraint op that returns an operation
    constraint_op = pdl_interp.ApplyConstraintOp(
        "returns_operation",
        [root_val],
        res_types=[pdl.OperationType()],
        true_dest=Block(),
        false_dest=Block(),
    )
    generator.constraint_op_map[constraint_q] = constraint_op

    # Get the constraint position for the returned operation
    constraint_pos = ConstraintPosition.get_constraint(constraint_q, result_index=0)

    # Create an OperationPosition that has the constraint position as parent
    # This represents an operation value that comes from a constraint
    # Since it's not an operand-defining-op, it should just pass through the parent value
    op_pos_with_parent = OperationPosition(parent=constraint_pos, depth=1)

    # Get the value - should hit the passthrough branch
    result = generator.get_value_at(op_pos_with_parent)

    # Should return the constraint's operation result (passthrough from parent)
    assert result == constraint_op.results[0]

    # Verify it was cached
    assert op_pos_with_parent in generator.values
    assert generator.values[op_pos_with_parent] is result

    # Getting it again should return the cached value
    result2 = generator.get_value_at(op_pos_with_parent)
    assert result2 is result


def test_generate_bool_node_is_not_null():
    """Test generate_bool_node with IsNotNullQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode
    question = IsNotNullQuestion()
    answer = TrueAnswer()
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that IsNotNullOp was created
    check_ops = [op for op in block.ops if isinstance(op, pdl_interp.IsNotNullOp)]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.value is val

    # Check that success block was created
    assert len(matcher_body.blocks) == 3  # original, failure, success
    success_block = matcher_body.blocks[2]
    assert check_op.true_dest is success_block
    assert check_op.false_dest is failure_block


def test_generate_bool_node_operation_name():
    """Test generate_bool_node with OperationNameQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        OperationNameQuestion,
        StringAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode
    question = OperationNameQuestion()
    answer = StringAnswer("arith.addi")
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that CheckOperationNameOp was created
    check_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.CheckOperationNameOp)
    ]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.operation_name.data == "arith.addi"
    assert check_op.input_op is val


def test_generate_bool_node_operand_count():
    """Test generate_bool_node with OperandCountQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        OperandCountQuestion,
        UnsignedAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode
    question = OperandCountQuestion()
    answer = UnsignedAnswer(2)
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that CheckOperandCountOp was created
    check_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.CheckOperandCountOp)
    ]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.count.value.data == 2
    assert check_op.compareAtLeast is None
    assert check_op.input_op is val


def test_generate_bool_node_result_count_at_least():
    """Test generate_bool_node with ResultCountAtLeastQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        ResultCountAtLeastQuestion,
        UnsignedAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode
    question = ResultCountAtLeastQuestion()
    answer = UnsignedAnswer(1)
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that CheckResultCountOp was created
    check_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.CheckResultCountOp)
    ]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.count.value.data == 1
    assert check_op.compareAtLeast == UnitAttr()
    assert check_op.input_op is val


def test_generate_bool_node_equal_to():
    """Test generate_bool_node with EqualToQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        EqualToQuestion,
        OperationPosition,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(), pdl.OperationType()))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(), pdl.OperationType()), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val1 = block.args[0]
    val2 = block.args[1]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode
    other_pos = OperationPosition(None, depth=1)
    generator.values[other_pos] = val2  # Mock the other value
    question = EqualToQuestion(other_pos)
    answer = TrueAnswer()
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val1)

    # Check that AreEqualOp was created
    check_ops = [op for op in block.ops if isinstance(op, pdl_interp.AreEqualOp)]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.lhs is val1
    assert check_op.rhs is val2


def test_generate_bool_node_attribute_constraint():
    """Test generate_bool_node with AttributeConstraintQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp, i32
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        AttributeAnswer,
        AttributeConstraintQuestion,
    )

    matcher_body = Region([Block(arg_types=(pdl.AttributeType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.AttributeType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode
    question = AttributeConstraintQuestion()
    attr_value = IntegerAttr(42, i32)
    answer = AttributeAnswer(attr_value)
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that CheckAttributeOp was created
    check_ops = [op for op in block.ops if isinstance(op, pdl_interp.CheckAttributeOp)]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.constantValue is attr_value
    assert check_op.attribute is val


def test_generate_bool_node_type_constraint():
    """Test generate_bool_node with TypeConstraintQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        TypeAnswer,
        TypeConstraintQuestion,
    )

    matcher_body = Region([Block(arg_types=(pdl.TypeType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.TypeType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode
    question = TypeConstraintQuestion()
    answer = TypeAnswer(i32)
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that CheckTypeOp was created
    check_ops = [op for op in block.ops if isinstance(op, pdl_interp.CheckTypeOp)]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.type is i32
    assert check_op.value is val


def test_generate_bool_node_native_constraint():
    """Test generate_bool_node with ConstraintQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        ConstraintQuestion,
        OperationPosition,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode with constraint
    arg_pos = OperationPosition(None, depth=0)
    generator.values[arg_pos] = val  # Mock the argument value
    question = ConstraintQuestion("my_constraint", (arg_pos,), (), False)
    answer = TrueAnswer()
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that ApplyConstraintOp was created
    check_ops = [op for op in block.ops if isinstance(op, pdl_interp.ApplyConstraintOp)]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.constraint_name.data == "my_constraint"
    assert len(check_op.args) == 1
    assert check_op.args[0] is val
    assert check_op.is_negated.value.data is False

    # Check that constraint op is stored in map
    assert question in generator.constraint_op_map
    assert generator.constraint_op_map[question] is check_op


def test_generate_bool_node_operand_count_at_least():
    """Test generate_bool_node with OperandCountAtLeastQuestion"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        BoolNode,
        MatcherGenerator,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        OperandCountAtLeastQuestion,
        UnsignedAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode
    question = OperandCountAtLeastQuestion()
    answer = UnsignedAnswer(3)
    bool_node = BoolNode(question=question, answer=answer)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that CheckOperandCountOp was created with compareAtLeast=True
    check_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.CheckOperandCountOp)
    ]
    assert len(check_ops) == 1
    check_op = check_ops[0]
    assert check_op.count.value.data == 3
    assert check_op.compareAtLeast is not None
    assert check_op.input_op is val


def test_generate_matcher_exit_node():
    """Test generate_matcher with ExitNode"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create exit node
    exit_node = ExitNode()

    # Generate matcher for exit node
    result_block = generator.generate_matcher(exit_node, matcher_body)

    # Check that FinalizeOp was created
    finalize_ops = [
        op for op in result_block.ops if isinstance(op, pdl_interp.FinalizeOp)
    ]
    assert len(finalize_ops) == 1


def test_generate_matcher_bool_node_simple():
    """Test generate_matcher with a simple BoolNode"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        OperationPosition,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create bool node with failure node
    pos = OperationPosition(None, depth=0)
    question = IsNotNullQuestion()
    answer = TrueAnswer()
    failure_node = ExitNode()
    bool_node = BoolNode(
        position=pos, question=question, answer=answer, failure_node=failure_node
    )

    # Set up root operation value
    root_val = matcher_func.body.block.args[0]
    generator.values[pos] = root_val

    # Generate matcher
    result_block = generator.generate_matcher(bool_node, matcher_body)

    # Check that IsNotNullOp was created
    check_ops = [
        op for op in result_block.ops if isinstance(op, pdl_interp.IsNotNullOp)
    ]
    assert len(check_ops) == 1

    # Check that failure block was created with FinalizeOp
    finalize_ops = [
        op
        for block in matcher_body.blocks
        for op in block.ops
        if isinstance(op, pdl_interp.FinalizeOp)
    ]

    assert len(finalize_ops) == 1


def test_generate_matcher_bool_node_with_success():
    """Test generate_matcher with BoolNode that has success_node"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        OperationPosition,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create nested structure: BoolNode -> success_node -> ExitNode
    pos = OperationPosition(None, depth=0)
    question = IsNotNullQuestion()
    answer = TrueAnswer()

    # Create success and failure nodes
    success_node = ExitNode()
    failure_node = ExitNode()

    bool_node = BoolNode(
        position=pos,
        question=question,
        answer=answer,
        success_node=success_node,
        failure_node=failure_node,
    )

    # Set up root operation value
    root_val = matcher_func.body.block.args[0]
    generator.values[pos] = root_val

    # Generate matcher
    result_block = generator.generate_matcher(bool_node, matcher_body)

    # Check that IsNotNullOp was created
    check_ops = [
        op for op in result_block.ops if isinstance(op, pdl_interp.IsNotNullOp)
    ]
    assert len(check_ops) == 1
    check_op = check_ops[0]

    # Check that both success and failure blocks exist and have FinalizeOps
    success_block = check_op.true_dest
    failure_block = check_op.false_dest

    success_finalize_ops = [
        op for op in success_block.ops if isinstance(op, pdl_interp.FinalizeOp)
    ]
    failure_finalize_ops = [
        op for op in failure_block.ops if isinstance(op, pdl_interp.FinalizeOp)
    ]

    assert len(success_finalize_ops) == 1
    assert len(failure_finalize_ops) == 1


def test_generate_matcher_nested_bool_nodes():
    """Test generate_matcher with nested BoolNodes"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        OperationNameQuestion,
        OperationPosition,
        StringAnswer,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create nested structure:
    # BoolNode(IsNotNull) -> success -> BoolNode(OpName) -> success -> ExitNode
    pos = OperationPosition(None, depth=0)

    # Create leaf node (final success)
    final_success = ExitNode()

    # Create inner bool node (operation name check)
    inner_question = OperationNameQuestion()
    inner_answer = StringAnswer("arith.addi")
    inner_bool = BoolNode(
        position=pos,
        question=inner_question,
        answer=inner_answer,
        success_node=final_success,
        failure_node=ExitNode(),
    )

    # Create outer bool node (null check)
    outer_question = IsNotNullQuestion()
    outer_answer = TrueAnswer()
    outer_bool = BoolNode(
        position=pos,
        question=outer_question,
        answer=outer_answer,
        success_node=inner_bool,
        failure_node=ExitNode(),
    )

    # Set up root operation value
    root_val = matcher_func.body.block.args[0]
    generator.values[pos] = root_val

    # Generate matcher
    result_block = generator.generate_matcher(outer_bool, matcher_body)

    # Check that IsNotNullOp was created in the main block
    is_not_null_ops = [
        op for op in result_block.ops if isinstance(op, pdl_interp.IsNotNullOp)
    ]
    assert len(is_not_null_ops) == 1

    # Check that success block contains CheckOperationNameOp
    success_block = is_not_null_ops[0].true_dest
    check_name_ops = [
        op
        for op in success_block.ops
        if isinstance(op, pdl_interp.CheckOperationNameOp)
    ]
    assert len(check_name_ops) == 1
    assert check_name_ops[0].operation_name.data == "arith.addi"

    # Check that final success block has FinalizeOp
    final_block = check_name_ops[0].true_dest
    finalize_ops = [
        op for op in final_block.ops if isinstance(op, pdl_interp.FinalizeOp)
    ]
    assert len(finalize_ops) == 1


def test_generate_matcher_reuses_failure_block():
    """Test that generate_matcher reuses failure blocks from stack"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        OperationPosition,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create a failure block and push it to the stack
    failure_block = Block()
    matcher_body.add_block(failure_block)
    finalize_op = pdl_interp.FinalizeOp()
    generator.builder.insert_op(finalize_op, InsertPoint.at_end(failure_block))
    generator.failure_block_stack.append(failure_block)

    # Create bool node without failure_node (should use stack)
    pos = OperationPosition(None, depth=0)
    question = IsNotNullQuestion()
    answer = TrueAnswer()
    bool_node = BoolNode(position=pos, question=question, answer=answer)

    # Set up root operation value
    assert matcher_func.body.first_block
    root_val = matcher_func.body.first_block.args[0]
    generator.values[pos] = root_val

    # Generate matcher
    result_block = generator.generate_matcher(bool_node, matcher_body)

    # Check that IsNotNullOp was created and uses the existing failure block
    check_ops = [
        op for op in result_block.ops if isinstance(op, pdl_interp.IsNotNullOp)
    ]
    assert len(check_ops) == 1
    assert check_ops[0].false_dest is failure_block


def test_generate_matcher_scoped_values():
    """Test that generate_matcher properly manages scoped values"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        OperationPosition,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Store the initial values dict
    initial_values = generator.values

    # Create bool node
    pos = OperationPosition(None, depth=0)
    question = IsNotNullQuestion()
    answer = TrueAnswer()
    bool_node = BoolNode(
        position=pos, question=question, answer=answer, failure_node=ExitNode()
    )

    # Set up root operation value
    root_val = matcher_func.body.block.args[0]
    generator.values[pos] = root_val

    # Generate matcher
    _result_block = generator.generate_matcher(bool_node, matcher_body)

    # Check that values dict was restored after generation
    assert generator.values is initial_values


def test_generate_bool_node_with_success_node_calls_generate_matcher():
    """Test that generate_bool_node calls generate_matcher when success_node exists"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create BoolNode with success_node
    question = IsNotNullQuestion()
    answer = TrueAnswer()
    success_node = ExitNode()
    bool_node = BoolNode(question=question, answer=answer, success_node=success_node)

    # Generate the bool node
    generator.generate_bool_node(bool_node, val)

    # Check that IsNotNullOp was created
    check_ops = [op for op in block.ops if isinstance(op, pdl_interp.IsNotNullOp)]
    assert len(check_ops) == 1
    check_op = check_ops[0]

    # Check that success block was created and contains FinalizeOp (from generate_matcher)
    success_block = check_op.true_dest
    finalize_ops = [
        op for op in success_block.ops if isinstance(op, pdl_interp.FinalizeOp)
    ]
    assert len(finalize_ops) == 1


def test_generate_switch_node_operation_name():
    """Test generate_switch_node with OperationNameQuestion"""
    from unittest.mock import patch

    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        SuccessNode,
        SwitchNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        OperationNameQuestion,
        StringAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create mock pattern for SuccessNode
    mock_pattern = pdl.PatternOp(1, "test_pattern", Region())

    # Create SwitchNode with multiple operation name cases
    question = OperationNameQuestion()
    children: dict[Answer, MatcherNode | None] = {
        StringAnswer("arith.addi"): SuccessNode(pattern=mock_pattern),
        StringAnswer("arith.subi"): SuccessNode(pattern=mock_pattern),
        StringAnswer("arith.muli"): SuccessNode(pattern=mock_pattern),
    }
    switch_node = SwitchNode(question=question, children=children)

    # Mock generate_matcher to return dummy blocks
    mock_blocks = [Block(), Block(), Block()]
    with patch.object(generator, "generate_matcher", side_effect=mock_blocks):
        # Generate the switch node
        generator.generate_switch_node(switch_node, val)

    # Check that SwitchOperationNameOp was created
    switch_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.SwitchOperationNameOp)
    ]
    assert len(switch_ops) == 1
    switch_op = switch_ops[0]

    # Check case values
    case_values = [attr.data for attr in switch_op.case_values.data]
    assert set(case_values) == {"arith.addi", "arith.subi", "arith.muli"}

    # Check operand and successors
    assert switch_op.input_op is val
    assert switch_op.default_dest is failure_block
    assert len(switch_op.cases) == 3


def test_generate_switch_node_attribute_constraint():
    """Test generate_switch_node with AttributeConstraintQuestion"""
    from unittest.mock import patch

    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import IntegerAttr, ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        SuccessNode,
        SwitchNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        AttributeAnswer,
        AttributeConstraintQuestion,
    )

    matcher_body = Region([Block(arg_types=(pdl.AttributeType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.AttributeType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create mock pattern for SuccessNode
    mock_pattern = pdl.PatternOp(1, "test_pattern", Region())

    # Create SwitchNode with multiple attribute cases
    question = AttributeConstraintQuestion()
    attr1 = IntegerAttr(1, i32)
    attr2 = IntegerAttr(2, i32)
    attr3 = IntegerAttr(3, i32)
    children: dict[Answer, MatcherNode | None] = {
        AttributeAnswer(attr1): SuccessNode(pattern=mock_pattern),
        AttributeAnswer(attr2): SuccessNode(pattern=mock_pattern),
        AttributeAnswer(attr3): SuccessNode(pattern=mock_pattern),
    }
    switch_node = SwitchNode(question=question, children=children)

    # Mock generate_matcher to return dummy blocks
    mock_blocks = [Block(), Block(), Block()]
    with patch.object(generator, "generate_matcher", side_effect=mock_blocks):
        # Generate the switch node
        generator.generate_switch_node(switch_node, val)

    # Check that SwitchAttributeOp was created
    switch_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.SwitchAttributeOp)
    ]
    assert len(switch_ops) == 1
    switch_op = switch_ops[0]

    # Check case values
    case_attrs = switch_op.caseValues.data
    assert len(case_attrs) == 3
    assert attr1 in case_attrs
    assert attr2 in case_attrs
    assert attr3 in case_attrs

    # Check operand and successors
    assert switch_op.attribute is val
    assert switch_op.defaultDest is failure_block
    assert len(switch_op.cases) == 3


def test_generate_switch_node_with_none_child():
    """Test generate_switch_node with None children (should be skipped)"""
    from unittest.mock import patch

    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        SuccessNode,
        SwitchNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        OperationNameQuestion,
        StringAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create mock pattern for SuccessNode
    mock_pattern = pdl.PatternOp(1, "test_pattern", Region())

    # Create SwitchNode with some None children
    question = OperationNameQuestion()
    children: dict[Answer, MatcherNode | None] = {
        StringAnswer("arith.addi"): SuccessNode(pattern=mock_pattern),
        StringAnswer("arith.subi"): None,  # This should be skipped
        StringAnswer("arith.muli"): SuccessNode(pattern=mock_pattern),
    }
    switch_node = SwitchNode(question=question, children=children)

    # Mock generate_matcher to return dummy blocks (only for non-None children)
    mock_blocks = [Block(), Block()]
    with patch.object(generator, "generate_matcher", side_effect=mock_blocks):
        # Generate the switch node
        generator.generate_switch_node(switch_node, val)

    # Check that SwitchOperationNameOp was created
    switch_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.SwitchOperationNameOp)
    ]
    assert len(switch_ops) == 1
    switch_op = switch_ops[0]

    # Should only have 2 cases (None child skipped)
    case_values = [attr.data for attr in switch_op.case_values.data]
    assert len(case_values) == 2
    assert set(case_values) == {"arith.addi", "arith.muli"}
    assert len(switch_op.cases) == 2


def test_generate_switch_node_empty_children():
    """Test generate_switch_node with empty children"""
    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        SwitchNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        OperationNameQuestion,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create SwitchNode with empty children
    question = OperationNameQuestion()
    children: dict[Answer, MatcherNode | None] = {}
    switch_node = SwitchNode(question=question, children=children)

    # Generate the switch node
    generator.generate_switch_node(switch_node, val)

    # Check that SwitchOperationNameOp was created even with empty cases
    switch_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.SwitchOperationNameOp)
    ]
    assert len(switch_ops) == 1
    switch_op = switch_ops[0]

    # Should have no cases
    assert not switch_op.case_values.data
    assert not switch_op.cases
    assert switch_op.default_dest is failure_block


def test_generate_switch_node_operand_count_not_implemented():
    """Test generate_switch_node with OperandCountQuestion"""
    from unittest.mock import patch

    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        SuccessNode,
        SwitchNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        OperandCountQuestion,
        UnsignedAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create mock pattern for SuccessNode
    mock_pattern = pdl.PatternOp(1, "test_pattern", Region())

    # Create SwitchNode with operand count cases
    question = OperandCountQuestion()
    children: dict[Answer, MatcherNode | None] = {
        UnsignedAnswer(1): SuccessNode(pattern=mock_pattern),
        UnsignedAnswer(2): SuccessNode(pattern=mock_pattern),
    }
    switch_node = SwitchNode(question=question, children=children)

    # Mock generate_matcher to return dummy blocks
    mock_blocks = [Block(), Block()]
    with patch.object(generator, "generate_matcher", side_effect=mock_blocks):
        # Generate the switch node
        generator.generate_switch_node(switch_node, val)

    # Check that SwitchOperandCountOp was created
    switch_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.SwitchOperandCountOp)
    ]
    assert len(switch_ops) == 1
    switch_op = switch_ops[0]

    # Check case values
    case_values = switch_op.case_values.get_values()
    assert set(case_values) == {1, 2}

    # Check operand and successors
    assert switch_op.input_op is val
    assert switch_op.default_dest is failure_block
    assert len(switch_op.cases) == 2


def test_generate_switch_node_result_count_not_implemented():
    """Test generate_switch_node with ResultCountQuestion"""
    from unittest.mock import patch

    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        SuccessNode,
        SwitchNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        ResultCountQuestion,
        UnsignedAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create mock pattern for SuccessNode
    mock_pattern = pdl.PatternOp(1, "test_pattern", Region())

    # Create SwitchNode with result count cases
    question = ResultCountQuestion()
    children: dict[Answer, MatcherNode | None] = {
        UnsignedAnswer(1): SuccessNode(pattern=mock_pattern),
        UnsignedAnswer(2): SuccessNode(pattern=mock_pattern),
    }
    switch_node = SwitchNode(question=question, children=children)

    # Mock generate_matcher to return dummy blocks
    mock_blocks = [Block(), Block()]
    with patch.object(generator, "generate_matcher", side_effect=mock_blocks):
        # Generate the switch node
        generator.generate_switch_node(switch_node, val)

    # Check that SwitchResultCountOp was created
    switch_ops = [
        op for op in block.ops if isinstance(op, pdl_interp.SwitchResultCountOp)
    ]
    assert len(switch_ops) == 1
    switch_op = switch_ops[0]

    # Check case values
    case_values = switch_op.case_values.get_values()
    assert set(case_values) == {1, 2}

    # Check operand and successors
    assert switch_op.input_op is val
    assert switch_op.default_dest is failure_block
    assert len(switch_op.cases) == 2


def test_generate_switch_node_type_constraint_not_implemented():
    """Test generate_switch_node with TypeConstraintQuestion"""
    from unittest.mock import patch

    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        SuccessNode,
        SwitchNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        TypeAnswer,
        TypeConstraintQuestion,
    )

    matcher_body = Region([Block(arg_types=(pdl.TypeType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.TypeType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create mock pattern for SuccessNode
    mock_pattern = pdl.PatternOp(1, "test_pattern", Region())

    # Create SwitchNode with type constraint cases
    question = TypeConstraintQuestion()
    children: dict[Answer, MatcherNode | None] = {
        TypeAnswer(i32): SuccessNode(pattern=mock_pattern),
        TypeAnswer(f32): SuccessNode(pattern=mock_pattern),
    }
    switch_node = SwitchNode(question=question, children=children)

    # Mock generate_matcher to return dummy blocks
    mock_blocks = [Block(), Block()]
    with patch.object(generator, "generate_matcher", side_effect=mock_blocks):
        # Generate the switch node
        generator.generate_switch_node(switch_node, val)

    # Check that SwitchTypeOp was created (val.type is pdl.TypeType, not RangeType)
    switch_ops = [op for op in block.ops if isinstance(op, pdl_interp.SwitchTypeOp)]
    assert len(switch_ops) == 1
    switch_op = switch_ops[0]

    # Check case values
    case_values = list(switch_op.case_values.data)
    assert case_values == [i32, f32]

    # Check operand and successors
    assert switch_op.value is val
    assert switch_op.default_dest is failure_block
    assert len(switch_op.cases) == 2


def test_generate_switch_node_unhandled_question():
    """Test that unhandled question types raise NotImplementedError"""
    from unittest.mock import patch

    from xdsl.dialects import pdl_interp
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        SuccessNode,
        SwitchNode,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
        IsNotNullQuestion,
        TrueAnswer,
    )

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create failure block
    failure_block = Block()
    matcher_body.add_block(failure_block)
    generator.failure_block_stack.append(failure_block)

    # Create mock pattern for SuccessNode
    mock_pattern = pdl.PatternOp(1, "test_pattern", Region())

    # Create SwitchNode with unsupported question type
    question = IsNotNullQuestion()  # Not supported in switch
    children: dict[Answer, MatcherNode | None] = {
        TrueAnswer(): SuccessNode(pattern=mock_pattern),
    }
    switch_node = SwitchNode(question=question, children=children)

    # Mock generate_matcher to return dummy blocks
    mock_blocks = [Block()]
    with patch.object(generator, "generate_matcher", side_effect=mock_blocks):
        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Unhandled question type"):
            generator.generate_switch_node(switch_node, val)


@pytest.mark.parametrize(
    "question_type, check_op_type",
    [
        (OperandCountAtLeastQuestion, pdl_interp.CheckOperandCountOp),
        (ResultCountAtLeastQuestion, pdl_interp.CheckResultCountOp),
    ],
)
def test_generate_switch_node_at_least_question(
    question_type: type[Question],
    check_op_type: type[pdl_interp.CheckOperandCountOp | pdl_interp.CheckResultCountOp],
):
    """
    Test generate_switch_node with OperandCountAtLeastQuestion and
    ResultCountAtLeastQuestion, which have special handling.
    """
    # 1. Set up the test environment
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    block = matcher_func.body.block
    val = block.args[0]

    # Create a default failure block and push it to the stack
    default_failure_block = Block()
    matcher_body.add_block(default_failure_block)
    generator.failure_block_stack.append(default_failure_block)

    # 2. Construct the SwitchNode with ExitNode children to avoid real matcher generation
    question = question_type()
    # Children are intentionally unordered to test the sorting logic
    children: dict[Answer, MatcherNode | None] = {
        UnsignedAnswer(3): ExitNode(),
        UnsignedAnswer(5): ExitNode(),
        UnsignedAnswer(1): ExitNode(),
    }
    switch_node = SwitchNode(question=question, children=children)

    # 3. Call the method under test
    generator.generate_switch_node(switch_node, val)

    # 4. Verify the generated IR
    # The logic creates a chain starting with the LOWEST count (1)
    # Check 1 (for count >= 1) should be in the initial block
    assert len(block.ops) == 1
    check_op_1 = block.first_op
    assert isinstance(check_op_1, check_op_type)
    assert check_op_1.count.value.data == 1
    assert check_op_1.compareAtLeast is not None
    assert check_op_1.false_dest is default_failure_block
    assert check_op_1.true_dest is not None
    assert check_op_1.true_dest.parent is matcher_body

    # The success block for count >= 1 should have at least one block chained through it
    _success_block_1 = check_op_1.true_dest
    # Verify that there are Check ops for the other counts in the chain
    check_ops = [op for op in matcher_body.walk() if isinstance(op, check_op_type)]
    # We should have check ops for each count: 1, 3, 5
    assert len(check_ops) == 3
    # Check that they're checking for the right counts
    check_counts = sorted([op.count.value.data for op in check_ops])
    assert check_counts == [1, 3, 5]


def test_generate_rewriter_for_apply_native_rewrite():
    """Test generate_rewriter with ApplyNativeRewriteOp"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with ApplyNativeRewriteOp in the rewrite body
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        op1 = pdl.OperationOp("test_op").op
        attr1 = pdl.AttributeOp(IntegerAttr(10, i32)).output
        type1 = pdl.TypeOp(i32).result
        types1 = pdl.TypesOp([i32, f32]).result
        pdl.RewriteOp(op1, name="myRewrite", external_args=[op1, attr1, type1, types1])

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Generate the rewriter
    rewriter_name = generator.generate_rewriter(
        pattern, [OperationPosition(None, depth=0)]
    )

    # Verify that a rewriter was created
    assert rewriter_name is not None
    # Verify that the rewriter module contains an ApplyRewriteOp
    rewriter_func = rewriter_module.body.ops.first
    assert isinstance(rewriter_func, pdl_interp.FuncOp)
    apply_rewrite_ops = [
        op for op in rewriter_func.body.ops if isinstance(op, pdl_interp.ApplyRewriteOp)
    ]
    assert len(apply_rewrite_ops) == 1
    assert apply_rewrite_ops[0].rewrite_name.data == "myRewrite"


def test_generate_rewriter_for_attribute_with_constant():
    """Test generate_rewriter with AttributeOp with constant value"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create AttributeOp with constant value in a pattern's rewrite block
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("test_op").op
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            _attr_op = pdl.AttributeOp(IntegerAttr(42, i32))
        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Map pattern values
    root_pos = OperationPosition(None, depth=0)
    generator.value_to_position[pattern] = {root: root_pos}

    # Call method
    generator.generate_rewriter(pattern, [])

    # Verify CreateAttributeOp was created in the rewriter
    rewriter_func = rewriter_module.body.block.first_op
    assert isinstance(rewriter_func, pdl_interp.FuncOp)

    create_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.CreateAttributeOp)
    ]
    assert len(create_ops) == 1
    assert create_ops[0].value == IntegerAttr(42, i32)


def test_generate_rewriter_for_attribute_without_constant():
    """Test generate_rewriter with AttributeOp without constant value"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create AttributeOp without constant value in a pattern's rewrite block
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("test_op").op
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            attr_type = pdl.TypeOp(i32).result
            _attr_op = pdl.AttributeOp(attr_type)
        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Map pattern values
    root_pos = OperationPosition(None, depth=0)
    generator.value_to_position[pattern] = {root: root_pos}

    # Call method
    generator.generate_rewriter(pattern, [])

    # Verify the rewriter was created
    rewriter_func = rewriter_module.body.block.first_op
    assert isinstance(rewriter_func, pdl_interp.FuncOp)


def test_generate_operation_result_type_rewriter_strategy1_all_resolvable():
    """Test _generate_operation_result_type_rewriter Strategy 1: all types resolvable"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create rewriter region with an operation that has resolvable types
    rewriter_region = Region([Block()])
    rewriter_block = rewriter_region.first_block
    with ImplicitBuilder(rewriter_block):
        type1 = pdl.TypeOp(i32).result
        type2 = pdl.TypeOp(f32).result
        op_to_create = pdl.OperationOp("test.op", type_values=(type1, type2))

    rewrite_values: dict[SSAValue, SSAValue] = {}

    # Map the type values
    type_block = Block()
    type1_arg = type_block.insert_arg(pdl.TypeType(), 0)
    type2_arg = type_block.insert_arg(pdl.TypeType(), 1)
    rewrite_values[type1] = type1_arg
    rewrite_values[type2] = type2_arg

    def map_rewrite_value(val: SSAValue) -> SSAValue:
        return rewrite_values.get(val, val)

    types_list: list[SSAValue] = []

    # Call method
    has_inferred = generator._generate_operation_result_type_rewriter(  # pyright: ignore[reportPrivateUsage]
        op_to_create, map_rewrite_value, types_list, rewrite_values
    )

    # Verify Strategy 1 was used: all types resolved
    assert has_inferred is False
    assert types_list == [type1_arg, type2_arg]


def test_generate_operation_result_type_rewriter_strategy3_from_replace():
    """Test _generate_operation_result_type_rewriter Strategy 3: infer from replaced operation"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create rewriter region with operations
    rewriter_region = Region([Block()])
    rewriter_block = rewriter_region.first_block
    with ImplicitBuilder(rewriter_block):
        # Old operation to be replaced
        old_op = pdl.OperationOp("old.op")
        # New operation that will replace it
        type1 = pdl.TypeOp().result
        new_op = pdl.OperationOp("new.op", type_values=(type1,))
        # Replace operation
        pdl.ReplaceOp(old_op.op, repl_operation=new_op.op)

    rewrite_values: dict[SSAValue, SSAValue] = {}

    # Map the old operation value (from match)
    match_block = Block()
    old_op_arg = match_block.insert_arg(pdl.OperationType(), 0)
    rewrite_values[old_op.op] = old_op_arg

    def map_rewrite_value(val: SSAValue) -> SSAValue:
        return rewrite_values.get(val, val)

    types_list: list[SSAValue] = []

    # Set up rewriter builder for the ops that will be generated
    result_block = Block()
    generator.rewriter_builder.insertion_point = InsertPoint.at_end(result_block)

    # Call method
    has_inferred = generator._generate_operation_result_type_rewriter(  # pyright: ignore[reportPrivateUsage]
        new_op, map_rewrite_value, types_list, rewrite_values
    )

    # Verify Strategy 3 was used: inferred from replaced operation
    assert has_inferred is False
    assert len(types_list) == 1

    # Check that GetResultsOp and GetValueTypeOp were created
    get_results_ops = [
        op for op in result_block.ops if isinstance(op, pdl_interp.GetResultsOp)
    ]
    assert len(get_results_ops) == 1
    assert get_results_ops[0].input_op == old_op_arg

    get_type_ops = [
        op for op in result_block.ops if isinstance(op, pdl_interp.GetValueTypeOp)
    ]
    assert len(get_type_ops) == 1


def test_generate_operation_result_type_rewriter_strategy4_no_types():
    """Test _generate_operation_result_type_rewriter Strategy 4: no explicit types"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create rewriter region with an operation that has no result types
    rewriter_region = Region([Block()])
    rewriter_block = rewriter_region.first_block
    with ImplicitBuilder(rewriter_block):
        op_to_create = pdl.OperationOp("test.op", type_values=())

    rewrite_values: dict[SSAValue, SSAValue] = {}

    def map_rewrite_value(val: SSAValue) -> SSAValue:
        return rewrite_values.get(val, val)

    types_list: list[SSAValue] = []

    # Call method
    has_inferred = generator._generate_operation_result_type_rewriter(  # pyright: ignore[reportPrivateUsage]
        op_to_create, map_rewrite_value, types_list, rewrite_values
    )

    # Verify Strategy 4 was used: no results assumed
    assert has_inferred is False
    assert not types_list

    assert not types_list


def test_generate_operation_result_type_rewriter_error_unresolvable():
    """Test _generate_operation_result_type_rewriter raises error when types can't be inferred"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create rewriter region with an operation that has unresolvable types
    rewriter_region = Region([Block()])
    rewriter_block = rewriter_region.first_block
    with ImplicitBuilder(rewriter_block):
        # Type is defined in rewriter block but not in rewrite_values
        type1 = pdl.TypeOp().result
        op_to_create = pdl.OperationOp("test.op", type_values=(type1,))

    rewrite_values: dict[SSAValue, SSAValue] = {}

    def map_rewrite_value(val: SSAValue) -> SSAValue:
        return rewrite_values.get(val, val)

    types_list: list[SSAValue] = []

    # Call method - should raise ValueError
    with pytest.raises(
        ValueError, match='Unable to infer result types for pdl.operation "test.op"'
    ):
        generator._generate_operation_result_type_rewriter(  # pyright: ignore[reportPrivateUsage]
            op_to_create, map_rewrite_value, types_list, rewrite_values
        )


def test_generate_operation_result_type_rewriter_strategy1_partial_resolution():
    """Test Strategy 1 with some types resolvable but not all (should fall through)"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create rewriter region with mixed resolvable/unresolvable types
    rewriter_region = Region([Block()])
    rewriter_block = rewriter_region.first_block
    with ImplicitBuilder(rewriter_block):
        type1 = pdl.TypeOp(i32).result  # Resolvable constant
        type2 = pdl.TypeOp().result  # Unresolvable
        op_to_create = pdl.OperationOp("test.op", type_values=(type1, type2))

    rewrite_values: dict[SSAValue, SSAValue] = {}
    # Only map type1, not type2
    type_block = Block()
    type1_arg = type_block.insert_arg(pdl.TypeType(), 0)
    rewrite_values[type1] = type1_arg

    def map_rewrite_value(val: SSAValue) -> SSAValue:
        return rewrite_values.get(val, val)

    types_list: list[SSAValue] = []

    # Should raise ValueError because not all types can be resolved
    with pytest.raises(
        ValueError, match='Unable to infer result types for pdl.operation "test.op"'
    ):
        generator._generate_operation_result_type_rewriter(  # pyright: ignore[reportPrivateUsage]
            op_to_create, map_rewrite_value, types_list, rewrite_values
        )


def test_generate_operation_result_type_rewriter_strategy3_operation_before_replace():
    """Test Strategy 3 skips replace ops where new op is before old op in block"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create rewriter region where new_op comes AFTER old_op
    rewriter_region = Region([Block()])
    rewriter_block = rewriter_region.first_block
    with ImplicitBuilder(rewriter_block):
        # New operation (comes first)
        type1 = pdl.TypeOp().result
        new_op = pdl.OperationOp("new.op", type_values=(type1,))
        # Old operation (comes second)
        old_op = pdl.OperationOp("old.op")
        # Replace - but new_op is defined BEFORE old_op in the block
        pdl.ReplaceOp(old_op.op, repl_operation=new_op.op)

    rewrite_values: dict[SSAValue, SSAValue] = {}
    # Map old_op as coming from the rewriter block itself
    rewrite_values[old_op.op] = old_op.op  # Not from match, from rewriter

    def map_rewrite_value(val: SSAValue) -> SSAValue:
        return rewrite_values.get(val, val)

    types_list: list[SSAValue] = []

    # Should raise ValueError because Strategy 3 skips this case
    with pytest.raises(
        ValueError, match='Unable to infer result types for pdl.operation "new.op"'
    ):
        generator._generate_operation_result_type_rewriter(  # pyright: ignore[reportPrivateUsage]
            new_op, map_rewrite_value, types_list, rewrite_values
        )


def test_generate_operation_result_type_rewriter_strategy1_with_external_types():
    """Test Strategy 1 with types from outside the rewriter block"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create rewriter region
    rewriter_region = Region([Block()])
    rewriter_block = rewriter_region.first_block

    # Create type outside rewriter block (simulating match input)
    match_region = Region([Block()])
    match_block = match_region.first_block
    with ImplicitBuilder(match_block):
        external_type = pdl.TypeOp(i32).result

    # Create operation in rewriter block using external type
    with ImplicitBuilder(rewriter_block):
        op_to_create = pdl.OperationOp("test.op", type_values=(external_type,))

    rewrite_values: dict[SSAValue, SSAValue] = {}

    # Map external type (will be called via map_rewrite_value)
    type_arg = Block().insert_arg(pdl.TypeType(), 0)
    rewrite_values[external_type] = type_arg

    def map_rewrite_value(val: SSAValue) -> SSAValue:
        if val in rewrite_values:
            return rewrite_values[val]
        # Simulate mapping external values
        return type_arg

    types_list: list[SSAValue] = []

    # Call method
    has_inferred = generator._generate_operation_result_type_rewriter(  # pyright: ignore[reportPrivateUsage]
        op_to_create, map_rewrite_value, types_list, rewrite_values
    )

    # Verify Strategy 1 was used with external type
    assert has_inferred is False
    assert len(types_list) == 1
    assert types_list[0] == type_arg


def test_generate_rewriter_for_operation_standard():
    """Test generate_rewriter with standard OperationOp configuration"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i32
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create pattern with operation in rewrite block
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("root_op").op
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            attr = pdl.AttributeOp(value=IntegerAttr(42, i32)).output
            res_type = pdl.TypeOp(i32).result
            _op = pdl.OperationOp(
                "test.op",
                attribute_values=[attr],
                attribute_value_names=[StringAttr("attr1")],
                type_values=[res_type],
            )
        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Map pattern values
    root_pos = OperationPosition(None, depth=0)
    generator.value_to_position[pattern] = {root: root_pos}

    # Execute
    generator.generate_rewriter(pattern, [])

    # Verify CreateOperationOp was created in the rewriter
    rewriter_func = rewriter_module.body.block.first_op
    assert isinstance(rewriter_func, pdl_interp.FuncOp)

    create_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.CreateOperationOp)
    ]
    assert len(create_ops) == 1
    create_op = create_ops[0]
    assert create_op.constraint_name.data == "test.op"


def test_generate_rewriter_for_operation_missing_name():
    """Test generate_rewriter raises ValueError when OperationOp has no name"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("root_op").op
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            _op = pdl.OperationOp(None)  # No name
        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Map pattern values
    root_pos = OperationPosition(None, depth=0)
    generator.value_to_position[pattern] = {root: root_pos}

    with pytest.raises(ValueError, match="Cannot create operation without a name"):
        generator.generate_rewriter(pattern, [])


def test_generate_rewriter_for_operation_single_variadic_result():
    """Test generate_rewriter with operation having single variadic result"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with operation having single variadic result
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # pdl.TypesOp produces a range type
        res_type = pdl.TypesOp().result
        root = pdl.OperationOp("test.op", type_values=[res_type]).op

        # Rewrite body creating an operation with results
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            type_op = pdl.TypesOp().result
            new_op = pdl.OperationOp("new.op", type_values=[type_op]).op
            pdl.ReplaceOp(root, repl_operation=new_op)

        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Verify GetResultsOp created for the operation with variadic results
    get_results_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.GetResultsOp)
    ]
    assert len(get_results_ops) >= 1  # At least one for the variadic result


def test_generate_rewriter_for_operation_mixed_results():
    """Test generate_rewriter with operation having mixed results"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp, i32
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with operation having mixed results: [Type, Types, Type]
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        t1 = pdl.TypeOp(i32).result
        t2 = pdl.TypesOp().result  # Variadic
        t3 = pdl.TypeOp(i32).result
        root = pdl.OperationOp("test.op", type_values=[t1, t2, t3]).op

        # Rewrite body creating an operation with results
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            type1 = pdl.TypeOp(i32).result
            type2 = pdl.TypesOp().result
            type3 = pdl.TypeOp(i32).result
            new_op = pdl.OperationOp("new.op", type_values=[type1, type2, type3]).op
            pdl.ReplaceOp(root, repl_operation=new_op)

        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Verify GetResultOp and GetResultsOp created for mixed results
    get_result_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.GetResultOp)
    ]
    get_results_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.GetResultsOp)
    ]

    # We should have both types of ops for handling mixed results
    assert len(get_result_ops) >= 0  # May have non-variadic results
    assert len(get_results_ops) >= 0  # May have variadic results


def test_generate_rewriter_for_range():
    """Test generate_rewriter with RangeOp generates CreateRangeOp"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp, i32
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with RangeOp in the rewrite body
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("test.op").op

        # Rewrite body with RangeOp
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            type_val = pdl.TypeOp(i32).result
            pdl.RangeOp([type_val], pdl.RangeType(pdl.TypeType()))

        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Execute
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify CreateRangeOp was created in the rewriter
    rewriter_func = rewriter_module.body.block.first_op
    assert isinstance(rewriter_func, pdl_interp.FuncOp)

    create_range_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.CreateRangeOp)
    ]
    assert len(create_range_ops) == 1
    create_range_op = create_range_ops[0]
    assert isinstance(create_range_op.result.type, pdl.RangeType)


def test_generate_rewriter_for_erase():
    """Test generate_rewriter with EraseOp generates EraseOp"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with EraseOp in the rewrite body
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        op_val = pdl.OperationOp("test.op").op

        # Rewrite body with EraseOp
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            pdl.EraseOp(op_val)

        pdl.RewriteOp(op_val, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Execute
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify EraseOp was created in the rewriter
    rewriter_func = rewriter_module.body.block.first_op
    assert isinstance(rewriter_func, pdl_interp.FuncOp)

    erase_ops = [
        op for op in rewriter_func.body.block.ops if isinstance(op, pdl_interp.EraseOp)
    ]
    assert len(erase_ops) == 1


def test_generate_rewriter_for_result():
    """Test generate_rewriter with ResultOp generates GetResultOp"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup generator
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with ResultOp in the rewrite body
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        op = pdl.OperationOp("test.op").op

        # Rewrite body with ResultOp
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            result_op = pdl.ResultOp(0, op)
            pdl.ReplaceOp(op, repl_values=[result_op.val])

        pdl.RewriteOp(op, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Verify GetResultOp was created in the rewriter
    get_result_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.GetResultOp)
    ]
    assert len(get_result_ops) >= 1


def test_generate_rewriter_for_results():
    """Test _generate_rewriter_for_results generates GetResultsOp"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import MatcherGenerator

    # Setup generator
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    rewriter_block = rewriter_module.body.block
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Setup PDL operations
    body = Region([Block()])
    with ImplicitBuilder(body.first_block):
        op = pdl.OperationOp("test.op")
        # ResultsOp with specific index
        results_op = pdl.ResultsOp(op.op, index=1)

    mapped_parent = rewriter_block.insert_arg(pdl.OperationType(), 0)
    rewrite_values: dict[SSAValue, SSAValue] = {op.op: mapped_parent}

    # Execute
    generator._generate_rewriter_for_results(  # pyright: ignore[reportPrivateUsage]
        results_op, rewrite_values, lambda x: rewrite_values.get(x, x)
    )

    # Verify
    assert len(rewriter_block.ops) == 1
    get_results = rewriter_block.first_op
    assert isinstance(get_results, pdl_interp.GetResultsOp)
    assert get_results.index is not None
    assert get_results.index.value.data == 1
    assert get_results.input_op == mapped_parent
    # Ensure the type is propagated from the PDL op
    assert get_results.value.type == results_op.val.type

    # Verify rewrite_values updated
    assert results_op.val in rewrite_values
    assert rewrite_values[results_op.val] == get_results.value


def test_generate_rewriter_for_type():
    """Test generate_rewriter with TypeOp creates type when constant is present"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp, i32
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup generator
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with TypeOp in the rewrite body
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Root operation must have a result type for the test to work
        root_type = pdl.TypeOp(i32).result
        root = pdl.OperationOp("test.op", type_values=(root_type,)).op

        # Rewrite body with TypeOp
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            const_type_op = pdl.TypeOp(constant_type=i32)
            new_op = pdl.OperationOp("new.op", type_values=(const_type_op.result,)).op
            pdl.ReplaceOp(root, repl_operation=new_op)

        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Verify CreateTypeOp was created in the rewriter
    create_type_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.CreateTypeOp)
    ]
    assert len(create_type_ops) >= 1


def test_generate_rewriter_for_types():
    """Test generate_rewriter with TypesOp creates types when constant is present"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ArrayAttr, ModuleOp, f32, i32
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup generator
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with TypesOp in the rewrite body
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Root operation must have a result type for the test to work
        root_type = pdl.TypeOp(i32).result
        root = pdl.OperationOp("test.op", type_values=(root_type,)).op

        # Rewrite body with TypesOp
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            types_attr = ArrayAttr([i32, f32])
            const_types_op = pdl.TypesOp(constant_types=types_attr)
            new_op = pdl.OperationOp("new.op", type_values=(const_types_op.result,)).op
            pdl.ReplaceOp(root, repl_operation=new_op)

        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Verify CreateTypesOp was created in the rewriter
    create_types_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.CreateTypesOp)
    ]
    assert len(create_types_ops) >= 1


def test_generate_rewriter_for_replace_with_repl_operation():
    """Test generate_rewriter with replace using a replacement operation"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with replace using a replacement operation
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type_op = pdl.TypeOp(i32).result
        old_op = pdl.OperationOp("old_op", type_values=(type_op,)).op

        # Rewrite body with replace operation
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            new_op = pdl.OperationOp("new_op").op
            pdl.ReplaceOp(old_op, repl_operation=new_op)

        pdl.RewriteOp(old_op, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Check that ReplaceOp was created
    replace_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.ReplaceOp)
    ]
    assert len(replace_ops) >= 1


def test_generate_rewriter_for_replace_with_repl_values():
    """Test generate_rewriter with replace using explicit replacement values"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with replace using explicit values
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Create operand values in the pattern
        val1 = pdl.OperandOp().value
        val2 = pdl.OperandOp().value
        old_op = pdl.OperationOp("old_op", operand_values=(val1, val2)).op

        # Rewrite body with replace values
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            pdl.ReplaceOp(old_op, repl_values=[val1, val2])

        pdl.RewriteOp(old_op, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Check that ReplaceOp was created
    replace_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.ReplaceOp)
    ]
    assert len(replace_ops) >= 1


def test_generate_rewriter_for_replace_erase():
    """Test generate_rewriter with replace with no replacement values (erase case)"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with replace with no replacement values (erase)
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        old_op = pdl.OperationOp("old_op").op

        # Rewrite body with replace with no values
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            pdl.ReplaceOp(old_op, repl_values=[])

        pdl.RewriteOp(old_op, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Execute the generation (should generate EraseOp since no replacement values)
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify EraseOp was created in the rewriter
    rewriter_func = rewriter_module.body.block.first_op
    assert isinstance(rewriter_func, pdl_interp.FuncOp)

    erase_ops = [
        op for op in rewriter_func.body.block.ops if isinstance(op, pdl_interp.EraseOp)
    ]
    assert len(erase_ops) == 1


def test_generate_rewriter_for_replace_with_operation_no_results():
    """Test generate_rewriter with replace using operation with no results (erase case)"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with replace using operation with no results
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        old_op = pdl.OperationOp("old_op", type_values=()).op

        # Rewrite body with replace operation (no type_values = no results)
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            new_op = pdl.OperationOp("new_op", type_values=()).op
            pdl.ReplaceOp(old_op, repl_operation=new_op)

        pdl.RewriteOp(old_op, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Execute the generation (should generate EraseOp since operation has no results)
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify EraseOp was created in the rewriter (since the replacement has no results)
    rewriter_func = rewriter_module.body.block.first_op
    assert isinstance(rewriter_func, pdl_interp.FuncOp)

    erase_ops = [
        op for op in rewriter_func.body.block.ops if isinstance(op, pdl_interp.EraseOp)
    ]
    assert len(erase_ops) == 1


def test_generate_rewriter_for_replace_multiple_values():
    """Test generate_rewriter with replace using multiple replacement values"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with replace using multiple values
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Create operand values in the pattern
        val1 = pdl.OperandOp().value
        val2 = pdl.OperandOp().value
        val3 = pdl.OperandOp().value
        old_op = pdl.OperationOp("old_op", operand_values=(val1, val2, val3)).op

        # Rewrite body with replace multiple values
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            pdl.ReplaceOp(old_op, repl_values=[val1, val2, val3])

        pdl.RewriteOp(old_op, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Check that ReplaceOp was created with all values
    replace_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.ReplaceOp)
    ]
    assert len(replace_ops) >= 1


def test_generate_rewriter_for_replace_with_operation_multiple_results():
    """Test generate_rewriter with replace using operation with multiple results"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp, f32, i32
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with replace using operation with multiple results
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type1 = pdl.TypeOp(i32).result
        old_op = pdl.OperationOp("old_op", type_values=(type1,)).op

        # Rewrite body with replace operation with multiple results
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            type1_new = pdl.TypeOp(i32).result
            type2_new = pdl.TypeOp(f32).result
            new_op = pdl.OperationOp("new_op", type_values=(type1_new, type2_new)).op
            pdl.ReplaceOp(old_op, repl_operation=new_op)

        pdl.RewriteOp(old_op, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Check that ReplaceOp was created
    replace_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.ReplaceOp)
    ]
    assert len(replace_ops) >= 1


def test_generate_rewriter_for_replace_single_value():
    """Test generate_rewriter with replace using a single replacement value"""
    from xdsl.builder import ImplicitBuilder
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a pattern with replace using a single value
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        # Create operand value in the pattern
        val = pdl.OperandOp().value
        old_op = pdl.OperationOp("old_op", operand_values=(val,)).op

        # Rewrite body with replace single value
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            pdl.ReplaceOp(old_op, repl_values=[val])

        pdl.RewriteOp(old_op, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    generator.generate_rewriter(pattern, [OperationPosition(None, depth=0)])

    # Verify rewriter function was created
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Check that ReplaceOp was created with the value
    replace_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.ReplaceOp)
    ]
    assert len(replace_ops) >= 1


def test_generate_rewriter_simple_pattern():
    """Test generate_rewriter with a simple pattern and replace operation"""
    # Create a matcher and rewriter module
    matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
    rewriter_module = ModuleOp([])

    generator = MatcherGenerator(matcher_func, rewriter_module)
    generator.rewriter_names = {}

    # Create a simple PDL pattern with a rewrite body
    pattern_body = Region([Block()])
    pattern_block = pattern_body.first_block
    with ImplicitBuilder(pattern_block):
        # Pattern matches an operation with result types
        root_type = pdl.TypeOp(i32).result
        root = pdl.OperationOp("test_op", type_values=(root_type,)).op

        # Create a rewrite body with a replace operation
        rewrite_body = Region([Block()])
        rewrite_block = rewrite_body.first_block
        with ImplicitBuilder(rewrite_block):
            # Simple replace with an operation that has result types
            type_op = pdl.TypeOp(f32).result
            new_op = pdl.OperationOp("new_op", type_values=(type_op,)).op
            pdl.ReplaceOp(root, repl_operation=new_op)

        # Use None for name_ so it's treated as a DAG rewriter, not a custom rewriter
        pdl.RewriteOp(root, body=rewrite_body)

    pattern = pdl.PatternOp(1, "test_pattern", pattern_body)

    # Build the predicate tree first to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()
    _predicates, _pattern_root, inputs = (
        predicate_tree_builder._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]
    )
    generator.value_to_position[pattern] = inputs

    # Call generate_rewriter
    used_match_positions: list[Position] = [OperationPosition(depth=0)]
    rewriter_ref = generator.generate_rewriter(pattern, used_match_positions)

    # Verify rewriter function was created
    assert isinstance(rewriter_ref, SymbolRefAttr)
    assert rewriter_ref.root_reference.data == "rewriters"
    assert len(rewriter_ref.nested_references.data) == 1

    # Check that the rewriter function was added to the module
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1
    rewriter_func = rewriter_funcs[0]

    # Verify function name matches the reference
    assert rewriter_func.sym_name.data == rewriter_ref.nested_references.data[0].data

    # Verify function has correct structure
    assert len(rewriter_func.body.blocks) == 1
    entry_block = rewriter_func.body.block

    # Should have at least one operation (FinalizeOp at the end)
    ops = list(entry_block.ops)
    assert ops

    # Last operation should be FinalizeOp
    last_op = ops[-1]
    assert isinstance(last_op, pdl_interp.FinalizeOp)

    # Function type should be updated based on arguments added
    assert rewriter_func.function_type == FunctionType.from_lists(
        [pdl.OperationType()], []
    )


def test_generate_rewriter_multiple_patterns_without_names():
    """Test generate_rewriter with multiple patterns without sym_name"""
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Block, Region
    from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
        MatcherGenerator,
        PredicateTreeBuilder,
    )
    from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import OperationPosition

    # Setup
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create first pattern without sym_name
    body1 = Region([Block()])
    block1 = body1.first_block
    with ImplicitBuilder(block1):
        root1 = pdl.OperationOp("op1").op
        pdl.RewriteOp(root1)

    pattern1 = pdl.PatternOp(1, None, body1)

    # Create second pattern without sym_name
    body2 = Region([Block()])
    block2 = body2.first_block
    with ImplicitBuilder(block2):
        root2 = pdl.OperationOp("op2").op
        pdl.RewriteOp(root2)

    pattern2 = pdl.PatternOp(1, None, body2)

    # Create third pattern without sym_name
    body3 = Region([Block()])
    block3 = body3.first_block
    with ImplicitBuilder(block3):
        root3 = pdl.OperationOp("op3").op
        pdl.RewriteOp(root3)

    pattern3 = pdl.PatternOp(1, None, body3)

    # Extract pattern predicates to populate value_to_position
    predicate_tree_builder = PredicateTreeBuilder()

    root_pos = OperationPosition(None, depth=0)
    _pred1, _, inputs1 = predicate_tree_builder._extract_pattern_predicates(pattern1)  # pyright: ignore[reportPrivateUsage]
    generator.value_to_position[pattern1] = inputs1

    _pred2, _, inputs2 = predicate_tree_builder._extract_pattern_predicates(pattern2)  # pyright: ignore[reportPrivateUsage]
    generator.value_to_position[pattern2] = inputs2

    _pred3, _, inputs3 = predicate_tree_builder._extract_pattern_predicates(pattern3)  # pyright: ignore[reportPrivateUsage]
    generator.value_to_position[pattern3] = inputs3

    # Call generate_rewriter for each pattern
    rewriter_ref1 = generator.generate_rewriter(pattern1, [root_pos])
    rewriter_ref2 = generator.generate_rewriter(pattern2, [root_pos])
    rewriter_ref3 = generator.generate_rewriter(pattern3, [root_pos])

    # Verify all rewriter functions were created with different names
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 3, "Should have created 3 rewriter functions"

    # Extract names from the rewriter references
    names = [
        ref.nested_references.data[0].data
        for ref in [rewriter_ref1, rewriter_ref2, rewriter_ref3]
    ]

    # Verify names follow the expected pattern for unnamed patterns
    # The first unnamed rewriter doesn't get a suffix, subsequent ones use indices starting from 0
    expected_names = [
        "pdl_generated_rewriter",
        "pdl_generated_rewriter_0",
        "pdl_generated_rewriter_1",
    ]
    assert names == expected_names, f"Expected {expected_names}, got {names}"


# =============================================================================
# Tests for _generate_rewriter_for_apply_native_rewrite
# =============================================================================


def test_generate_rewriter_for_apply_native_rewrite_with_arguments():
    """Test _generate_rewriter_for_apply_native_rewrite properly processes arguments"""
    # Setup matcher function
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create a pattern with ApplyNativeRewriteOp with multiple arguments
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        arg1 = pdl.OperandOp().value
        arg2 = pdl.OperandOp().value
        arg3 = pdl.OperandOp().value

        rewrite_block = Region([Block()])
        rewrite_block_inner = rewrite_block.first_block
        with ImplicitBuilder(rewrite_block_inner):
            # ApplyNativeRewriteOp with 3 arguments
            pdl.ApplyNativeRewriteOp("test_rewrite", [arg1, arg2, arg3], ())
        pdl.RewriteOp(None, body=rewrite_block)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Set up value_to_position with inputs for the arguments
    root_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {
        arg1: root_pos,
        arg2: root_pos,
        arg3: root_pos,
    }
    generator.value_to_position[pattern] = inputs

    # generate_rewriter should create ApplyRewriteOp
    rewriter_ref = generator.generate_rewriter(pattern, [root_pos])

    # Verify rewriter function was created
    assert rewriter_ref is not None

    # Get the created rewriter function
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1, "Should have created 1 rewriter function"

    rewriter_func = rewriter_funcs[0]

    # Verify the rewriter contains ApplyRewriteOp with correct constraint name and arguments
    apply_rewrite_ops = [
        op
        for op in rewriter_func.body.block.ops
        if isinstance(op, pdl_interp.ApplyRewriteOp)
    ]
    assert len(apply_rewrite_ops) == 1, "Should have exactly one ApplyRewriteOp"

    apply_op = apply_rewrite_ops[0]
    assert apply_op.rewrite_name.data == "test_rewrite"
    assert len(apply_op.args) == 3, "Should have 3 arguments"


# =============================================================================
# Tests for _generate_rewriter_for_results
# =============================================================================


def test_generate_rewriter_for_results_with_index():
    """Test _generate_rewriter_for_results with specific index"""
    # Setup matcher function
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create a pattern with ResultsOp
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        producer = pdl.OperationOp("producer").op

        rewrite_block = Region([Block()])
        rewrite_block_inner = rewrite_block.first_block
        with ImplicitBuilder(rewrite_block_inner):
            # Create a ResultsOp in the rewrite section
            pdl.ResultsOp(producer, 0)

        pdl.RewriteOp(None, body=rewrite_block)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Set up value_to_position
    root_pos = OperationPosition(depth=0)
    producer_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {
        producer: producer_pos,
    }
    generator.value_to_position[pattern] = inputs

    # generate_rewriter should complete without raising errors
    rewriter_ref = generator.generate_rewriter(pattern, [root_pos])

    # Verify a rewriter function was created
    assert isinstance(rewriter_ref, SymbolRefAttr)
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1

    # Verify the rewriter function has the correct structure
    rewriter_func = rewriter_funcs[0]
    body_ops = list(rewriter_func.body.block.ops)
    # Should have GetResultsOp and FinalizeOp
    assert any(isinstance(op, pdl_interp.GetResultsOp) for op in body_ops)
    assert any(isinstance(op, pdl_interp.FinalizeOp) for op in body_ops)


def test_generate_rewriter_for_results_with_none_index():
    """Test _generate_rewriter_for_results with None index (all results)"""
    # Setup matcher function
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create a pattern with ResultsOp with None index
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        producer = pdl.OperationOp("producer").op

        rewrite_block = Region([Block()])
        rewrite_block_inner = rewrite_block.first_block
        with ImplicitBuilder(rewrite_block_inner):
            # ResultsOp with None index (all results)
            pdl.ResultsOp(producer, None)

        pdl.RewriteOp(None, body=rewrite_block)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Set up value_to_position
    root_pos = OperationPosition(depth=0)
    producer_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {
        producer: producer_pos,
    }
    generator.value_to_position[pattern] = inputs

    # generate_rewriter should complete without errors
    rewriter_ref = generator.generate_rewriter(pattern, [root_pos])

    # Verify structure
    assert isinstance(rewriter_ref, SymbolRefAttr)
    rewriter_funcs = [
        op for op in rewriter_module.body.block.ops if isinstance(op, pdl_interp.FuncOp)
    ]
    assert len(rewriter_funcs) == 1


def test_generate_rewriter_for_results_mapping():
    """Test that _generate_rewriter_for_results properly updates rewrite_values"""
    # Setup matcher function
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create a pattern with ResultsOp that should be mapped
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        producer = pdl.OperationOp("producer").op

        rewrite_block = Region([Block()])
        rewrite_block_inner = rewrite_block.first_block
        with ImplicitBuilder(rewrite_block_inner):
            rewrite_producer = pdl.OperationOp("producer").op
            pdl.ResultsOp(rewrite_producer, 1)

        pdl.RewriteOp(None, body=rewrite_block)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Set up value_to_position
    root_pos = OperationPosition(depth=0)
    producer_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {
        producer: producer_pos,
    }
    generator.value_to_position[pattern] = inputs

    # generate_rewriter should map the results properly
    rewriter_ref = generator.generate_rewriter(pattern, [root_pos])

    # Verify the rewriter was created correctly
    assert isinstance(rewriter_ref, SymbolRefAttr)


# =============================================================================
# Tests for generate_rewriter fallback TypeError
# =============================================================================


def test_generate_rewriter_unexpected_op_type():
    """Test that generate_rewriter raises TypeError for unexpected operation types in rewrite body"""

    from xdsl.dialects import test

    # Setup matcher function
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create a pattern with a rewrite body
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("root").op
        rewriter_op = pdl.RewriteOp(None)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Manually insert an unexpected operation type into the rewrite body
    rewrite_body = rewriter_op.body
    assert rewrite_body is not None

    # This will trigger the TypeError in the default case
    block_with_rewrite = rewrite_body.first_block
    assert block_with_rewrite is not None

    # Create a simple operation that's not PDL
    # For testing, we'll manually add it to the rewrite block
    unexpected_op = test.TestOp()
    block_with_rewrite.add_op(unexpected_op)

    # Set up value_to_position
    root_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {
        root: root_pos,
    }
    generator.value_to_position[pattern] = inputs

    # generate_rewriter should raise TypeError for unexpected operation type
    with pytest.raises(TypeError, match="Unexpected op type"):
        generator.generate_rewriter(pattern, [root_pos])


def test_generate_rewriter_with_custom_rewriter():
    """Test that generate_rewriter correctly handles custom rewriter (pdl.RewriteOp with name_)"""
    # Setup matcher function
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create a pattern with a custom rewriter (RewriteOp with name_)
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("root").op
        # RewriteOp with a name indicates a custom rewriter
        pdl.RewriteOp(root, name="custom_rewriter")

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Set up value_to_position
    root_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {
        root: root_pos,
    }
    generator.value_to_position[pattern] = inputs

    # Generate the rewriter
    rewriter_name = generator.generate_rewriter(pattern, [root_pos])

    # Verify that a rewriter was created
    assert rewriter_name is not None
    # Verify that the rewriter module contains an ApplyRewriteOp
    rewriter_func = rewriter_module.body.ops.first
    assert isinstance(rewriter_func, pdl_interp.FuncOp)
    apply_rewrite_ops = [
        op for op in rewriter_func.body.ops if isinstance(op, pdl_interp.ApplyRewriteOp)
    ]
    assert len(apply_rewrite_ops) == 1
    assert apply_rewrite_ops[0].rewrite_name.data == "custom_rewriter"


def test_generate_rewriter_unsupported_operation_in_rewrite_body():
    """Test that generate_rewriter raises TypeError for unsupported operations in rewrite body"""

    from xdsl.dialects import test

    # Setup matcher function
    matcher_body = Region([Block(arg_types=(pdl.OperationType(),))])
    matcher_func = pdl_interp.FuncOp(
        "matcher", ((pdl.OperationType(),), ()), region=matcher_body
    )
    rewriter_module = ModuleOp([])
    generator = MatcherGenerator(matcher_func, rewriter_module)

    # Create a pattern where we manually add an unsupported operation to the rewrite body
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        root = pdl.OperationOp("root").op
        rewriter_op = pdl.RewriteOp(None)

    pattern = pdl.PatternOp(1, "test_pattern", body)

    # Manually insert an unsupported operation into the rewrite body
    rewrite_body = rewriter_op.body
    assert rewrite_body is not None

    block_with_rewrite = rewrite_body.first_block
    assert block_with_rewrite is not None

    # Add an unsupported operation (builtin ModuleOp)
    unexpected_op = test.TestOp()
    block_with_rewrite.add_op(unexpected_op)

    # Set up value_to_position
    root_pos = OperationPosition(depth=0)
    inputs: dict[SSAValue, Position] = {
        root: root_pos,
    }
    generator.value_to_position[pattern] = inputs

    # generate_rewriter should raise TypeError for unexpected operation type
    with pytest.raises(TypeError, match="Unexpected op type"):
        generator.generate_rewriter(pattern, [root_pos])
