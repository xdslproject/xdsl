from xdsl.builder import ImplicitBuilder
from xdsl.dialects import pdl
from xdsl.dialects.builtin import IntegerType, StringAttr, f32, i32
from xdsl.ir import Block, Region, SSAValue
from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
    PatternAnalyzer,
)
from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
    AttributeAnswer,
    AttributeConstraintQuestion,
    AttributePosition,
    ConstraintQuestion,
    EqualToQuestion,
    IsNotNullQuestion,
    OperandCountQuestion,
    OperationNameQuestion,
    OperationPosition,
    Position,
    PositionalPredicate,
    Predicate,
    ResultCountQuestion,
    ResultPosition,
    StringAnswer,
    TrueAnswer,
    TypeAnswer,
    TypeConstraintQuestion,
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
