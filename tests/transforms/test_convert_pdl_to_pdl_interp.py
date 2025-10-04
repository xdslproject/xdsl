from xdsl.builder import ImplicitBuilder
from xdsl.dialects import pdl
from xdsl.dialects.builtin import f32, i32
from xdsl.ir import Block, Region
from xdsl.transforms.convert_pdl_to_pdl_interp.conversion import (
    PatternAnalyzer,
    PredicateTreeBuilder,
)
from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
    IsNotNullQuestion,
    OperandCountQuestion,
    OperationNameQuestion,
    OperationPosition,
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
