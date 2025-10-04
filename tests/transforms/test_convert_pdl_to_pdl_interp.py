from xdsl.builder import ImplicitBuilder
from xdsl.dialects import pdl
from xdsl.dialects.builtin import f32
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


def test_extract_tree_predicates():
    body = Region([Block()])
    block = body.first_block
    with ImplicitBuilder(block):
        type = pdl.TypeOp(f32).result
        pdl.OperationOp("op1", type_values=(type,)).op

        pdl.RewriteOp(None, name="rewrite")
    pattern = pdl.PatternOp(1, "pattern", body)

    p = PredicateTreeBuilder()
    predicates, _, _ = p._extract_pattern_predicates(pattern)  # pyright: ignore[reportPrivateUsage]

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
