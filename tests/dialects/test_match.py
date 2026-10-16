import pytest

from xdsl.dialects import match, pdl, test
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    UnitAttr,
    i32,
)
from xdsl.ir import Attribute, Block, Region
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import create_ssa_value


@pytest.mark.parametrize("nested", [False, True])
def test_matcher_success(nested: bool):
    matcher = match.MatcherOp(sym_name="matcher", sym_visibility="private")
    root = matcher.body.block.args[0]
    success = match.SuccessOp(SymbolRefAttr("rewriters", ["rewrite"]), 0, [root])
    matcher.body.block.add_ops(
        [
            test.TestOp(regions=[Region(Block([success, test.TestTermOp()]))])
            if nested
            else success,
            match.HasNameOp(root, "test.op"),
        ]
    )
    # Like the reference, accept any symbol, without imposing a callable signature.
    rewriters = ModuleOp(
        [ModuleOp([], sym_name=StringAttr("rewrite"))], sym_name=StringAttr("rewriters")
    )
    ModuleOp([rewriters, matcher]).verify()
    assert matcher.sym_name == StringAttr("matcher")
    assert matcher.sym_visibility == StringAttr("private")
    assert success.inputs == (root,)


@pytest.mark.parametrize(
    "body,error",
    [
        (Region(), "expected a single block"),
        (Region(Block()), "expected body block to have exactly one argument"),
        (
            Region(Block(arg_types=[pdl.OperationType(), pdl.OperationType()])),
            "expected body block to have exactly one argument",
        ),
        (Region(Block(arg_types=[i32])), "expected body block argument to be of type"),
        (
            Region(Block(arg_types=[pdl.OperationType()])),
            "expected matcher to contain at least one `match.success`",
        ),
    ],
)
def test_matcher_invalid_body(body: Region, error: str):
    with pytest.raises(VerifyException, match=error):
        match.MatcherOp(body).verify()


def test_matcher_isolation():
    matcher = match.MatcherOp()
    matcher.body.block.add_ops(
        [
            match.HasNameOp(create_ssa_value(pdl.OperationType()), "test.op"),
            match.SuccessOp("rewrite", 1),
        ]
    )
    module = ModuleOp([ModuleOp([], sym_name=StringAttr("rewrite")), matcher])
    with pytest.raises(VerifyException, match="IsolatedFromAbove"):
        module.verify()


@pytest.mark.parametrize(
    "rewriter", [SymbolRefAttr("missing"), SymbolRefAttr("rewriters", ["missing"])]
)
def test_success_unknown_symbol(rewriter: SymbolRefAttr):
    matcher = match.MatcherOp()
    matcher.body.block.add_op(match.SuccessOp(rewriter, 1))
    module = ModuleOp([ModuleOp([], sym_name=StringAttr("rewriters")), matcher])
    with pytest.raises(VerifyException, match="references an unknown rewriter symbol"):
        module.verify()


def test_success_requires_matcher():
    module = ModuleOp(
        [ModuleOp([], sym_name=StringAttr("rewrite")), match.SuccessOp("rewrite", 1)]
    )
    with pytest.raises(VerifyException, match=r"expects ancestor op 'match\.matcher'"):
        module.verify()


def test_success_negative_benefit():
    matcher = match.MatcherOp()
    matcher.body.block.add_op(match.SuccessOp("rewrite", -1))
    module = ModuleOp([ModuleOp([], sym_name=StringAttr("rewrite")), matcher])
    with pytest.raises(VerifyException, match="expected integer >= 0"):
        module.verify()


@pytest.mark.parametrize(
    "inner_type",
    [
        pdl.AttributeType(),
        pdl.OperationType(),
        pdl.TypeType(),
        pdl.ValueType(),
        pdl.RangeType(pdl.AttributeType()),
        pdl.RangeType(pdl.OperationType()),
        pdl.RangeType(pdl.TypeType()),
        pdl.RangeType(pdl.ValueType()),
    ],
)
def test_optional_unwrap(inner_type: pdl.AnyPDLType | pdl.RangeType[pdl.AnyPDLType]):
    optional = match.OptionalType(inner_type)
    unwrap = match.IsNotNullOp(create_ssa_value(optional))
    unwrap.verify()
    assert unwrap.unwrapped.type == inner_type


@pytest.mark.parametrize(
    "inner_type", [i32, StringAttr("not a type"), match.OptionalType(pdl.ValueType())]
)
def test_invalid_optional_type(inner_type: Attribute):
    with pytest.raises(VerifyException):
        match.OptionalType.new([inner_type])


def test_is_not_null_type_mismatch():
    unwrap = match.IsNotNullOp.create(
        operands=[create_ssa_value(match.OptionalType(pdl.ValueType()))],
        result_types=[pdl.TypeType()],
    )
    with pytest.raises(VerifyException, match=r"attribute !pdl\.value expected"):
        unwrap.verify()


@pytest.mark.parametrize("op_type", [match.GetOperandOp, match.GetResultOp])
def test_indexed_navigation(op_type: type[match.GetOperandOp | match.GetResultOp]):
    root = create_ssa_value(pdl.OperationType())
    op = op_type(3, root)
    op.verify()
    assert op.index == IntegerAttr(3, i32)
    assert op.result.type == match.OptionalType(pdl.ValueType())
    # The reference does not constrain indices to be nonnegative or fix the inner type.
    op_type(-1, root, match.OptionalType(pdl.TypeType())).verify()
    invalid = op_type.create(
        operands=[root],
        properties={"index": IntegerAttr(0, i32)},
        result_types=[pdl.ValueType()],
    )
    with pytest.raises(VerifyException, match=r"match.optional"):
        invalid.verify()


def test_attribute_navigation():
    root = create_ssa_value(pdl.OperationType())
    op = match.GetAttributeOp("value", root)
    op.verify()
    assert op.attribute_name == StringAttr("value")
    assert op.result.type == match.OptionalType(pdl.AttributeType())
    match.GetAttributeOp("value", root, match.OptionalType(pdl.TypeType())).verify()


@pytest.mark.parametrize(
    "value_type", [pdl.ValueType(), pdl.RangeType(pdl.ValueType())]
)
def test_defining_op_navigation(
    value_type: pdl.ValueType | pdl.RangeType[pdl.ValueType],
):
    value = create_ssa_value(value_type)
    op = match.GetDefiningOpOp(value)
    op.verify()
    assert op.result.type == match.OptionalType(pdl.OperationType())
    match.GetDefiningOpOp(value, match.OptionalType(pdl.TypeType())).verify()


@pytest.mark.parametrize(
    "op_type", [match.CheckOperandCountOp, match.CheckResultCountOp]
)
def test_count_checks(
    op_type: type[match.CheckOperandCountOp | match.CheckResultCountOp],
):
    root = create_ssa_value(pdl.OperationType())
    for count, at_least in ((0, False), (3, True)):
        op = op_type(root, count, at_least)
        op.verify()
        assert op.count == IntegerAttr(count, i32)
        assert op.at_least == (UnitAttr() if at_least else None)
    with pytest.raises(VerifyException, match="expected integer >= 0"):
        op_type(root, -1).verify()


@pytest.mark.parametrize("value", [IntegerAttr(7, i32), StringAttr("literal")])
def test_has_attr_value(value: Attribute):
    attribute = create_ssa_value(pdl.AttributeType())
    op = match.HasAttrValueOp(attribute, value)
    op.verify()
    assert op.attribute is attribute
    assert op.value == value
