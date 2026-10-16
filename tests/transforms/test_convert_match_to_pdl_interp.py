import pytest

from xdsl.context import Context
from xdsl.dialects import builtin, match, pdl, pdl_interp, test
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.parser import Parser
from xdsl.transforms.convert_match_to_pdl_interp import ConvertMatchToPDLInterpPass
from xdsl.utils.exceptions import PassFailedException


@pytest.mark.parametrize("name", [None, "matcher", "pattern"])
def test_scalar_lowering(name: str | None):
    matcher = match.MatcherOp(sym_name=name)
    root = matcher.body.block.args[0]
    result = match.GetResultOp(2, root)
    unwrap = match.IsNotNullOp(result.result)
    matcher.body.block.add_ops(
        [
            match.CheckOperandCountOp(root, 2, at_least=True),
            match.CheckResultCountOp(root, 3, at_least=True),
            result,
            unwrap,
            match.SuccessOp("rewrite", 7, [unwrap.unwrapped, root]),
        ]
    )
    module = ModuleOp([ModuleOp([], sym_name=StringAttr("rewrite")), matcher])
    module.verify()
    ConvertMatchToPDLInterpPass().apply(Context(), module)
    module.verify()

    assert not any(isinstance(op, match.MatcherOp) for op in module.walk())
    function = module.body.block.last_op
    assert isinstance(function, pdl_interp.FuncOp)
    assert function.sym_name == StringAttr("matcher")
    checks = [
        op
        for op in function.walk()
        if isinstance(
            op, pdl_interp.CheckOperandCountOp | pdl_interp.CheckResultCountOp
        )
    ]
    assert [op.count.value.data for op in checks] == [2, 3]
    assert all(op.compareAtLeast is not None for op in checks)
    get_result = next(
        op for op in function.walk() if isinstance(op, pdl_interp.GetResultOp)
    )
    assert get_result.index.value.data == 2
    record = next(
        op for op in function.walk() if isinstance(op, pdl_interp.RecordMatchOp)
    )
    assert record.inputs == (get_result.value, function.body.blocks[0].args[0])
    assert record.matched_ops == (function.body.blocks[0].args[0],)
    assert record.rootKind is None
    assert record.benefit.value.data == 7


@pytest.mark.parametrize(
    "source,error",
    [
        ("", "expected exactly one top-level"),
        (
            """
            match.matcher root(%root: !pdl.operation) { match.success @rewrite benefit(1) }
            match.matcher root(%root: !pdl.operation) { match.success @rewrite benefit(1) }
            """,
            "expected exactly one top-level",
        ),
        (
            """
            builtin.module {
              builtin.module @rewrite {}
              match.matcher root(%root: !pdl.operation) { match.success @rewrite benefit(1) }
            }
            """,
            "expected exactly one top-level",
        ),
        (
            """
            builtin.module @matcher {}
            match.matcher root(%root: !pdl.operation) { match.success @rewrite benefit(1) }
            """,
            "@matcher already exists",
        ),
        (
            """
            match.matcher root(%root: !pdl.operation) {
              match.success @rewrite benefit(1)
              match.has_name %root, "test.op"
            }
            """,
            "expected one match.success at the end",
        ),
        (
            """
            match.matcher root(%root: !pdl.operation) {
              match.success @rewrite benefit(1)
              match.success @rewrite benefit(2)
            }
            """,
            "expected one match.success at the end",
        ),
        (
            """
            match.matcher root(%root: !pdl.operation) {
              %v = match.get_operand -1 of %root : !match.optional<!pdl.value>
              match.success @rewrite benefit(1)
            }
            """,
            "negative indices are not supported",
        ),
        (
            """
            match.matcher root(%root: !pdl.operation) {
              %v = match.get_result 0 of %root : !match.optional<!pdl.type>
              match.success @rewrite benefit(1)
            }
            """,
            "unsupported optional result type",
        ),
        (
            """
            match.matcher root(%root: !pdl.operation) {
              "test.op"() : () -> ()
              match.success @rewrite benefit(1)
            }
            """,
            "unsupported operation in scalar matcher: test.op",
        ),
    ],
)
def test_unsupported_matchers(source: str, error: str):
    ctx = Context()
    for dialect in (builtin.Builtin, match.Match, pdl.PDL, test.Test):
        ctx.load_dialect(dialect)
    module = Parser(ctx, "builtin.module @rewrite {}\n" + source).parse_module()
    module.verify()
    before = module.clone()
    with pytest.raises(PassFailedException, match=error):
        ConvertMatchToPDLInterpPass().apply(ctx, module)
    assert module.is_structurally_equivalent(before)
