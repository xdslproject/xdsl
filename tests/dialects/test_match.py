import pytest

from xdsl.dialects import match, pdl, test
from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr, i32
from xdsl.ir import Block, Region
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
