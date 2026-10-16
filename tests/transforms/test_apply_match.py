from unittest.mock import patch

import pytest

from xdsl.context import Context
from xdsl.dialects import match, test
from xdsl.dialects.builtin import ModuleOp, StringAttr, SymbolRefAttr
from xdsl.interpreter import Interpreter
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.transforms.apply_match import ApplyMatchPass, MatchRewritePattern
from xdsl.utils.exceptions import InterpretationError, PassFailedException


@pytest.mark.parametrize("raises", [False, True])
def test_rewrite_invocation_and_cleanup(raises: bool):
    matcher = match.MatcherOp()
    root_handle = matcher.body.block.args[0]
    matcher.body.block.add_ops(
        [
            match.HasNameOp(root_handle, "test.op"),
            match.SuccessOp("rewrite", 1, [root_handle]),
        ]
    )
    module = ModuleOp([matcher, ModuleOp([], sym_name=StringAttr("rewrite"))])
    module.verify()
    interpreter = Interpreter(module)
    pattern = MatchRewritePattern(matcher, interpreter)
    root = test.TestOp()
    ModuleOp([root])
    rewriter = PatternRewriter(root)

    with patch.object(
        interpreter,
        "call_op",
        side_effect=RuntimeError("rewrite failed") if raises else None,
    ) as call:
        if raises:
            with pytest.raises(RuntimeError, match="rewrite failed"):
                pattern.match_and_rewrite(root, rewriter)
        else:
            pattern.match_and_rewrite(root, rewriter)
        call.assert_called_once_with(SymbolRefAttr("rewrite"), (root,))

    with pytest.raises(InterpretationError, match="Expected an active root"):
        PDLInterpFunctions.get_root(interpreter)
    with pytest.raises(InterpretationError, match="Expected an active rewriter"):
        PDLInterpFunctions.get_rewriter(interpreter)


@pytest.mark.parametrize("count", [0, 2])
def test_requires_one_matcher(count: int):
    with pytest.raises(PassFailedException, match="expected exactly one top-level"):
        ApplyMatchPass().apply(
            Context(), ModuleOp([match.MatcherOp() for _ in range(count)])
        )
