from collections.abc import Callable
from unittest.mock import Mock

import pytest

from xdsl.dialects import builtin, test
from xdsl.frontend.pymatch import (
    MatchFailure,
    PythonMatchRewritePattern,
    check_operand_count,
    check_result_count,
    get_attribute,
    get_defining_op,
    get_operand,
    get_result,
    has_attr_value,
    has_name,
)
from xdsl.ir import Block
from xdsl.pattern_rewriter import PatternRewriter


def test_scalar_helpers():
    block = Block(arg_types=[builtin.i32, builtin.i64])
    root = test.TestOp(
        block.args,
        [builtin.i64],
        {"a": builtin.StringAttr("attribute")},
        {"p": builtin.StringAttr("property")},
    )
    has_name(root, "test.op")
    check_operand_count(root, 2)
    check_result_count(root, 1)
    assert get_operand(root, 1) is block.args[1]
    assert get_result(root, 0) is root.results[0]
    assert get_defining_op(root.results[0]) is root
    assert get_attribute(root, "p") == builtin.StringAttr("property")
    has_attr_value(get_attribute(root, "a"), builtin.StringAttr("attribute"))

    failures: list[tuple[Callable[..., object], tuple[object, ...]]] = [
        (has_name, (root, "other")),
        (check_operand_count, (root, 1)),
        (check_result_count, (root, 2)),
        (get_operand, (root, 2)),
        (get_result, (root, 1)),
        (get_defining_op, (block.args[0],)),
        (get_attribute, (root, "missing")),
        (has_attr_value, (builtin.IntegerAttr(0, 32), builtin.IntegerAttr(0, 64))),
    ]
    for helper, args in failures:
        with pytest.raises(MatchFailure):
            helper(*args)

    for helper in (get_operand, get_result, check_operand_count, check_result_count):
        with pytest.raises(ValueError, match="nonnegative"):
            helper(root, -1)


def test_python_rewrite_captures_and_failures():
    root = test.TestOp(result_types=[builtin.i32])
    Block([root])
    rewriter = PatternRewriter(root)
    captures = (root, root.results[0])
    matcher = Mock(return_value=captures)
    rewrite = Mock()
    pattern = PythonMatchRewritePattern(matcher, rewrite)
    pattern.match_and_rewrite(root, rewriter)
    rewrite.assert_called_once_with(captures, rewriter)
    rewrite.reset_mock()

    matcher.side_effect = MatchFailure()
    pattern.match_and_rewrite(root, rewriter)
    rewrite.assert_not_called()
    matcher.side_effect = ValueError("bad matcher")
    with pytest.raises(ValueError, match="bad matcher"):
        pattern.match_and_rewrite(root, rewriter)

    matcher.side_effect = None
    rewrite.side_effect = MatchFailure()
    with pytest.raises(MatchFailure):
        pattern.match_and_rewrite(root, rewriter)
