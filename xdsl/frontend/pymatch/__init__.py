"""
Scalar matching helpers for ordinary Python and the experimental Match frontend.

Navigation returns a real xDSL object or raises MatchFailure. Predicate helpers
return None on success and raise MatchFailure on mismatch. Invalid indices and
counts are programming errors, not mismatches. Matchers must not mutate payload IR.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar

from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern


class MatchFailure(Exception):
    """An expected mismatch in the scalar Python matching helpers."""


def has_name(op: Operation, name: str) -> None:
    if op.name != name:
        raise MatchFailure()


def check_operand_count(op: Operation, count: int) -> None:
    if count < 0:
        raise ValueError("expected a nonnegative count")
    if len(op.operands) != count:
        raise MatchFailure()


def check_result_count(op: Operation, count: int) -> None:
    if count < 0:
        raise ValueError("expected a nonnegative count")
    if len(op.results) != count:
        raise MatchFailure()


def get_operand(op: Operation, index: int) -> SSAValue:
    if index < 0:
        raise ValueError("expected a nonnegative index")
    if index >= len(op.operands):
        raise MatchFailure()
    return op.operands[index]


def get_result(op: Operation, index: int) -> SSAValue:
    if index < 0:
        raise ValueError("expected a nonnegative index")
    if index >= len(op.results):
        raise MatchFailure()
    return op.results[index]


def get_defining_op(value: SSAValue) -> Operation:
    if not isinstance(value, OpResult):
        raise MatchFailure()
    return value.owner


def get_attribute(op: Operation, name: str) -> Attribute:
    value = op.attributes.get(name, op.properties.get(name))
    if value is None:
        raise MatchFailure()
    return value


def has_attr_value(attribute: Attribute, value: Attribute) -> None:
    if attribute != value:
        raise MatchFailure()


CapturesT = TypeVar("CapturesT", bound=tuple[Operation | SSAValue | Attribute, ...])


@dataclass
class PythonMatchRewritePattern(RewritePattern, Generic[CapturesT]):
    """Pass a successful match's typed tuple to its Python rewrite function."""

    matcher: Callable[[Operation], CapturesT]
    rewrite: Callable[[CapturesT, PatternRewriter], None]

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        try:
            captures = self.matcher(op)
        except MatchFailure:
            return
        self.rewrite(captures, rewriter)
