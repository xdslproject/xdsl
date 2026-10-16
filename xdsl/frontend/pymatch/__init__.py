"""
Typed Python match/rewrite patterns and the experimental Match frontend.

Use `@rewrite_for(matcher)` with a matcher returning a capture tuple or None.
The rewrite takes the unpacked captures followed by a PatternRewriter.

The older helper-based PyMatchContext uses the scalar helpers below: navigation
returns an xDSL object or raises MatchFailure, and predicates raise on mismatch.
Invalid indices/counts are programming errors. Matchers must not mutate payload IR.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from inspect import signature
from typing import Generic, cast, get_type_hints

from typing_extensions import TypeVar, TypeVarTuple, Unpack

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern


class MatchFailure(Exception):
    """An expected mismatch in the legacy scalar Python matching helpers."""


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


RootT = TypeVar("RootT", bound=Operation)
Captures = TypeVarTuple("Captures")


@dataclass
class PythonMatchRewritePattern(RewritePattern, Generic[RootT, Unpack[Captures]]):
    """
    Dispatch on the matcher's root annotation, then unpack successful captures.

    Only None denotes mismatch; an empty tuple is a successful match. Neither
    matcher nor rewrite exceptions are suppressed. Matchers must not mutate IR.

    `.module` lazily compiles the restricted Python subset documented in
    `pymatch.compiler`, including the rewrite. Direct calls still execute Python.
    """

    matcher: Callable[[RootT], tuple[Unpack[Captures]] | None]
    rewrite: Callable[[Unpack[Captures], PatternRewriter], None]
    _root_type: type[RootT] = field(init=False)

    def __post_init__(self) -> None:
        parameters = tuple(signature(self.matcher).parameters.values())
        if len(parameters) != 1 or parameters[0].kind not in (
            parameters[0].POSITIONAL_ONLY,
            parameters[0].POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError("expected one positional matcher argument")
        root_type = get_type_hints(self.matcher).get(parameters[0].name)
        if not isinstance(root_type, type) or not issubclass(root_type, Operation):
            raise TypeError("expected an Operation subclass as the root annotation")
        self._root_type = cast(type[RootT], root_type)

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self._root_type):
            return
        captures = self.matcher(op)
        if captures is not None:
            self.rewrite(*captures, rewriter)

    def __call__(self, *args: Unpack[tuple[Unpack[Captures], PatternRewriter]]) -> None:
        self.rewrite(*args)

    @cached_property
    def module(self) -> ModuleOp:
        from xdsl.frontend.pymatch.compiler import compile_pattern

        return compile_pattern(self.matcher, self.rewrite)


def rewrite_for(
    matcher: Callable[[RootT], tuple[Unpack[Captures]] | None],
) -> Callable[
    [Callable[[Unpack[Captures], PatternRewriter], None]],
    PythonMatchRewritePattern[RootT, Unpack[Captures]],
]:
    """Type-check a rewrite's positional captures against its matcher's tuple."""

    def decorate(
        rewrite: Callable[[Unpack[Captures], PatternRewriter], None],
    ) -> PythonMatchRewritePattern[RootT, Unpack[Captures]]:
        return PythonMatchRewritePattern(matcher, rewrite)

    return decorate
