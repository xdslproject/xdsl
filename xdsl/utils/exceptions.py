"""Custom xDSL exceptions.

This module contains all custom exceptions used by xDSL.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any

if typing.TYPE_CHECKING:
    from xdsl.ir import Attribute
    from xdsl.parser import Span
    from xdsl.utils.parse_pipeline import Token


class DiagnosticException(Exception):
    pass


class VerifyException(DiagnosticException):
    pass


class PyRDLError(Exception):
    pass


class PyRDLOpDefinitionError(Exception):
    pass


class PyRDLAttrDefinitionError(Exception):
    pass


class InvalidIRException(Exception):
    pass


class InterpretationError(Exception):
    """
    An error that can be raised during interpretation, or Interpreter setup.
    """

    pass


@dataclass
class BuilderNotFoundException(Exception):
    """
    Exception raised when no builders are found for a given attribute type
    and a given tuple of arguments.
    """

    attribute: type[Attribute]
    args: tuple[Any, ...]

    def __str__(self) -> str:
        return (
            f"No builder found for attribute {self.attribute} with "
            f"arguments {self.args}"
        )


@dataclass
class ParseError(Exception):
    span: Span
    msg: str

    def __str__(self) -> str:
        return self.span.print_with_context(self.msg)

    def with_context(self) -> str:
        return self.span.print_with_context(self.msg)


@dataclass
class MultipleSpansParseError(ParseError):
    ref_text: str | None
    refs: list[tuple[Span, str | None]]

    def __repr__(self) -> str:
        res = super().__repr__() + "\n"
        res += self.ref_text or "With respect to:\n"
        for span, msg in self.refs:
            res += span.print_with_context(msg) + "\n"
        return res


class PassPipelineParseError(BaseException):
    def __init__(self, token: Token, msg: str):
        super().__init__(
            "Error parsing pass pipeline specification:\n"
            + token.span.print_with_context(msg)
        )
