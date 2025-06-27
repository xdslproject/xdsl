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


class UnregisteredConstructException(Exception):
    """
    An exception raised when a dialect, operation, type,
    or attribute is not registered.
    """


class AlreadyRegisteredConstructException(Exception):
    """
    An exception raised when a dialect, operation, type,
    or attribute is registered twice.
    """


class DiagnosticException(Exception):
    pass


class VerifyException(DiagnosticException):
    pass


class PassFailedException(DiagnosticException):
    """
    A diagnostic error which can be raised during the execution of a pass, used to
    signify that the pass did not succeed.
    """

    pass


class PyRDLError(Exception):
    """
    An error in our IRDL eDSL.
    """


class PyRDLOpDefinitionError(PyRDLError):
    """
    An error in the Operation definition eDSL.
    """


class PyRDLAttrDefinitionError(PyRDLError):
    """
    An error in the Attribute definition eDSL.
    """


class PyRDLTypeError(TypeError, PyRDLError):
    """
    A type error in our IRDL eDSL.
    """


class InvalidIRException(Exception):
    pass


class ShrinkException(Exception):
    """
    Exception for test case reduction when used in conjunction with the [Shrink Ray](https://github.com/DRMacIver/shrinkray)
    reducer.

    To find a reduced version of a test case, raise this exception on the line of code you want to hit,
    and pass the `--shrink` argument to `xdsl-opt`, by changing its invocation from:
    `xdsl-opt input_file.mlir -p my,pass,pipeline`
    to:
    `shrinkray "xdsl-opt -p my,pass,pipeline --shrink" input_file.mlir`.
    """

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


@dataclass
class MultipleSpansParseError(ParseError):
    ref_text: str | None
    refs: list[tuple[Span, str | None]]

    def __str__(self) -> str:
        res = self.span.print_with_context(self.msg)
        if self.ref_text is not None:
            res += self.ref_text + "\n"
        for span, msg in self.refs:
            res += span.print_with_context(msg) + "\n"
        return res


class PassPipelineParseError(BaseException):
    def __init__(self, token: Token, msg: str):
        super().__init__(
            "Error parsing pass pipeline specification:\n"
            + token.span.print_with_context(msg)
        )
