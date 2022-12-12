"""Custom xDSL exceptions.

This module contains all custom exceptions used by xDSL.
"""

from dataclasses import dataclass
from typing import Any
from xdsl.ir import Attribute


class DiagnosticException(Exception):
    pass


class VerifyException(DiagnosticException):
    pass


@dataclass
class BuilderNotFoundException(Exception):
    """
    Exception raised when no builders are found for a given attribute type
    and a given tuple of arguments.
    """
    attribute: type[Attribute]
    args: tuple[Any]

    def __str__(self) -> str:
        return f"No builder found for attribute {self.attribute} with " \
               f"arguments {self.args}"
