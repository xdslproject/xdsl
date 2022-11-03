from dataclasses import dataclass
from io import StringIO
from typing import Any
from xdsl.printer import Printer


@dataclass
class CodegenException(Exception):
    """Exception type during xDSL code generation."""
    msg: str

    def __str__(self) -> str:
        return f"Exception in code generation: {self.msg}."


def prettify(xdsl_obj: Any) -> str:
    """Can be used in error messages to ensure pretty-printing of xDSL objects."""
    text = StringIO("")
    printer = Printer(stream=text)
    printer.print(xdsl_obj)
    return text.getvalue().strip()
