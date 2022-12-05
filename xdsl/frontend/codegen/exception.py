from dataclasses import dataclass, field
from io import StringIO
from typing import Any, List
from xdsl.printer import Printer


@dataclass
class CodegenInternalException(Exception):
    """
    Internal exception type during xDSL code generation. Should be used for internal errors,
    such as trying to pop an operand from epty stack.
    """

    msg: str

    def __init__(self, format_msg: str, xdsl_args: List[Any] = []) -> None:
        super().__init__()
        xdsl_args = [self.to_str(obj) for obj in xdsl_args]
        self.msg = format_msg if len(xdsl_args) == 0 else format_msg.format(*xdsl_args)
        if not self.msg.endswith("."):
            self.msg = f"{self.msg}."

    def to_str(xdsl_obj: Any) -> str:
        text = StringIO("")
        printer = Printer(stream=text)
        printer.print(xdsl_obj)
        return text.getvalue().strip()

    def __str__(self) -> str: 
        return f"Internal code generation exception. {self.msg}"


@dataclass
class CodegenException(CodegenInternalException):
    """
    Exception type during xDSL code generation. This is a user-facing exception which
    should be informative and used for unsupported cases or type checks.
    """

    line: int
    col: int

    def __init__(self, line: int, col: int, format_msg: str, xdsl_args: List[Any] = []):
        super().__init__(format_msg, xdsl_args)
        self.line = line
        self.col = col

    def __str__(self) -> str:
        return f"Code generation exception at {self.line}:{self.col}. {self.msg}"
