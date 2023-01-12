from dataclasses import dataclass
from io import StringIO
from typing import Any, List
from xdsl.printer import Printer


@dataclass
class FrontendProgramException(Exception):
    """Exception type used when something goes wrong with `FrontendProgram`."""

    msg: str

    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.msg}"


@dataclass
class InternalCodeGenerationException(FrontendProgramException):
    """
    Exception type used when xDSL code generation fails. Should be used for internal
    errors, such as trying to pop an operand from an empty stack.
    """

    def __init__(self, format_msg: str, xdsl_args: List[Any] = []) -> None:
        super().__init__(format_msg)

        def to_str(xdsl_obj: Any) -> str:
            text = StringIO("")
            printer = Printer(stream=text)
            printer.print(xdsl_obj)
            return text.getvalue().strip()

        xdsl_args = [to_str(obj) for obj in xdsl_args]
        self.msg = format_msg if len(xdsl_args) == 0 else format_msg.format(*xdsl_args)
        if not self.msg.endswith("."):
            self.msg = f"{self.msg}."

    def __str__(self) -> str:
        return f"Internal code generation exception. {self.msg}"


@dataclass
class CodeGenerationException(InternalCodeGenerationException):
    """
    Exception type used when xDSL code generation fails. Should be used for user-facing
    errors, e.g. unsupported functionality or failed type checks.
    """

    line: int
    col: int

    def __init__(self, line: int, col: int, format_msg: str, xdsl_args: List[Any] = []):
        super().__init__(format_msg, xdsl_args)
        self.line = line
        self.col = col

    def __str__(self) -> str:
        return f"Code generation exception at {self.line}:{self.col}. {self.msg}"
