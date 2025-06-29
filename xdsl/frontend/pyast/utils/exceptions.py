"""Custom exceptions for the PyAST frontend."""

from dataclasses import dataclass


@dataclass
class FrontendProgramException(Exception):
    """Exception type used when something goes wrong with `FrontendProgram`."""

    msg: str
    """The exception message."""

    def __str__(self) -> str:
        """Get a textual representation of the exception."""
        return f"{self.msg}"


@dataclass(init=False)
class CodeGenerationException(FrontendProgramException):
    """Exception type used when xDSL code generation fails.

    Should be used for user-facing errors, e.g. unsupported functionality or
    failed type checks.
    """

    file: str | None
    """The file where the exception occured."""

    line: int
    """The line where the exception occured."""

    col: int
    """The column where the exception occured."""

    def __init__(
        self,
        file: str | None,
        line: int,
        col: int,
        msg: str,
    ):
        """Instantiate the exception with the correct argument ordering."""
        super().__init__(msg)
        self.file = file
        self.line = line
        self.col = col

    def __str__(self) -> str:
        """Get a textual representation of the exception."""
        str = "Code generation exception at "
        if self.file:
            return (
                str + f'"{self.file}", line {self.line} column {self.col}: {self.msg}'
            )
        else:
            return str + f"line {self.line} column {self.col}: {self.msg}"
