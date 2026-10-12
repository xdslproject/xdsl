from dataclasses import dataclass

from xdsl.utils.lexer import Location


@dataclass
class FrontendProgramException(Exception):
    """
    Exception type used when something goes wrong with the Python AST frontend.
    """

    msg: str

    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.msg}"


@dataclass
class CodeGenerationException(FrontendProgramException):
    """
    Exception type used when xDSL code generation fails. Should be used for
    user-facing errors, e.g. unsupported functionality or failed type checks.
    """

    location: Location

    def __init__(
        self,
        location: Location,
        msg: str,
    ):
        super().__init__(msg)
        self.location = location

    def __str__(self) -> str:
        return f"Code generation exception at {self.location}: {self.msg}"
