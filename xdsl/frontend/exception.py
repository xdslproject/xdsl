from dataclasses import dataclass


@dataclass
class FrontendProgramException(Exception):
    """
    Exception type used when something goes wrong with `FrontendProgram`.
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

    line: int
    col: int

    def __init__(self, line: int, col: int, msg: str):
        super().__init__(msg)
        self.line = line
        self.col = col

    def __str__(self) -> str:
        return f"Code generation exception at {self.line}:{self.col}. {self.msg}"
