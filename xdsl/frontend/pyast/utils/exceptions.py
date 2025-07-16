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

    file: str | None
    line: int
    col: int

    def __init__(
        self,
        file: str | None,
        line: int,
        col: int,
        msg: str,
    ):
        super().__init__(msg)
        self.file = file
        self.line = line
        self.col = col

    def __str__(self) -> str:
        str = "Code generation exception at "
        if self.file:
            return (
                str + f'"{self.file}", line {self.line} column {self.col}: {self.msg}'
            )
        else:
            return str + f"line {self.line} column {self.col}: {self.msg}"
