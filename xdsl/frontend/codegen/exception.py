from dataclasses import dataclass


@dataclass
class CodegenException(Exception):
    """Exception type during xDSL code generation."""
    msg: str

    def __str__(self) -> str:
        return f"Exception in code generation: {self.msg}."
