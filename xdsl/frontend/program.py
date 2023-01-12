import ast
import subprocess

from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List
from xdsl.dialects.builtin import ModuleOp
from xdsl.printer import Printer


@dataclass
class FrontendProgramException(Exception):
    """
    Exception type used when something goes wrong with `FrontendProgram`.
    """

    msg: str

    def __init__(self, msg):
        super().__init__()
        self.msg = msg

    def __str__(self) -> str:
        return f"{self.msg}"


@dataclass
class FrontendProgram:
    """
    Class to store the Python AST of a program written in the frontend. This
    program can be compiled and translated to xDSL or MLIR.
    """

    stmts: List[ast.stmt] = field(default=None)
    """AST nodes stored for compilation to xDSL."""

    globals: Dict[str, Any] = field(default=None)
    """Global information for this program, including all the imports."""

    xdsl_program: ModuleOp = field(default=None)
    """Generated xDSL program when AST is compiled."""

    def _check_can_compile(self):
        if self.stmts is None or self.globals is None:
            msg = \
"""
Cannot compile program without the code context. Try to use:
    p = FrontendProgram()
    with CodeContext(p):
        # Your code here."""
            raise FrontendProgramException(msg)

    def compile(self, desymref=True) -> None:
        """Generates xDSL from the source program."""

        # Both statements and globals msut be initialized from within the `CodeContext`.
        self._check_can_compile()

        # TODO: Land basic code generation in the next patch.
        self.xdsl_program = ModuleOp.from_region_or_ops([])
        self.xdsl_program.verify()

        # Optionally run desymrefication pass to produce actual SSA.
        if desymref:
            self.desymref()

    def desymref(self) -> None:
        """Desymrefy the generated xDSL."""

        # TODO: Land desymref in the next patch.
        raise FrontendProgramException(
            "Running desymref pass is not supported.")

        self.xdsl_program.verify()

    def _check_can_print(self):
        if self.xdsl_program is None:
            msg = \
"""
Cannot print the program IR without compiling it first. Make sure to use:
    p = FrontendProgram()
    with CodeContext(p):
        # Your code here.
    p.compile()"""
            raise FrontendProgramException(msg)

    def _print(self, target) -> str:
        self._check_can_print()

        file = StringIO("")
        printer = Printer(stream=file, target=target)
        printer.print_op(self.xdsl_program)
        return file.getvalue().strip()

    def mlir(self) -> str:
        return self._print(Printer.Target.MLIR)

    def xdsl(self) -> str:
        return self._print(Printer.Target.XDSL)

    def mlir_roundtrip(self, mlir_opt_path, mlir_opt_args=[]) -> str:
        """
        Runs 'mlir-opt' on the generated IR with specified arguments and returns the output as a string.
        """
        cmd = [mlir_opt_path] + mlir_opt_args
        ip = self._print(Printer.Target.MLIR).encode("utf-8")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, input=ip)
        return result.stdout.decode("utf-8").strip()
