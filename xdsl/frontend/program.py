import ast
import subprocess

from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.codegen.codegen_visitor import CodeGeneration
from xdsl.frontend.codegen.type_conversion import TypeConverter
from xdsl.passes.desymref import DesymrefyPass
from xdsl.printer import Printer


@dataclass
class FrontendProgram:
    """
    Class to store the Python AST of a program written in the frontend. This
    program can be compiled and translated to xDSL/MLIR.
    """

    stmts: List[ast.stmt] = field(init=False)
    """AST nodes stored for compilation to xDSL."""

    globals: Dict[str, Any] = field(init=False)
    """Global information for this program, including all the imports."""

    xdsl_program: ModuleOp = field(init=False)
    """Generated xDSL program when AST is compiled."""

    def compile(self, desymref=True) -> None:
        """Generates xDSL from the source program."""
        self.xdsl_program = CodeGeneration.run_with_type_converter(TypeConverter(self.globals), self.stmts)
        self.xdsl_program.verify()

        # Optionally run desymrefication pass to produce actual SSA.
        if desymref:
            self.desymref()

    def desymref(self) -> None:
        """Desymrefy the generated xDSL."""
        DesymrefyPass.run(self.xdsl_program)
        self.xdsl_program.verify()

    def _print(self, target) -> str:
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
