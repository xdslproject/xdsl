import ast
import subprocess

from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.codegen.codegen_visitor import CodegenVisitor
from xdsl.frontend.codegen.functions import FunctionVisitor
from xdsl.passes.desymref import DesymrefyPass
from xdsl.printer import Printer


@dataclass
class FrontendProgram:
    """
    Class to store the Python AST of a program written in the frontend. This
    program can be compiled and translated to xDSL.
    """

    stmts: List[ast.stmt] = field(init=False)
    """AST nodes stored for compilation to xDSL."""

    globals: Dict[str, Any] = field(init=False)
    """Global information for this prgram, including all the imports."""

    xdsl_program: ModuleOp = field(init=False)
    """Generated xDSL program when AST is compiled."""

    def compile(self, desymref=True):
        """Generates xDSL from the source program."""

        # TODO: what about cross-module functions? Support them later. For now,
        # let's store all of them.
        func_visitor = FunctionVisitor(self.globals)
        for stmt in self.stmts:
            func_visitor.visit(stmt)

        # Run code generation.
        visitor = CodegenVisitor(self.globals, func_visitor.functions)
        for stmt in self.stmts:
            visitor.visit(stmt)
        ops = visitor.inserter.op_container

        # Ensure that the code is encapsulated in a single module.
        if len(ops) == 1 and isinstance(ops[0], ModuleOp):
            self.xdsl_program = ops[0]
        else:
            self.xdsl_program = ModuleOp.from_region_or_ops(ops)

        # Verify the generated code.
        self.xdsl_program.verify()

        # Optionally run desymrefication pass to produce actual SSA.
        if desymref:
            self.desymref()

    def desymref(self):
        """Desymrefy the generated xDSL."""
        DesymrefyPass.run(self.xdsl_program)
        self.xdsl_program.verify()

    def print(self, target):
        file = StringIO("")
        printer = Printer(stream=file, target=target)
        printer.print_op(self.xdsl_program)
        return file.getvalue().strip()

    def mlir(self):
        return self.print(Printer.Target.MLIR)

    def xdsl(self):
        return self.print(Printer.Target.XDSL)

    def mlir_roundtrip(self, mlir_opt_path, args=[]):
        cmd = [mlir_opt_path] + args
        ip = self.print(Printer.Target.MLIR).encode("utf-8")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, input=ip)
        return result.stdout.decode("utf-8").strip()
