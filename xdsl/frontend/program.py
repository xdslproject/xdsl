import ast
import subprocess

from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List
from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.codegen.codegen_visitor import CodegenVisitor
from xdsl.frontend.codegen.functions import LocalCallAnalyzer, LocalFunctionAnalyzer
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
        type_converter = TypeConverter(self.globals)

        # TODO: at the moment, compilation focuses on a nicely formed programs, i.e. a single
        # module with a number of functions. Technically, in xDSL/MLIR we can have nested
        # modules with nested functions. But for the purpose of the front-end, these are not
        # too important so we focus on more commion scenario.

        # First, analyze all functions.
        lfa = LocalFunctionAnalyzer(type_converter)
        for stmt in self.stmts:
            lfa.visit(stmt)

        # Analyze all function calls to make sure templates are instantiated.
        lca = LocalCallAnalyzer(lfa.function_infos)
        for stmt in self.stmts:
            lca.visit(stmt)

        # func_visitor = FunctionVisitor(type_converter)
        # for stmt in self.stmts:
        #     func_visitor.visit(stmt)

        # Generate templates.
        visitor = CodegenVisitor(type_converter, lfa.function_infos)
        for function_info in lfa.function_infos.values():
            if function_info.template_instantiation:
                visitor.visit(function_info.ast_node)

        # Run code generation.
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
