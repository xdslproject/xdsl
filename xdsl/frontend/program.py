import ast

from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Dict, List
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp
from xdsl.frontend.codegen.codegen_visitor import CodegenVisitor
from xdsl.passes.desymref import DesymrefPass
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

    def compile(self):
        """Generates xDSL from the source program."""

        # Run code generation.
        visitor = CodegenVisitor(self.globals)
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

    def optimize(self):
        """Optimized the generated xDSL."""

        # TODO: for now this only runs desymrefication.
        for op in self.xdsl_program.body.ops:
            # TODO: desymref runs on function at the moment.
            if isinstance(op, FuncOp):
                DesymrefPass.run(op)

    def __str__(self):
        """Printing support of generated xDSL."""
        file = StringIO("")
        printer = Printer(stream=file)
        printer.print_op(self.xdsl_program)
        return file.getvalue().strip()
