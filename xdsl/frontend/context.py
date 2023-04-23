import ast

from contextlib import AbstractContextManager
from dataclasses import dataclass
from inspect import getsource
from sys import _getframe  # type: ignore
from typing import Any

from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.python_code_check import PythonCodeCheck


@dataclass
class CodeContext(AbstractContextManager[Any]):
    """
    The CodeContext with block marks the scope in which the code in the custom
    DSL can be written. This code will be translated to xDSL/MLIR.
    """

    program: FrontendProgram
    """
    Underlying frontend program which can be compiled and translated to
    xDSL/MLIR.
    """

    def __enter__(self) -> None:
        # First, get the Python AST from the code.
        frame = _getframe(1)
        self.program.file = frame.f_code.co_filename
        src = getsource(frame)
        python_ast = ast.parse(src)

        # Get all the global information and record it as well. In particular,
        # this contains all the imports.
        self.program.globals = frame.f_globals

        # Find where the program starts.
        for node in ast.walk(python_ast):
            if (
                isinstance(node, ast.With)
                and node.lineno == frame.f_lineno - frame.f_code.co_firstlineno + 1
            ):
                # Found the program AST. Store it for later compilation or
                # execution.
                self.program.stmts = node.body

    def __exit__(self, *args: Any):
        # Having proccessed all the code in the context, check it is well-formed
        # and can be compiled/executed.
        assert self.program.stmts is not None
        PythonCodeCheck.run(self.program.stmts, self.program.file)
