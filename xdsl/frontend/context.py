import ast

from contextlib import AbstractContextManager
from dataclasses import dataclass
from inspect import getsource
from sys import _getframe
from typing import Any, Dict
from xdsl.frontend.program import FrontendProgram


@dataclass
class CodeContext(AbstractContextManager):
    """
    The CodeContext with block marks the scope in which code in the custom DSL
    can be written. This code will be translated to xDSL.
    """

    program: FrontendProgram
    """
    Underlying fronted program which can be compiled and transalted to xDSL.
    """

    def __enter__(self):
        # First, get the Python AST from the code.
        frame = _getframe(1)
        src = getsource(frame)
        python_ast = ast.parse(src)

        # Get all the global information and record it as well. In particular,
        # this contains all the imports.
        self.program.globals: Dict[str, Any] = frame.f_globals

        # Find the where DSL program starts.
        for node in ast.walk(python_ast):
            if isinstance(node, ast.With) and \
               node.lineno == frame.f_lineno - frame.f_code.co_firstlineno + 1:

                # Found the program AST. Store it for later complation.
                self.program.stmts = node.body

    def __exit__(self, *args):
        pass
