
import logging
import ast

from contextlib import AbstractContextManager
from inspect import getsource
from sys import _getframe
from xdsl.frontend.visitors.new.ast_visitor import ASTToXDSL
from xdsl.frontend.visitors.new.xdsl_program import XDSLProgram

from xdsl.printer import Printer
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.visitors.scoping import ScopingVisitor
from xdsl.frontend.visitors.frontend_to_xdsl import FrontendToXDSL


class CodeContext(AbstractContextManager):
    """
    The Frontend with block marks the scope in which code in the custom DSL can be
    written. This code will be translated to MLIR.
    """

    def __init__(self, prog: FrontendProgram, log_level=logging.INFO):
        """
        :param prog: The frontend program object that should store the code derived
        from this CodeContext block.

        :param log_level: set verbosity of the logging.
        """

        self.prog = prog

        self.logger = logging.getLogger('heco_logger')
        self.logger.setLevel(log_level)

    def __enter__(self):
        # Get the source code of the parent frame
        parent_frame = _getframe(1)
        parent_globals = parent_frame.f_globals
        src = getsource(parent_frame)
        python_ast = ast.parse(src)

        # Store src code and AST in program. The src file path is
        # used to get the entire src code from the calling file
        # (and not only the functions that are on the call stack
        # to this function call).
        self.prog.set_src(parent_frame)

        # Find the current 'with' block in the source code
        for item in ast.walk(python_ast):
            if isinstance(item, ast.With) and item.lineno == parent_frame.f_lineno - parent_frame.f_code.co_firstlineno + 1:
                # self.logger.debug(f"Parsing the program from line {item.lineno + 1}")

                program = XDSLProgram()
                visitor = ASTToXDSL(parent_globals, program, self.logger)
                # self.logger.info("Start Avisiting the AST...")
                for node in item.body:
                    visitor.visit(node)

                printer = Printer()
                for module in program.modules:
                    printer.print_op(module)

    def __exit__(self, *args):
        pass
