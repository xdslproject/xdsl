
import logging
import ast

from contextlib import AbstractContextManager
from inspect import getsource
from sys import _getframe

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
                self.logger.debug(
                    f"Start parsing With block at line {item.lineno}")

                # First pass: gather block labels
                # XXX: attention, block labels are global in the frontend.
                self.logger.info("Start first pass over pyAST.")
                block_label_visitor = ScopingVisitor(
                    parent_globals, self.logger)
                for line in item.body:
                    block_label_visitor.visit(line)
                program_state = block_label_visitor.state
                program_state.reset()

                # Second pass: translate frontend pyast to xDSL
                visitor = FrontendToXDSL(
                    parent_globals, program_state, self.logger)
                stmts = []
                self.logger.info("Start second pass over pyAST.")
                for line in item.body:
                    self.logger.debug(
                        f"Parsing the following Python AST:\n{ast.dump(line)}")
                    stmts.append(visitor.visit(line))

                visitor.state.finalize_region(stmts)
                visitor.state.finalize_module()

                printer = Printer()
                for module_metadata in program_state.module_metadata_stack:
                    printer.print_op(module_metadata.module)

    def __exit__(self, *args):
        self.logger.debug("Exit with block.")
