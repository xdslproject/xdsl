import logging
import os

from inspect import getsource, getmodule
from ast import parse


class FrontendProgram:
    """
    Class to store the Python AST of a program written in the frontend. This
    program can be compiled, which performs the translation to MLIR.
    """

    def __init__(self, log_level=logging.INFO):
        """
        :param log_level: set verbosity of the logging.
        """
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger("frontend_program_logger")
        self.logger.setLevel(log_level)

    def compile(self) -> None:
        """
        Compile the function to MLIR (but not yet specify input/output values).
        The compiled main function is stored internally in `self.cpp_program`.
        """

        try:
            raise NotImplementedError("Compilation is not yet implemented.")
        except Exception as e:
            self.logger.error(e)
            exit(1)

    def set_src(self, parent_frame):
        """
        Stores the source information. 

        self.src_context:   stores the code of the functions on the call stack
        self.src_code_ast:  stores the src code stores the source code of the 
                            entire file from which the CodeContext was called from.

        :param parent_frame:    code frame of the CodeContext With block that contains
                                the frontend code.
        """

        self.src_module = getmodule(parent_frame)
        self.src_context = getsource(parent_frame)
        self.src_call_stack_ast = parse(self.src_context)

        self.src_file_path = os.path.realpath(self.src_module.__file__)
        with open(self.src_file_path, "r") as src_fp:
            self.src_code = src_fp.read()
        self.src_code_ast = parse(self.src_code)
