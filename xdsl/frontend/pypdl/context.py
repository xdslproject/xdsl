from collections.abc import Callable
from dataclasses import dataclass
from inspect import currentframe

from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pyast.program import P, PyASTProgram, R
from xdsl.frontend.pypdl.builder import PyPDLRewriteBuilder


@dataclass
class PyPDLContext(PyASTContext):
    """Encapsulate the mapping between Python and IR types and operations."""

    def pdl_rewrite(
        self,
        decorated_func: Callable[P, R],
    ) -> PyASTProgram[P, R]:
        """Get a program wrapper for a PDL rewrite by decorating a function."""
        func_file, func_globals, func_ast = self._get_func_info(
            currentframe(), decorated_func, None
        )
        builder = PyPDLRewriteBuilder(
            type_registry=self.type_registry,
            function_registry=self.function_registry,
            file=func_file,
            globals=func_globals,
            function_ast=func_ast,
            desymref=True,
        )
        return self._get_wrapped_program(decorated_func, builder)
