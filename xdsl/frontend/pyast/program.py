from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Generic, ParamSpec

from typing_extensions import TypeVar

from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.utils.builder import PyASTBuilder

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class PyASTProgram(Generic[P, R]):
    """Wrapper to associate an IR representation with a Python function."""

    name: Final[str]
    """The name of the function describing the program."""

    func: Final[Callable[P, R]]
    """A callable object for the function describing the program."""

    _builder: Final[PyASTBuilder]
    """An internal object to contextually build an IR module from the function."""

    _module: ModuleOp | None = None
    """An internal object to cache the built IR module."""

    @property
    def module(self) -> ModuleOp:
        """Lazily build the module when required, once."""
        if self._module is None:
            self._module = self._builder.build()
        return self._module

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Pass through calling the object to its Python implementation."""
        return self.func(*args, **kwargs)
