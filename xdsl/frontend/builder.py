from attr import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.frontend.pyast.utils.type_conversion import FunctionRegistry, TypeRegistry


@dataclass
class PyASTBuilder:
    """Build an operation ."""

    type_registry: TypeRegistry
    function_registry: FunctionRegistry
    desymref: bool

    @property
    def module(self) -> ModuleOp:
        return ModuleOp([])
