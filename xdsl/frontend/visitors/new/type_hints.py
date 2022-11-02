import ast

from dataclasses import dataclass, field
from typing import _GenericAlias, Any, Dict, Optional, Type
from xdsl.dialects.builtin import Signedness, SignednessAttr
from xdsl.ir import Attribute

@dataclass
class TypeHintException(Exception):
    """
    Exception type if type conversion failed or is not supported.
    """
    msg: str

    def __str__(self) -> str:
        return f"Type conversion failed with: {self.msg}."


@dataclass
class TypeHintToXDSL:
    """Class responsible for cobersion of Python type hints to concrete xDSL types."""

    imports: Dict[str, Any]
    """Stores all imports in the current Python program."""

    type_cache: Dict[str, Attribute] = field(default_factory=dict)
    """Caches all xDSL types we have created so far to avoid repeated conversions."""

    def _convert_name(self, hint: ast.Name) -> Attribute:
        # First, get check if we have already converted this type hint.
        ty_name: str = hint.id
        if ty_name in self.type_cache:
            return self.type_cache[ty_name]
        
        ty = self.imports[ty_name]
        
        # If the type is a generic type, go through the type arguments and materialize them.
        if isinstance(ty, _GenericAlias):
            args = []
            for ty_arg in ty.__args__:
                # TODO: there shpuld be a materializer!
                v = ty_arg.__args__[0]
                args.append(v)
            
            # Finally, get the constructor of this type and build an xDSL type.
            # TODO: we assume top-level type to be frontend type!
            constructor = ty.to_xdsl()
            return constructor(*args)
        
        # TODO: what is not generic type? We can enfore it probably.
        
        # Otherwise, abort.
        raise TypeHintException(f"{type(ty)} is not an instance of {_GenericAlias}")

    def convert_hint(self, hint: Type) -> Optional[Attribute]:
        """Entry-point for all hint conversions."""
        
        # Type hint can be not provided, e.g. when returning None from the function implicitly. Then
        # simply return None and the caller should decide what to do next.
        if hint is None:
            return None
        
        # In general, a type hint is a Subscript AST node, e.g. Foo[Literal[2]]. For now,
        # let's not support it and instead ask user to define a TypeAlias.
        if isinstance(hint, ast.Subscript):
            raise TypeHintException(f"hints as subscripts are not supported, try to convert to type alias instead, e.g. x: TypeAlias = {type(hint)}")

        # Type hint can also be a TypeAlias, e.g. one can define foo = Foo[Literal[2]].
        if isinstance(hint, ast.Name):
            return self._convert_name(hint)

        # In all other cases, abort.
        raise TypeHintException(f"unknown hint of type {type(hint)}")
