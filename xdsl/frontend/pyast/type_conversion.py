import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from typing_extensions import TypeForm

from xdsl.frontend.pyast.exception import FrontendProgramException
from xdsl.ir import Attribute, Operation, SSAValue, TypeAttribute


class FunctionRegistry:
    """A mapping between Python callables and IR operation types."""

    def __init__(self):
        """Instantiate the function registry."""
        self._mapping: dict[Callable[..., Any], type[Operation]] = {}

    def insert(
        self, callable: Callable[..., Any], operation_type: type[Operation]
    ) -> None:
        """Insert a relation between a Python callable and an IR operation type."""
        if callable in self._mapping:
            raise FrontendProgramException(
                f"Cannot re-register function '{callable.__qualname__}'"
            )
        self._mapping[callable] = operation_type

    def get_operation_type(
        self, callable: Callable[..., Any]
    ) -> type[Operation] | None:
        """Get the IR operation type from a Python callable"""
        return self._mapping.get(callable, None)

    def resolve_operation(
        self,
        module_name: str,
        function_name: str,
        args: tuple[SSAValue[Attribute], ...] = tuple(),
        kwargs: dict[str, SSAValue[Attribute]] = dict(),
    ) -> Operation | None:
        """Get a concrete IR operation from a function name and its arguments."""
        # Start at the module, and walk till a function leaf is found
        function = importlib.import_module(module_name)
        for attr in function_name.split("."):
            function = getattr(function, attr, None)
        if function is None:
            return None
        assert callable(function), "Guaranteed by type signature of registration method"
        if (operation_type := self.get_operation_type(function)) is not None:
            return operation_type(*args, **kwargs)
        return None


class TypeRegistry:
    """A mapping between Python type annotations and IR type attributes."""

    def __init__(self):
        """Instantiate the function registry."""
        self._mapping: dict[type | TypeForm[Attribute], TypeAttribute] = {}

    def insert(
        self,
        annotation: type | TypeForm[Attribute],
        attribute: TypeAttribute,
    ) -> None:
        """Insert a relation between a Python type annotation and an IR type attribute."""
        # Enforce type is not generic/final if not subclass attribute
        if annotation in self._mapping:
            raise FrontendProgramException(
                f"Cannot re-register type name '{annotation.__qualname__}'"
            )
        self._mapping[annotation] = attribute

    def get_annotation(
        self, attribute: TypeAttribute
    ) -> type | TypeForm[Attribute] | None:
        """Get the Python type annotation from an IR type attribute.

        This supports many-to-one mappings by resolving greedily on the first
        mapping inserted.
        """
        for key, value in self._mapping.items():
            if value == attribute:
                return key
        return None

    def resolve_attribute(
        self, annotation_name: str, globals: dict[str, Any]
    ) -> TypeAttribute | None:
        """Get an IR type attribute from a string annotation."""
        annotation = cast(
            type | TypeForm[Attribute],
            eval(annotation_name, globals, None),
        )
        return self._mapping.get(annotation, None)


@dataclass
class TypeConverter:
    """Responsible for conversion of Python type hints to xDSL types."""

    globals: dict[str, Any] = field(default_factory=dict[str, Any])
    """
    Stores all globals in the current Python program, including imports. This is
    useful because we can lookup a class which corresponds to the type
    annotation without explicitly constructing it.
    """

    type_registry: TypeRegistry = field(default_factory=TypeRegistry)
    """Mappings between source code and ir type, indexed by name."""

    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)
    """Mappings between methods on objects and their operations."""

    file: str | None = field(default=None)
