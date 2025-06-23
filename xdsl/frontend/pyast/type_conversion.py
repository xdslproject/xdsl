import ast
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    Any,
    _GenericAlias,  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue]
)

from typing_extensions import TypeForm

import xdsl.dialects.builtin as xdsl_builtin
import xdsl.frontend.pyast.dialects.builtin as frontend_builtin
from xdsl.frontend.pyast.dialects.builtin import (
    _FrontendType,  # pyright: ignore[reportPrivateUsage]
)
from xdsl.frontend.pyast.exception import (
    CodeGenerationException,
    FrontendProgramException,
)
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

    def get_callable(
        self, operation_type: type[Operation]
    ) -> Callable[..., Any] | None:
        """Get the Python Callable from an IR operation type.

        This supports many-to-one mappings by resolving greedily on the first
        mapping inserted.
        """
        for key, value in self._mapping.items():
            if value == operation_type:
                return key
        return None

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
        function = importlib.import_module(module_name)
        for attr in function_name.split("."):
            function = getattr(function, attr, None)
        if function is None:
            raise FrontendProgramException(
                f"Unable to resolve function '{module_name}.{function_name}'"
            )
        assert callable(function)  # Guaranteed by types a registration time
        if (operation_type := self.get_operation_type(function)) is not None:
            return operation_type(*args, **kwargs)
        return None


class TypeRegistry:
    """A mapping between Python type annotations and IR type attributes."""

    def __init__(self):
        """Instantiate the function registry."""
        self._mapping: dict[type | TypeForm[Attribute], TypeAttribute] = {}
        self._type_names: dict[str, type | TypeForm[Attribute]] = {}

    def insert(
        self, annotation: type | TypeForm[Attribute], attribute: TypeAttribute
    ) -> None:
        """Insert a relation between a Python type annotation and an IR type attribute."""
        # Enforce type is not generic/final if not subclass attribute
        # Resolve attributes
        annotation_name = annotation.__qualname__
        if annotation_name in self._type_names:
            raise FrontendProgramException(
                f"Cannot re-register type name '{annotation_name}'"
            )
        self._type_names[annotation_name] = annotation
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

    def get_attribute(
        self, annotation: type | TypeForm[Attribute]
    ) -> TypeAttribute | None:
        """Get the Python type annotation from an IR type attribute."""
        return self._mapping.get(annotation, None)

    def resolve_attribute(self, annotation_name: str) -> TypeAttribute | None:
        """Get an IR type attribute from a string annotation."""
        annotation = self._type_names.get(annotation_name, None)
        if annotation is None:
            return None
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

    name_to_xdsl_type_map: dict[str, Attribute] = field(
        default_factory=dict[str, Attribute]
    )
    """
    Map to cache xDSL types created so far to avoid repeated conversions.
    """

    xdsl_to_frontend_type_map: dict[type[Attribute], type[_FrontendType]] = field(
        default_factory=dict[type[Attribute], type[_FrontendType]]
    )
    """
    Map to lookup frontend types based on xDSL type. Useful if we want to see
    what overloaded Python operations does this xDSL type support.
    """

    file: str | None = field(default=None)

    def __post_init__(self) -> None:
        # Cache index type because it is always used implicitly in loops and
        # many other IR constructs.
        index = frontend_builtin._Index  # pyright: ignore[reportPrivateUsage]
        self._cache_type(index, xdsl_builtin.IndexType(), "index")

    def _cache_type(
        self,
        frontend_type: type[_FrontendType],
        xdsl_type: Attribute,
        type_name: str,
    ) -> None:
        """Records frontend and corresponding xDSL types in cache."""
        if type_name not in self.name_to_xdsl_type_map:
            self.name_to_xdsl_type_map[type_name] = xdsl_type
        if xdsl_type.__class__ not in self.xdsl_to_frontend_type_map:
            self.xdsl_to_frontend_type_map[xdsl_type.__class__] = frontend_type

    def _convert_name(self, type_hint: ast.Name) -> Attribute:
        # First, check if we have already converted this type hint.
        type_name = type_hint.id
        if type_name in self.name_to_xdsl_type_map:
            return self.name_to_xdsl_type_map[type_name]

        # Otherwise, it must be some frontend type, and we can look up its class
        # using the imports.
        if type_name not in self.globals:
            raise CodeGenerationException(
                self.file,
                type_hint.lineno,
                type_hint.col_offset,
                f"Unknown type hint '{type_name}'.",
            )
        type_class = self.globals[type_name]

        # First, type can be generic, e.g. `class _Integer(Generic[_W, _S])`.
        if isinstance(type_class, _GenericAlias):
            generic_type_arguments = type_class.__args__
            arguments_for_constructor: list[Any] = []
            for type_argument in generic_type_arguments:
                # Convert Literal[...] to concrete values.
                materialized_arguments = type_argument.__args__
                if len(materialized_arguments) != 1:
                    raise CodeGenerationException(
                        self.file,
                        type_hint.lineno,
                        type_hint.col_offset,
                        f"Expected 1 type argument for generic type '{type_name}', got "
                        f"{len(materialized_arguments)} type arguments instead.",
                    )
                arguments_for_constructor.append(materialized_arguments[0])
                continue

            # Finally, get the constructor of this type and build an xDSL type.
            if issubclass(type_class.__origin__, _FrontendType):
                xdsl_type = type_class.to_xdsl()(*arguments_for_constructor)
                self._cache_type(type_class.__origin__, xdsl_type, type_name)
                return xdsl_type

            # If this is not a subclass of FrontendType, then abort.
            raise CodeGenerationException(
                self.file,
                type_hint.lineno,
                type_hint.col_offset,
                f"'{type_name}' is not a frontend type.",
            )

        # Otherwise, type can be a simple non-generic frontend type, e.g. `class
        # _Index(FrontendType)`.
        if issubclass(type_class, _FrontendType):
            xdsl_type = type_class.to_xdsl()()
            self._cache_type(type_class, xdsl_type, type_name)
            return xdsl_type

        raise CodeGenerationException(
            self.file,
            type_hint.lineno,
            type_hint.col_offset,
            f"Unknown type hint for type '{type_name}' inside 'ast.Name' expression.",
        )

    def convert_type_hint(self, type_hint: ast.expr) -> Attribute:
        """Convertes a Python/frontend type given as AST expression into xDSL type."""

        # Type hint should always be provided if this function is called.
        assert type_hint is not None

        # TODO: Type hint can be a Subscript AST node, for example
        # `Foo[Literal[2]]``. Support this in the future patches.
        if isinstance(type_hint, ast.Subscript):
            raise CodeGenerationException(
                self.file,
                type_hint.lineno,
                type_hint.col_offset,
                "Converting subscript type hints is not supported.",
            )

        # Type hint can also be a TypeAlias. For example, one can define
        # `foo = Foo[Literal[2]]`. This case also handles standard Python types, like
        # int, float, etc.
        if isinstance(type_hint, ast.Name):
            return self._convert_name(type_hint)

        raise CodeGenerationException(
            self.file,
            type_hint.lineno,
            type_hint.col_offset,
            f"Unknown type hint AST node '{type_hint}'.",
        )
