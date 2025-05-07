import ast
import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    Any,
    TypeAlias,
    _GenericAlias,  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue]
)

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

TypeName: TypeAlias = str
FunctionRegistry: TypeAlias = dict[Callable[..., Any], type[Operation]]


class TypeRegistry(dict[type, TypeAttribute]):
    """Mappings between source code and IR type.

    This mapping must be one-to-one, with each source type having only IR type.
    This is to ensure that the lowering then reconstructing an AST is
    idempotent, as with a many-to-one mapping the source type to reconstruct
    cannot necessarily be correctly selected.
    """

    def valid_insert(self, key: type, value: TypeAttribute) -> bool:
        """Check that both the key and value are unique."""
        return key not in self and value not in self.values()

    def get_backwards(self, lookup: TypeAttribute) -> type | None:
        """Get a dictionary mapping values to keys."""
        for key, value in self.items():
            if value == lookup:
                return key
        return None


@dataclass
class TypeConverter:
    """Responsible for conversion of Python type hints to xDSL types."""

    globals: dict[str, Any] = field(default_factory=dict[str, Any])
    """
    Stores all globals in the current Python program, including imports. This is
    useful because we can lookup a class which corresponds to the type
    annotation without explicitly constructing it.
    """

    type_names: dict[TypeName, type] = field(default_factory=dict[TypeName, type])
    """Mappings from source type names to source types."""

    type_registry: TypeRegistry = field(default_factory=TypeRegistry)
    """Mappings between source code and ir type, indexed by name."""

    function_registry: FunctionRegistry = field(default_factory=FunctionRegistry)
    """Mappings between methods on objects and their operations."""

    name_to_xdsl_type_map: dict[TypeName, Attribute] = field(
        default_factory=dict[TypeName, Attribute]
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
        type_name: TypeName,
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

    def get_ir_type(
        self,
        source_type_name: TypeName,
    ) -> TypeAttribute | None:
        """Get the IR type by its source code type name.

        Normally, the attribute is a class which can be instantiated with no
        parameters. However, in some cases it is parameterised, such as
        `IntegerType` with its bitwidth. In this case, `Annotated` types such as
        `I1` defined as `Annotated[IntegerType, IntegerType(1)]` are provided
        already by xDSL, so we can extract the attribute instance from this.
        """
        if source_type_name not in self.type_names:
            return None
        source_type = self.type_names[source_type_name]
        if source_type not in self.type_registry:
            return None
        return self.type_registry[source_type]

    def get_source_type(self, ir_type: TypeAttribute) -> type | None:
        """Get the source type from its IR type."""
        return self.type_registry.get_backwards(ir_type)

    def resolve_function(
        self,
        module_name: str,
        function_name: str,
    ) -> Callable[..., Any]:
        """Resolve a function in the current namespace."""
        function = importlib.import_module(module_name)
        for attr in function_name.split("."):
            function = getattr(function, attr, None)
        if function is None:
            raise FrontendProgramException(
                f"Unable to resolve function '{module_name}.{function_name}'"
            )
        assert callable(function)  # Guaranteed by types a registration time
        return function

    def get_operation(
        self,
        method: Callable[..., Any],
        args: tuple[SSAValue[Attribute], ...] = tuple(),
        kwargs: dict[str, SSAValue[Attribute]] = dict(),
    ) -> Operation | None:
        """Get the method attribute type from a type and method name."""
        if method in self.function_registry:
            return self.function_registry[method].__call__(*args, **kwargs)
        return None
