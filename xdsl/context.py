import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from xdsl.ir import Attribute, Dialect, Operation


@dataclass
class Context:
    """Contains structures for operations/attributes registration."""

    allow_unregistered: bool = field(default=False)

    _loaded_dialects: dict[str, Dialect] = field(default_factory=dict[str, Dialect])
    _loaded_ops: dict[str, type[Operation]] = field(
        default_factory=dict[str, type[Operation]]
    )
    _loaded_attrs: dict[str, type[Attribute]] = field(
        default_factory=dict[str, type[Attribute]]
    )
    _registered_dialects: dict[str, Callable[[], Dialect]] = field(
        default_factory=dict[str, Callable[[], Dialect]]
    )
    """
    A dictionary of all registered dialects that are not yet loaded. This is used to
    only load the respective Python files when the dialect is actually used.
    """

    def clone(self) -> "Context":
        return Context(
            self.allow_unregistered,
            self._loaded_dialects.copy(),
            self._loaded_ops.copy(),
            self._loaded_attrs.copy(),
            self._registered_dialects.copy(),
        )

    @property
    def loaded_ops(self) -> "Iterable[type[Operation]]":
        """
        Returns all the loaded operations. Not valid across mutations of this object.
        """
        return self._loaded_ops.values()

    @property
    def loaded_attrs(self) -> "Iterable[type[Attribute]]":
        """
        Returns all the loaded attributes. Not valid across mutations of this object.
        """
        return self._loaded_attrs.values()

    @property
    def loaded_dialects(self) -> "Iterable[Dialect]":
        """
        Returns all the loaded attributes. Not valid across mutations of this object.
        """
        return self._loaded_dialects.values()

    @property
    def registered_dialect_names(self) -> Iterable[str]:
        """
        Returns the names of all registered dialects. Not valid across mutations of this object.
        """
        return self._registered_dialects.keys()

    def register_dialect(
        self, name: str, dialect_factory: "Callable[[], Dialect]"
    ) -> None:
        """
        Register a dialect without loading it. The dialect is only loaded in the context
        when an operation or attribute of that dialect is parsed, or when explicitely
        requested with `load_registered_dialect`.
        """
        if name in self._registered_dialects:
            raise ValueError(f"'{name}' dialect is already registered")
        self._registered_dialects[name] = dialect_factory

    def load_registered_dialect(self, name: str) -> None:
        """Load a dialect that is already registered in the context."""
        if name not in self._registered_dialects:
            raise ValueError(f"'{name}' dialect is not registered")
        dialect = self._registered_dialects[name]()
        self._loaded_dialects[dialect.name] = dialect

        for op in dialect.operations:
            self.load_op(op)

        for attr in dialect.attributes:
            self.load_attr(attr)

    def load_dialect(self, dialect: "Dialect"):
        """
        Load a dialect. Operation and Attribute names should be unique.
        If the dialect is already registered in the context, use
        `load_registered_dialect` instead.
        """
        if dialect.name in self._registered_dialects:
            raise ValueError(
                f"'{dialect.name}' dialect is already registered, use 'load_registered_dialect' instead"
            )
        self.register_dialect(dialect.name, lambda: dialect)
        self.load_registered_dialect(dialect.name)

    def load_op(self, op: "type[Operation]") -> None:
        """Load an operation definition. Operation names should be unique."""
        if op.name in self._loaded_ops:
            raise Exception(f"Operation {op.name} has already been loaded")
        self._loaded_ops[op.name] = op

    def load_attr(self, attr: "type[Attribute]") -> None:
        """Load an attribute definition. Attribute names should be unique."""
        if attr.name in self._loaded_attrs:
            raise Exception(f"Attribute {attr.name} has already been loaded")
        self._loaded_attrs[attr.name] = attr

    def _get_known_op(self, name: str) -> "type[Operation] | None":
        if name in self._loaded_ops:
            return self._loaded_ops[name]
        if "." in name:
            dialect_name, _ = Dialect.split_name(name)
            if (
                dialect_name in self._registered_dialects
                and dialect_name not in self._loaded_dialects
            ):
                self.load_registered_dialect(dialect_name)
                return self._get_known_op(name)

    def get_optional_op(
        self, name: str, *, dialect_stack: Sequence[str] = ()
    ) -> "type[Operation] | None":
        """
        Get an operation class from its name if it exists or is contained in one of the
        dialects in the dialect stack.
        If the operation is not registered, return None unless unregistered operations
        are allowed in the context, in which case return an UnregisteredOp.
        """
        # Check if the name is known.
        if op_type := self._get_known_op(name):
            return op_type

        # Check appending each dialect in the dialect stack.
        for dialect_name in reversed(dialect_stack):
            dialect_and_name = f"{dialect_name}.{name}"
            if op_type := self._get_known_op(dialect_and_name):
                return op_type

        # If the context allows unregistered operations then create an UnregisteredOp
        if self.allow_unregistered:
            from xdsl.dialects.builtin import UnregisteredOp

            op_type = UnregisteredOp.with_name(name)
            self._loaded_ops[name] = op_type
            return op_type

    def get_op(
        self, name: str, *, dialect_stack: Sequence[str] = ()
    ) -> "type[Operation]":
        """
        Get an operation class from its name if it exists or is contained in one of the
        dialects in the dialect stack.
        If the operation is not registered, raise an exception unless unregistered
        operations are allowed in the context, in which case return an UnregisteredOp.
        """
        if op_type := self.get_optional_op(name, dialect_stack=dialect_stack):
            return op_type
        raise Exception(f"Operation {name} is not registered")

    def get_optional_attr(
        self,
        name: str,
        create_unregistered_as_type: bool = False,
    ) -> "type[Attribute] | None":
        """
        Get an attribute class from its name if it exists.
        If the attribute is not registered, return None unless unregistered attributes
        are allowed in the context, in which case return an UnregisteredAttr.
        Since UnregisteredAttr may be a type (for MLIR compatibility), an
        additional flag is required to create an UnregisterAttr that is
        also a type.
        """
        # If the attribute is already loaded, returns it.
        if name in self._loaded_attrs:
            return self._loaded_attrs[name]

        # Otherwise, check if the attribute dialect is registered.
        dialect_name, _ = Dialect.split_name(name)
        if (
            dialect_name in self._registered_dialects
            and dialect_name not in self._loaded_dialects
        ):
            self.load_registered_dialect(dialect_name)
            return self.get_optional_attr(name)

        # If the dialect is unregistered, but the context allows unregistered
        # attributes, return an UnregisteredOp.
        if self.allow_unregistered:
            from xdsl.dialects.builtin import UnregisteredAttr

            attr_type = UnregisteredAttr.with_name_and_type(
                name, create_unregistered_as_type
            )
            self._loaded_attrs[name] = attr_type
            return attr_type

        return None

    def get_attr(
        self,
        name: str,
        create_unregistered_as_type: bool = False,
    ) -> "type[Attribute]":
        """
        Get an attribute class from its name.
        If the attribute is not registered, raise an exception unless unregistered
        attributes are allowed in the context, in which case return an UnregisteredAttr.
        Since UnregisteredAttr may be a type (for MLIR compatibility), an
        additional flag is required to create an UnregisterAttr that is
        also a type.
        """
        if attr_type := self.get_optional_attr(name, create_unregistered_as_type):
            return attr_type
        raise Exception(f"Attribute {name} is not registered")

    def get_dialect(self, name: str) -> "Dialect":
        if (dialect := self.get_optional_dialect(name)) is None:
            raise Exception(f"Dialect {name} is not registered")
        return dialect

    def get_optional_dialect(self, name: str) -> "Dialect | None":
        if name in self._loaded_dialects:
            return self._loaded_dialects[name]
        return None


class MLContext(Context):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        warnings.warn("MLContext is deprecated, please use Context instead")
