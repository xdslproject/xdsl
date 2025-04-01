from dataclasses import dataclass, field
from typing import Any, NamedTuple, TypeAlias

from xdsl.ir import Attribute, Operation

TypeName: TypeAlias = str


class SourceIRTypePair(NamedTuple):
    """Pair of types for source code and its generated IR."""

    source: type
    ir: type[Attribute]


class TypeMethodPair(NamedTuple):
    """Pair of type and method for source code."""

    type_: type
    method: str


@dataclass
class TypeConverter:
    """Responsible for conversion of Python type hints to xDSL types."""

    type_registry: dict[TypeName, SourceIRTypePair] = field(default_factory=dict)
    """Mappings between source code and IR type, indexed by name."""

    method_registry: dict[TypeMethodPair, type[Operation]] = field(default_factory=dict)
    """Mappings between methods on objects and their operations."""

    globals: dict[str, Any] = field(default_factory=dict)
    """
    Stores all globals in the current Python program, including imports. This is
    useful because we can lookup a class which corresponds to the type
    annotation without explicitly constructing it.
    """

    def get_method(
        self,
        ir_type: type[Attribute],
        method: str,
    ) -> type[Operation] | None:
        """Get the method attribute type from a type and method name."""
        for source, ir in self.type_registry.values():
            if ir == ir_type:
                return self.method_registry[TypeMethodPair(source, method)]
        return None

    def get_ir_type(
        self,
        source_type: TypeName,
    ) -> Attribute:
        """Get the ir type by its source code type name"""
        return self.type_registry[source_type].ir()
