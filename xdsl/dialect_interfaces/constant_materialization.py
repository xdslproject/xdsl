from abc import ABC, abstractmethod

from xdsl.dialect_interfaces import DialectInterface
from xdsl.ir import Attribute, Operation


class ConstantMaterializationInterface(DialectInterface, ABC):
    """
    An interface for dialects that support constant materialization.

    A dialect that implements this interface should provide the `materialize_constant` method,
    which creates a constant operation of the dialect given a value and a type.

    This is useful for transformations that need to create constants in a dialect-specific way.
    """

    @abstractmethod
    def materialize_constant(
        self, value: Attribute, type: Attribute
    ) -> Operation | None:
        """
        Materializes a constant operation in the dialect.

        Args:
            value (Attribute): The attribute representing the constant value.
            type (Attribute): The type of the constant.

        Returns:
            Operation: The created constant operation.
        """
        raise NotImplementedError("Dialect does not implement materialize_constant")
