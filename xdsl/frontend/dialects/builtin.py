import xdsl.dialects.builtin as xdsl

from typing import Generic, TypeAlias, TypeVar, Literal
from xdsl.dialects.builtin import Signedness


class FrontendType:
    """Represents ay type in the frontend."""

    def to_xdsl():
        pass


W = TypeVar("W", bound=int, covariant=True)
S = TypeVar("S", bound=xdsl.Signedness, covariant=True)


class IntegerType(Generic[W, S], FrontendType):
    """Represents an integer type in the frontend."""

    def to_xdsl():
        return xdsl.IntegerType.from_width


# Type aliases for signless integers.
i1: TypeAlias = IntegerType[Literal[1], Literal[Signedness.SIGNLESS]]
i32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.SIGNLESS]]
i64: TypeAlias = IntegerType[Literal[64], Literal[Signedness.SIGNLESS]]

# Type aliases for signed integers.
si32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.SIGNED]]
si64: TypeAlias = IntegerType[Literal[64], Literal[Signedness.SIGNED]]

# Type aliases for unsigned integers.
ui32: TypeAlias = IntegerType[Literal[32], Literal[Signedness.UNSIGNED]]
ui64: TypeAlias = IntegerType[Literal[64], Literal[Signedness.UNSIGNED]]


class IndexType(FrontendType):
    """Represents an index type in the frontend."""

    def to_xdsl():
        return xdsl.IndexType


# Type alias for index type.
index: TypeAlias = IndexType


class Module:
    """Represents a builtin.module."""
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
