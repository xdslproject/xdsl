import xdsl.dialects.builtin as xdsl

from typing import Generic, Tuple, TypeAlias, TypeVar, Literal
from xdsl.dialects.builtin import Signedness


class FrontendType:
    """Represents ay type in the frontend."""

    def to_xdsl():
        pass


_Width = TypeVar("_Width", bound=int, covariant=True)
_Signedness = TypeVar("_Signedness", bound=xdsl.Signedness, covariant=True)


class IntegerType(Generic[_Width, _Signedness], FrontendType):
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


class Float16Type(FrontendType):
    """Represents a 16-bit floating-point type in the frontend."""

    def to_xdsl():
        return xdsl.Float16Type


class Float32Type(FrontendType):
    """Represents a 32-bit floating-point type in the frontend."""

    def to_xdsl():
        return xdsl.Float32Type


class Float64Type(FrontendType):
    """Represents a 64-bit floating-point type in the frontend."""

    def to_xdsl():
        return xdsl.Float64Type


# Type alias for floating-point types.
f16: TypeAlias = Float16Type
f32: TypeAlias = Float32Type
f64: TypeAlias = Float64Type


_Shape = TypeVar("_Shape", bound=Tuple[int, ...], covariant=True)
_TensorElementType = TypeVar("_TensorElementType", bound=FrontendType, covariant=True)


class TensorType(Generic[_TensorElementType, _Shape], FrontendType):
    """Represents a tensor type in the frontend."""

    def to_xdsl():
        return xdsl.TensorType.from_type_and_list


class Module:
    """Represents a builtin.module."""
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
