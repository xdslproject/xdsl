import xdsl.dialects.builtin as builtin

from typing import Generic, Tuple, TypeAlias, TypeVar, Literal
from xdsl.dialects.builtin import Signedness


class FrontendType:
    """Represents ay type in the frontend."""

    def to_xdsl():
        pass


# Type parameters for integers.
_Width = TypeVar("_Width", bound=int, covariant=True)
_Signedness = TypeVar("_Signedness", bound=Signedness, covariant=True)


class IntegerType(Generic[_Width, _Signedness], FrontendType):
    """Represents an integer type in the frontend."""

    def to_xdsl():
        return builtin.IntegerType.from_width
    
    def __add__(self, other: 'IntegerType[_Width, _Signedness]') -> 'IntegerType[_Width, _Signedness]':
        from xdsl.frontend.dialects.arith import addi
        return addi(self, other)
    
    def __sub__(self, other: 'IntegerType[_Width, _Signedness]') -> 'IntegerType[_Width, _Signedness]':
        from xdsl.frontend.dialects.arith import subi
        return subi(self, other)
    
    def __mul__(self, other: 'IntegerType[_Width, _Signedness]') -> 'IntegerType[_Width, _Signedness]':
        from xdsl.frontend.dialects.arith import muli
        return muli(self, other)
    
    def __and__(self, other: 'IntegerType[_Width, _Signedness]') -> 'IntegerType[_Width, _Signedness]':
        from xdsl.frontend.dialects.arith import andi
        return andi(self, other)

    def __rshift__(self, other: 'IntegerType[_Width, _Signedness]') -> 'IntegerType[_Width, _Signedness]':
        from xdsl.frontend.dialects.arith import shrsi
        return shrsi(self, other)

    def __eq__(self, other: 'IntegerType[_Width, _Signedness]') -> 'i1':
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "eq")
    
    def __ne__(self, other: 'IntegerType[_Width, _Signedness]') -> 'i1':
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "ne")
    
    def __le__(self, other: 'IntegerType[_Width, _Signedness]') -> 'i1':
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "sle")
    
    def __lt__(self, other: 'IntegerType[_Width, _Signedness]') -> 'i1':
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "slt")
    
    def __ge__(self, other: 'IntegerType[_Width, _Signedness]') -> 'i1':
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "sge")
    
    def __gt__(self, other: 'IntegerType[_Width, _Signedness]') -> 'i1':
        from xdsl.frontend.dialects.arith import cmpi
        return cmpi(self, other, "sgt")


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
        return builtin.IndexType


# Type alias for index type.
index: TypeAlias = IndexType


class Float16Type(FrontendType):
    """Represents a 16-bit floating-point type in the frontend."""

    def to_xdsl():
        return builtin.Float16Type

    def __add__(self, other: 'f16') -> 'f16':
        from xdsl.frontend.dialects.arith import addf
        return addf(self, other)
    
    def __sub__(self, other: 'f16') -> 'f16':
        from xdsl.frontend.dialects.arith import subf
        return subf(self, other)
    
    def __mul__(self, other: 'f16') -> 'f16':
        from xdsl.frontend.dialects.arith import mulf
        return mulf(self, other)


class Float32Type(FrontendType):
    """Represents a 32-bit floating-point type in the frontend."""

    def to_xdsl():
        return builtin.Float32Type
    
    def __add__(self, other: 'f32') -> 'f32':
        from xdsl.frontend.dialects.arith import addf
        return addf(self, other)
    
    def __sub__(self, other: 'f32') -> 'f32':
        from xdsl.frontend.dialects.arith import subf
        return subf(self, other)
    
    def __mul__(self, other: 'f32') -> 'f32':
        from xdsl.frontend.dialects.arith import mulf
        return mulf(self, other)


class Float64Type(FrontendType):
    """Represents a 64-bit floating-point type in the frontend."""

    def to_xdsl():
        return builtin.Float64Type
    
    def __add__(self, other: 'f64') -> 'f64':
        from xdsl.frontend.dialects.arith import addf
        return addf(self, other)
    
    def __sub__(self, other: 'f64') -> 'f64':
        from xdsl.frontend.dialects.arith import subf
        return subf(self, other)
    
    def __mul__(self, other: 'f64') -> 'f64':
        from xdsl.frontend.dialects.arith import mulf
        return mulf(self, other)


# Type alias for floating-point types.
f16: TypeAlias = Float16Type
f32: TypeAlias = Float32Type
f64: TypeAlias = Float64Type


# Type parameters for vectors.
_VectorShape = TypeVar("_VectorShape", bound=Tuple[int, ...], covariant=True)
_VectorElementType = TypeVar("_VectorElementType", bound=FrontendType, covariant=True)


class VectorType(Generic[_VectorElementType, _VectorShape], FrontendType):
    """Represents a vector type with in the frontend."""

    def to_xdsl():
        return builtin.VectorType.from_type_and_list


# Type parameters for ranked tensors.
_TensorShape = TypeVar("_TensorShape", bound=Tuple[int, ...], covariant=True)
_TensorElementType = TypeVar("_TensorElementType", bound=FrontendType, covariant=True)


class TensorType(Generic[_TensorElementType, _TensorShape], FrontendType):
    """Represents a tensor type with a known rank in the frontend."""

    def to_xdsl():
        return builtin.TensorType.from_type_and_list
    
    def __getitem__(self, *indices: index) -> _TensorElementType:
        from xdsl.frontend.dialects.tensor import extract
        return extract(self, indices)
    
    def __setitem__(self, *indices: index, v: _TensorElementType):
        from xdsl.frontend.dialects.tensor import insert
        return insert(v, self, indices)


# Type parameter for unranked tensors.
_UnrankedTensorElementType = TypeVar("_UnrankedTensorElementType", bound=FrontendType, covariant=True)


class UnrankedTensorType(Generic[_UnrankedTensorElementType], FrontendType):
    """Represents a tensor type with unknown rank in the frontend."""

    def to_xdsl():
        return builtin.UnrankedTensorType.from_type


class Module:
    """Represents a builtin.module."""
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
