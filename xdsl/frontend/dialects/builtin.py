from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar, Literal, List
from xdsl.ir import Attribute
import xdsl.dialects.builtin as orig_builtin


V = TypeVar("V", bound=int, covariant=True)

# XXX: we use Attribute instead of ParametrizedAttribute
#   because we don't need the parameters field. In fact,
#   it's incompatible with @dataclass (frozen vs. non-frozen)


class IntegerType(Generic[V], Attribute):
    """
    name: "integer_type"

    Parameters:
        - width: ParameterDef[IntAttr]
    """

    def __add__(self, other: 'IntegerType[V]') -> 'IntegerType[V]':
        from xdsl.frontend.dialects.arith import Addi
        return Addi(self, other)


class IndexType(Attribute):
    """name: index"""
    pass


i1 = IntegerType[Literal[1]]
i32 = IntegerType[Literal[32]]
i64 = IntegerType[Literal[64]]

AnyIntegerType = IntegerType[Any]

_IntegerAttrTyp = TypeVar("_IntegerAttrTyp",
                          bound=IntegerType | IndexType, covariant=True)


@dataclass
class IntegerAttr(Generic[_IntegerAttrTyp], Attribute):
    """
    name: "integer"

    Parameters:
        - value: ParameterDef[IntAttr]
        - typ: ParameterDef[_IntegerAttrTyp]
    """
    value: int
    _default_typ = orig_builtin.i32


class Float32Type(Attribute):
    """name: f32"""

    def __sub__(self, other: 'Float32Type') -> 'Float32Type':
        from xdsl.frontend.dialects.arith import Subf
        return Subf(self, other)

    def __add__(self, other: 'Float32Type') -> 'Float32Type':
        from xdsl.frontend.dialects.arith import Addf
        return Addf(self, other)

    def __mul__(self, other: 'Float32Type') -> 'Float32Type':
        from xdsl.frontend.dialects.arith import Mulf
        return Mulf(self, other)


class Float64Type(Attribute):
    """name: f64"""

    def __sub__(self, other: 'Float64Type') -> 'Float64Type':
        from xdsl.frontend.dialects.arith import Subf
        return Subf(self, other)

    def __add__(self, other: 'Float64Type') -> 'Float64Type':
        from xdsl.frontend.dialects.arith import Addf
        return Addf(self, other)

    def __mul__(self, other: 'Float64Type') -> 'Float64Type':
        from xdsl.frontend.dialects.arith import Mulf
        return Mulf(self, other)


f32 = Float32Type
f64 = Float64Type

_FloatAttrTyp = TypeVar("_FloatAttrTyp", bound=f32 | f64, covariant=True)

@dataclass
class FloatAttr(Generic[_FloatAttrTyp], Attribute):
    """
    name: "float"

    Parameters:
        - value: ParameterDef[FloatData]
        - type: ParameterDef[Float32Type | Float64Type]
    """
    value: float
    _default_typ = orig_builtin.f64

@dataclass
class FloatData(Attribute):
    """name: float_data"""
    value: float

@dataclass
class IntAttr(Attribute):
    """name: int"""
    value: int

_Shape = TypeVar("_Shape", bound=Sequence[Any], covariant=True)
_TensorTypeElems = TypeVar("_TensorTypeElems", bound=Attribute, covariant=True)


class TensorType(Generic[_Shape, _TensorTypeElems], Attribute):
    """name: integer_type"""
    pass

_ArrayElemTyp = TypeVar("_ArrayElemTyp", bound=Attribute)

@dataclass
class ArrayAttr(Generic[_ArrayElemTyp], Attribute):
    """name: array"""
    value: List[_ArrayElemTyp]

class Module:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
