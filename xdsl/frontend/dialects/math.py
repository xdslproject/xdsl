import xdsl.dialects.math as math

from typing import TypeVar, Union
from xdsl.frontend.dialects.builtin import IndexType, i1, i32, i64
from xdsl.ir import Operation

_IntType = TypeVar("_IntType", bound=Union[IndexType, i1, i32, i64], covariant=True)


def ipowi(lhs: _IntType, rhs: _IntType) -> _IntType:
    pass


def resolve_ipowi() -> Operation:
    return math.IPowI.get
