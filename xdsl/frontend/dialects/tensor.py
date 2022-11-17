import xdsl.dialects.tensor as tensor

from typing import List, Tuple, TypeVar
from xdsl.frontend.dialects.builtin import FrontendType, TensorType, index
from xdsl.ir import Operation


# Type parameters for ranked tensors.
_TensorShape = TypeVar("_TensorShape", bound=Tuple[int, ...], covariant=True)
_TensorElementType = TypeVar("_TensorElementType", bound=FrontendType, covariant=True)


def extract(tensor: TensorType[_TensorElementType, _TensorShape], *indices: index) -> _TensorElementType:
    pass


def resolve_extract() -> Operation:
    return tensor.Extract.get


def insert(value: _TensorElementType, tensor: TensorType[_TensorElementType, _TensorShape], *indices: index):
    pass


def resolve_insert() -> Operation:
    return tensor.Insert.get
