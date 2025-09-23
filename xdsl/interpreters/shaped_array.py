from __future__ import annotations

import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import accumulate, product
from math import prod
from typing import Generic

from typing_extensions import Self, TypeVar

from xdsl.dialects.builtin import PackableType, ShapedType
from xdsl.interpreters.utils.ptr import TypedPtr

_T = TypeVar("_T")


@dataclass
class ShapedArray(Generic[_T]):
    """
    A helper structure to represent instances of type MemRefType, TensorType, VectorType, etc.
    in the interpreter.
    """

    _data: TypedPtr[_T]
    shape: list[int]

    @property
    def element_type(self) -> PackableType[_T]:
        return self._data.xtype

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def data(self) -> list[_T]:
        return self._data.get_list(self.size)

    @property
    def data_ptr(self) -> TypedPtr[_T]:
        return self._data

    def copy(self) -> Self:
        return type(self)(self._data.copy(), self.shape.copy())

    def with_shape(self, new_shape: Sequence[int]) -> Self:
        return type(self)(self._data.copy(), list(new_shape))

    def offset(self, index: Sequence[int]) -> int:
        """
        Returns the index of the element in `self.data` for a given tuple of indices
        """
        if len(index) != len(self.shape):
            raise ValueError(f"Invalid indices {index} for shape {self.shape}")
        # For each dimension, the number of elements in the nested arrays
        strides = ShapedType.strides_for_shape(self.shape)
        offset = sum(i * stride for i, stride in zip(index, strides, strict=True))
        return offset

    def load(self, index: Sequence[int]) -> _T:
        """
        Returns the element for a given tuple of indices
        """
        return self._data[self.offset(index)]

    def store(self, index: Sequence[int], value: _T) -> None:
        """
        Returns the element for a given tuple of indices
        """
        self._data[self.offset(index)] = value

    def indices(self) -> Iterable[tuple[int, ...]]:
        """
        Iterates over the indices of this shaped array.
        """
        yield from product(*(range(dim) for dim in self.shape))

    def transposed(self, dim0: int, dim1: int) -> Self:
        """
        Returns a new ShapedArray, with the dimensions `dim0` and `dim1` transposed.
        """
        new_shape = list(self.shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        old_list = self.data
        new_data = type(self.data_ptr).new(old_list, xtype=self.data_ptr.xtype)

        result = type(self)(new_data, new_shape)

        for source_index in self.indices():
            dest_index = list(source_index)
            dest_index[dim0], dest_index[dim1] = source_index[dim1], source_index[dim0]
            result.store(tuple(dest_index), self.load(source_index))

        return result

    def __format__(self, format_spec: str) -> str:
        prod_dims: list[int] = list(accumulate(reversed(self.shape), operator.mul))
        size = prod_dims[-1]
        result = "[" * len(self.shape)

        for i in range(size):
            d = self._data[i]
            if i:
                n = sum(not i % p for p in prod_dims)
                result += "]" * n
                result += ", "
                result += "[" * n
            result += f"{d}"

        result += "]" * len(self.shape)
        return result
