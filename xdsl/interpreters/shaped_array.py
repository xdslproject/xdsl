from __future__ import annotations

import operator
from dataclasses import dataclass
from itertools import accumulate, product
from math import prod
from typing import Generic, Iterable, TypeAlias, TypeVar

from typing_extensions import Self

_T = TypeVar("_T")

Index: TypeAlias = tuple[int, ...]


@dataclass(init=False)
class ShapedArray(Generic[_T]):
    """
    A helper structure to represent instances of type MemrefType, TensorType, VectorType, etc.
    in the interpreter.
    """

    data: list[_T]
    shape: list[int]

    def __init__(self, data: list[_T] | _T, shape: list[int]):
        if not isinstance(data, list):
            data = [data] * prod(shape)

        self.data = data
        self.shape = shape

    def __post__init__(self):
        assert prod(self.shape) == len(self.data)

    def offset(self, index: Index) -> int:
        """
        Returns the index of the element in `self.data` for a given tuple of indices
        """
        if len(index) != len(self.shape):
            raise ValueError(f"Invalid indices {index} for shape {self.shape}")
        # For each dimension, the number of elements in the nested arrays
        prod_dims: list[int] = list(
            accumulate(reversed(self.shape), operator.mul, initial=1)
        )[:-1]
        offsets = map(operator.mul, reversed(index), prod_dims)
        offset = sum(offsets)
        return offset

    def load(self, index: tuple[int, ...]) -> _T:
        """
        Returns the element for a given tuple of indices
        """
        return self.data[self.offset(index)]

    def store(self, index: tuple[int, ...], value: _T) -> None:
        """
        Returns the element for a given tuple of indices
        """
        self.data[self.offset(index)] = value

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

        result = ShapedArray(list(self.data), new_shape)

        for source_index in self.indices():
            dest_index = list(source_index)
            dest_index[dim0], dest_index[dim1] = source_index[dim1], source_index[dim0]
            result.store(tuple(dest_index), self.load(source_index))

        return result

    def __format__(self, __format_spec: str) -> str:
        prod_dims: list[int] = list(accumulate(reversed(self.shape), operator.mul))
        assert prod_dims[-1] == len(self.data)
        result = "[" * len(self.shape)

        for i, d in enumerate(self.data):
            if i:
                n = sum(not i % p for p in prod_dims)
                result += "]" * n
                result += ", "
                result += "[" * n
            result += f"{d}"

        result += "]" * len(self.shape)
        return result
