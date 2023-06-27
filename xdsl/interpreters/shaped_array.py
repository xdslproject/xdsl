"""
A helper structure to represent instances of type MemrefType, TensorType, VectorType, etc.
in the interpreter.
"""

from itertools import accumulate
from math import prod
import operator
from typing import Generic, TypeVar
from attr import dataclass

_T = TypeVar("_T")


@dataclass
class ShapedArray(Generic[_T]):
    data: list[_T]
    shape: list[int]

    def __post__init__(self):
        assert prod(self.shape) == len(self.data)

    def offset(self, indices: tuple[int, ...]) -> int:
        """
        Returns the index of the element in `self.data` for a given tuple of indices
        """
        if len(indices) != len(self.shape):
            raise ValueError(f"Invalid indices {indices} for shape {self.shape}")
        # For each dimension, the number of elements in the nested arrays
        prod_dims: list[int] = list(
            accumulate(reversed(self.shape), operator.mul, initial=1)
        )[:-1]
        offsets = map(operator.mul, reversed(indices), prod_dims)
        offset = sum(offsets)
        return offset

    def load(self, indices: tuple[int, ...]) -> _T:
        return self.data[self.offset(indices)]

    def store(self, indices: tuple[int, ...], value: _T) -> None:
        self.data[self.offset(indices)] = value

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
