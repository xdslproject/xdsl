from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Generic, Literal

from typing_extensions import Self, TypeVar

from xdsl.dialects.builtin import (
    Float32Type,
    Float64Type,
    PackableType,
    i32,
    i64,
)

_T = TypeVar("_T")


@dataclass
class RawPtr:
    """
    Data structure to help simulate pointers into memory.
    """

    memory: bytearray
    offset: int = field(default=0)

    @property
    def memoryview(self) -> memoryview:
        return memoryview(self.memory)[self.offset :]

    def copy(self) -> RawPtr:
        return RawPtr(bytearray(self.memory), self.offset)

    @staticmethod
    def zeros(count: int) -> RawPtr:
        """
        Returns a new Ptr of size `count` with offset 0.
        """
        return RawPtr(bytearray(count))

    def __add__(self, offset: int) -> RawPtr:
        """
        Aliases the data, so storing into the offset stores for all other references
        to the list.
        """
        return RawPtr(self.memory, self.offset + offset)

    @property
    def int32(self) -> TypedPtr[int]:
        return TypedPtr(self, xtype=int32)

    @property
    def int64(self) -> TypedPtr[int]:
        return TypedPtr(self, xtype=int64)

    def index(self, index_bitwidth: int) -> TypedPtr[int]:
        if index_bitwidth != 32 and index_bitwidth != 64:
            raise ValueError(
                f"Invalid index bitwidth {index_bitwidth} monly 32 or 64 allowed"
            )
        return TypedPtr(self, xtype=index(index_bitwidth))

    @property
    def float32(self) -> TypedPtr[float]:
        return TypedPtr(self, xtype=float32)

    @property
    def float64(self) -> TypedPtr[float]:
        return TypedPtr(self, xtype=Float64Type())


int32 = i32
int64 = i64
float32 = Float32Type()
float64 = Float64Type()


def index(bitwidth: Literal[32, 64]) -> PackableType[int]:
    return int32 if bitwidth == 32 else int64


@dataclass
class TypedPtr(Generic[_T]):
    """
    A typed pointer into memory, similar to numpy's ndarray, but without the shape.
    """

    raw: RawPtr
    _: KW_ONLY
    xtype: PackableType[_T]

    @property
    def size(self) -> int:
        return self.xtype.compile_time_size

    def copy(self) -> Self:
        return type(self)(self.raw.copy(), xtype=self.xtype)

    def get_iter(self) -> Iterator[_T]:
        # The memoryview needs to be a multiple of the size of the packed format
        format_size = self.size
        mem_view = self.raw.memoryview
        remainder = len(mem_view) % format_size
        if remainder:
            mem_view = mem_view[:-remainder]
        return self.xtype.iter_unpack(mem_view)

    def get_list(self, count: int) -> list[_T]:
        return list(itertools.islice(self.get_iter(), count))

    def __getitem__(self, index: int) -> _T:
        raw_at_offset = self.raw + index * self.size
        return self.xtype.unpack(raw_at_offset.memoryview[: self.size], 1)[0]

    def __setitem__(self, index: int, value: _T):
        raw_at_offset = self.raw + index * self.size
        self.xtype.pack_into(raw_at_offset.memory, raw_at_offset.offset, value)

    @staticmethod
    def zeros(count: int, *, xtype: PackableType[_T]) -> TypedPtr[_T]:
        size = xtype.compile_time_size
        return TypedPtr(RawPtr.zeros(size * count), xtype=xtype)

    @staticmethod
    def new(els: Sequence[_T], *, xtype: PackableType[_T]) -> TypedPtr[_T]:
        """
        Returns a new TypedPtr with the specified els packed into memory.
        """
        el_size = xtype.compile_time_size
        res = RawPtr.zeros(len(els) * el_size)
        for i, el in enumerate(els):
            xtype.pack_into(res.memory, i * el_size, el)
        return TypedPtr(res, xtype=xtype)

    @staticmethod
    def new_float32(els: Sequence[float]) -> TypedPtr[float]:
        return TypedPtr[float].new(els, xtype=Float32Type())

    @staticmethod
    def new_float64(els: Sequence[float]) -> TypedPtr[float]:
        return TypedPtr[float].new(els, xtype=Float64Type())

    @staticmethod
    def new_int32(els: Sequence[int]) -> TypedPtr[int]:
        return TypedPtr[int].new(els, xtype=i32)

    @staticmethod
    def new_int64(els: Sequence[int]) -> TypedPtr[int]:
        return TypedPtr[int].new(els, xtype=i64)

    @staticmethod
    def new_index(els: Sequence[int], index_bitwidth: int) -> TypedPtr[int]:
        if index_bitwidth != 32 and index_bitwidth != 64:
            raise ValueError(
                f"Invalid index bitwidth {index_bitwidth} monly 32 or 64 allowed"
            )
        return TypedPtr[int].new(els, xtype=index(index_bitwidth))
