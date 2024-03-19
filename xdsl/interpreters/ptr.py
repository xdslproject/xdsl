from __future__ import annotations

import itertools
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

_T = TypeVar("_T")


@dataclass
class RawPtr:
    """
    Data structure to help simulate pointers into memory.
    """

    memory: bytearray
    offset: int = field(default=0)

    @staticmethod
    def zeros(count: int) -> RawPtr:
        """
        Returns a new Ptr of size `count` with offset 0.
        """
        return RawPtr(bytearray(count))

    @staticmethod
    def new(el_format: str, els: Sequence[tuple[Any, ...]]) -> RawPtr:
        """
        Returns a new Ptr. The first parameter is a format string as specified in the
        `struct` module, and elements to set.
        """
        el_size = struct.calcsize(el_format)
        res = RawPtr.zeros(len(els) * el_size)
        for i, el in enumerate(els):
            struct.pack_into(el_format, res.memory, i * el_size, *el)
        return res

    def get_iter(self, format: str) -> Iterator[Any]:
        # The memoryview needs to be a multiple of the size of the packed format
        format_size = struct.calcsize(format)
        mem_view = memoryview(self.memory)[self.offset :]
        remainder = len(mem_view) % format_size
        if remainder:
            mem_view = mem_view[:-remainder]
        return (values[0] for values in struct.iter_unpack(format, mem_view))

    def get(self, format: str) -> Any:
        return next(self.get_iter(format))

    def set(self, format: str, *item: Any):
        struct.pack_into(format, self.memory, self.offset, *item)

    def get_list(self, format: str, count: int):
        return list(itertools.islice(self.get_iter(format), count))

    def __add__(self, offset: int) -> RawPtr:
        """
        Aliases the data, so storing into the offset stores for all other references
        to the list.
        """
        return RawPtr(self.memory, self.offset + offset)

    @property
    def int32(self) -> TypedPtr[int]:
        return TypedPtr(self, "<i")

    @staticmethod
    def new_int32(els: Sequence[int]) -> RawPtr:
        return RawPtr.new("<i", [(el,) for el in els])

    @property
    def float32(self) -> TypedPtr[float]:
        return TypedPtr(self, "<f")

    @staticmethod
    def new_float32(els: Sequence[float]) -> RawPtr:
        return RawPtr.new("<f", [(el,) for el in els])

    @property
    def float64(self) -> TypedPtr[float]:
        return TypedPtr(self, "<d")

    @staticmethod
    def new_float64(els: Sequence[float]) -> RawPtr:
        return RawPtr.new("<d", [(el,) for el in els])


@dataclass
class TypedPtr(Generic[_T]):
    raw: RawPtr
    format: str

    @property
    def size(self) -> int:
        return struct.calcsize(self.format)

    def get_list(self, count: int) -> list[_T]:
        return self.raw.get_list(self.format, count)

    def __getitem__(self, index: int) -> _T:
        return (self.raw + index * self.size).get(self.format)

    def __setitem__(self, index: int, value: _T):
        (self.raw + index * self.size).set(self.format, value)
