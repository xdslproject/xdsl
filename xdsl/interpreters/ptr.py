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
    deallocated: bool = field(default=False)

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
        if self.deallocated:
            raise ValueError("Cannot get item of deallocated ptr")
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
        if self.deallocated:
            raise ValueError("Cannot set item of deallocated ptr")
        struct.pack_into(format, self.memory, self.offset, *item)

    def get_list(self, format: str, count: int):
        return list(itertools.islice(self.get_iter(format), count))

    def deallocate(self) -> None:
        self.deallocated = True

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
    def int64(self) -> TypedPtr[int]:
        return TypedPtr(self, "<I")

    @staticmethod
    def new_int64(els: Sequence[int]) -> RawPtr:
        return RawPtr.new("<I", [(el,) for el in els])

    @staticmethod
    def index_format_for_bitwidth(index_bitwidth: int) -> str:
        match index_bitwidth:
            case 32:
                return "<i"
            case 64:
                return "<I"
            case _:
                raise ValueError(f"Unsupported index bitwidth {index_bitwidth}")

    def index(self, index_bitwidth: int) -> TypedPtr[int]:
        return TypedPtr(self, RawPtr.index_format_for_bitwidth(index_bitwidth))

    @staticmethod
    def new_index(els: Sequence[int], index_bitwidth: int) -> RawPtr:
        return RawPtr.new(
            RawPtr.index_format_for_bitwidth(index_bitwidth), [(el,) for el in els]
        )

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

    @staticmethod
    def zeros(count: int, format: str) -> TypedPtr[Any]:
        size = struct.calcsize(format)
        return TypedPtr(RawPtr.zeros(size * count), format)

    @staticmethod
    def new(els: Sequence[Any], format: str) -> TypedPtr[Any]:
        return TypedPtr(RawPtr.new(format, tuple((el,) for el in els)), format)

    @staticmethod
    def new_float32(els: Sequence[float]) -> TypedPtr[float]:
        return TypedPtr(RawPtr.new_float32(els), "<f")

    @staticmethod
    def new_float64(els: Sequence[float]) -> TypedPtr[float]:
        return TypedPtr(RawPtr.new_float64(els), "<d")

    @staticmethod
    def new_int32(els: Sequence[int]) -> TypedPtr[int]:
        return TypedPtr(RawPtr.new_int32(els), "<i")

    @staticmethod
    def new_index(els: Sequence[int], index_bitwidth: int) -> TypedPtr[int]:
        return TypedPtr.new(els, RawPtr.index_format_for_bitwidth(index_bitwidth))
