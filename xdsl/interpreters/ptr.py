from __future__ import annotations

import itertools
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar, final

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
        return TypedPtr(self, int32)

    @staticmethod
    def new_int32(els: Sequence[int]) -> RawPtr:
        return RawPtr.new("<i", [(el,) for el in els])

    @property
    def int64(self) -> TypedPtr[int]:
        return TypedPtr(self, int64)

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
        if index_bitwidth != 32 and index_bitwidth != 64:
            raise ValueError(
                f"Invalid index bitwidth {index_bitwidth} monly 32 or 64 allowed"
            )
        return TypedPtr(self, index(index_bitwidth))

    @staticmethod
    def new_index(els: Sequence[int], index_bitwidth: int) -> RawPtr:
        return RawPtr.new(
            RawPtr.index_format_for_bitwidth(index_bitwidth), [(el,) for el in els]
        )

    @property
    def float32(self) -> TypedPtr[float]:
        return TypedPtr(self, float32)

    @staticmethod
    def new_float32(els: Sequence[float]) -> RawPtr:
        return RawPtr.new("<f", [(el,) for el in els])

    @property
    def float64(self) -> TypedPtr[float]:
        return TypedPtr(self, float64)

    @staticmethod
    def new_float64(els: Sequence[float]) -> RawPtr:
        return RawPtr.new("<d", [(el,) for el in els])


@final
@dataclass
class XType(Generic[_T]):

    type: type[_T]
    format: str


int32 = XType(int, "<i")
int64 = XType(int, "<I")
float32 = XType(float, "<f")
float64 = XType(float, "<d")


def index(bitwidth: Literal[32] | Literal[64]) -> XType[int]:
    return int32 if bitwidth == 32 else int64


@dataclass
class TypedPtr(Generic[_T]):
    raw: RawPtr
    xtype: XType[_T]

    @property
    def format(self) -> str:
        return self.xtype.format

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
    def zeros(count: int, xtype: XType[_T]) -> TypedPtr[_T]:
        size = struct.calcsize(xtype.format)
        return TypedPtr(RawPtr.zeros(size * count), xtype)

    @staticmethod
    def new(els: Sequence[_T], xtype: XType[_T]) -> TypedPtr[_T]:
        return TypedPtr(RawPtr.new(xtype.format, tuple((el,) for el in els)), xtype)

    @staticmethod
    def new_float32(els: Sequence[float]) -> TypedPtr[float]:
        return TypedPtr.new(els, float32)

    @staticmethod
    def new_float64(els: Sequence[float]) -> TypedPtr[float]:
        return TypedPtr.new(els, float64)

    @staticmethod
    def new_int32(els: Sequence[int]) -> TypedPtr[int]:
        return TypedPtr.new(els, int32)

    @staticmethod
    def new_index(els: Sequence[int], index_bitwidth: int) -> TypedPtr[int]:
        if index_bitwidth != 32 and index_bitwidth != 64:
            raise ValueError(
                f"Invalid index bitwidth {index_bitwidth} monly 32 or 64 allowed"
            )
        return TypedPtr.new(els, index(index_bitwidth))
