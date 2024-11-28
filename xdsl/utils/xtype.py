import struct
from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar, final

_TCov = TypeVar("_TCov", covariant=True)


@final
@dataclass(frozen=True)
class XType(Generic[_TCov]):
    """
    A typed format representation, similar to numpy's dtype.
    """

    type: type[_TCov]
    format: str
    """
    Format string as specified in the `struct` module.
    https://docs.python.org/3/library/struct.html
    """

    @property
    def size(self) -> int:
        return struct.calcsize(self.format)

    def sized_format(self, size: int) -> str:
        """
        Return the format string for a specific number of elements.
        """
        return self.format[0] + str(size) + self.format[1]


int8 = XType(int, "<b")
int16 = XType(int, "<h")
int32 = XType(int, "<i")
int64 = XType(int, "<q")
float32 = XType(float, "<f")
float64 = XType(float, "<d")
