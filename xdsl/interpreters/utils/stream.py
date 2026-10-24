import abc
from dataclasses import dataclass, field
from typing import Generic

from typing_extensions import TypeVar

from xdsl.dialects.builtin import I32, IntegerAttr, i32

T = TypeVar("T")
TCov = TypeVar("TCov", covariant=True)
TCon = TypeVar("TCon", contravariant=True)


class ReadableStream(abc.ABC, Generic[TCov]):
    """
    Abstract base class for readable stream interpreter model objects.
    """

    @abc.abstractmethod
    def read(self) -> TCov:
        raise NotImplementedError()


class WritableStream(abc.ABC, Generic[TCon]):
    """
    Abstract base class for readable stream interpreter model objects.
    """

    @abc.abstractmethod
    def write(self, value: TCon) -> None:
        raise NotImplementedError()


@dataclass
class Nats(ReadableStream[IntegerAttr[I32]]):
    """
    A stream designed for testing, outputs the next natural number each time it's read.
    """

    index = 0

    def read(self) -> IntegerAttr[I32]:
        self.index += 1
        return IntegerAttr(self.index, i32)


@dataclass
class Acc(WritableStream[IntegerAttr[I32]]):
    """
    A stream designed for testing, appends the next natural number written.
    """

    values: list[IntegerAttr[I32]] = field(default_factory=list[IntegerAttr[I32]])

    def write(self, value: IntegerAttr[I32]) -> None:
        return self.values.append(value)
