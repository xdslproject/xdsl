import abc
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")
TCov = TypeVar("TCov", covariant=True)
TCon = TypeVar("TCon", contravariant=True)


class ReadableStream(Generic[TCov], abc.ABC):
    """
    Abstract base class for readable stream interpreter model objects.
    """

    @abc.abstractmethod
    def read(self) -> TCov:
        raise NotImplementedError()


class WritableStream(Generic[TCon], abc.ABC):
    """
    Abstract base class for readable stream interpreter model objects.
    """

    @abc.abstractmethod
    def write(self, value: TCon) -> None:
        raise NotImplementedError()


@dataclass
class Nats(ReadableStream[int]):
    """
    A stream designed for testing, outputs the next natural number each time it's read.
    """

    index = 0

    def read(self) -> int:
        self.index += 1
        return self.index


@dataclass
class Acc(WritableStream[int]):
    """
    A stream designed for testing, appends the next natural number written.
    """

    values: list[int] = field(default_factory=list)

    def write(self, value: int) -> None:
        return self.values.append(value)
