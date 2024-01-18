from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any, Generic, TypeVar

from typing_extensions import Protocol

from xdsl.dialects import stream
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray

T = TypeVar("T")
TCov = TypeVar("TCov", covariant=True)
TCon = TypeVar("TCon", contravariant=True)


class ReadableStream(Protocol[TCov]):
    def read(self) -> TCov:
        raise NotImplementedError()


class WritableStream(Protocol[TCon]):
    def write(self, value: TCon) -> None:
        raise NotImplementedError()


@dataclass
class AnyReadableStream(Generic[TCov], ReadableStream[TCov]):
    _read: Callable[[], TCov]

    def read(self) -> TCov:
        return self._read()


@dataclass
class AnyWritableStream(Generic[TCon], WritableStream[TCon]):
    _write: Callable[[TCon], None]

    def write(self, value: TCon) -> None:
        self._write(value)


@dataclass
class StridePattern:
    ub: list[int]
    strides: list[int]


def strided_pointer_offset_iter(strides: list[int], ub: list[int]) -> Iterator[int]:
    indices_iter = product(*(range(b) for b in ub))
    offsets = [
        sum((stride * index for stride, index in zip(strides, indices)))
        for indices in indices_iter
    ]
    return iter(offsets)


@dataclass
class StridedMemrefInputStream(Generic[TCov], ReadableStream[TCov]):
    index_iter: Iterator[int]
    array: ShapedArray[TCov]

    def read(self) -> TCov:
        index = next(self.index_iter)
        value = self.array.data[index]
        return value


@dataclass
class StridedMemrefOutputStream(Generic[TCon], WritableStream[TCon]):
    index_iter: Iterator[int]
    array: ShapedArray[TCon]

    def write(self, value: TCon) -> None:
        index = next(self.index_iter)
        self.array.data[index] = value


@register_impls
class StreamFunctions(InterpreterFunctions):
    @impl(stream.ReadOp)
    def run_read(
        self, interpreter: Interpreter, op: stream.ReadOp, args: tuple[Any, ...]
    ) -> PythonValues:
        (stream,) = args
        stream: ReadableStream[Any] = stream
        return (stream.read(),)

    @impl(stream.WriteOp)
    def run_write(
        self, interpreter: Interpreter, op: stream.WriteOp, args: tuple[Any, ...]
    ) -> PythonValues:
        (value, stream) = args
        stream: WritableStream[Any] = stream
        stream.write(value)
        return ()
