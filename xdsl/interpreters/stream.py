from typing import Any, TypeVar

from typing_extensions import Protocol

from xdsl.dialects import stream
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)

T = TypeVar("T")
TCov = TypeVar("TCov", covariant=True)
TCon = TypeVar("TCon", contravariant=True)


class ReadableStream(Protocol[TCov]):
    def read(self) -> TCov:
        raise NotImplementedError()


class WritableStream(Protocol[TCon]):
    def write(self, value: TCon) -> None:
        raise NotImplementedError()


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
