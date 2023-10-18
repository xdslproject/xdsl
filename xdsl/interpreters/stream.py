from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any, Generic, TypeVar

from typing_extensions import Protocol

from xdsl.dialects import stream
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray

T = TypeVar("T")
TCov = TypeVar("TCov", covariant=True)
TCon = TypeVar("TCon", contravariant=True)


class InputStream(Protocol[TCov]):
    def read(self) -> TCov:
        raise NotImplementedError()


class OutputStream(Protocol[TCon]):
    def write(self, value: TCon) -> None:
        raise NotImplementedError()


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
class StridedMemrefInputStream(Generic[TCov], InputStream[TCov]):
    index_iter: Iterator[int]
    array: ShapedArray[TCov]

    def read(self) -> TCov:
        index = next(self.index_iter)
        value = self.array.data[index]
        return value


@dataclass
class StridedMemrefOutputStream(Generic[TCon], OutputStream[TCon]):
    index_iter: Iterator[int]
    array: ShapedArray[TCon]

    def write(self, value: TCon) -> None:
        index = next(self.index_iter)
        self.array.data[index] = value


@register_impls
class StreamFunctions(InterpreterFunctions):
    @impl(stream.GenericOp)
    def run_generic(
        self, interpreter: Interpreter, op: stream.GenericOp, args: tuple[Any, ...]
    ) -> PythonValues:
        input_streams: tuple[InputStream[Any], ...] = interpreter.get_values(op.inputs)
        output_streams: tuple[OutputStream[Any], ...] = interpreter.get_values(
            op.outputs
        )

        loop_ranges = op.static_loop_ranges

        for _ in product(*(range(loop_range.data) for loop_range in loop_ranges)):
            loop_args = tuple(i.read() for i in input_streams)
            loop_results = interpreter.run_ssacfg_region(op.body, loop_args, "for_loop")
            for o, r in zip(output_streams, loop_results):
                o.write(r)

        return ()

    @impl(stream.StridePatternOp)
    def run_stride_pattern(
        self, interpreter: Interpreter, op: stream.StridePatternOp, args: PythonValues
    ) -> PythonValues:
        return (StridePattern([b.data for b in op.ub], [s.data for s in op.strides]),)

    @impl(stream.StridedReadOp)
    def run_strided_read(
        self, interpreter: Interpreter, op: stream.StridedReadOp, args: tuple[Any, ...]
    ) -> PythonValues:
        (memref, pattern) = args
        memref: ShapedArray[Any] = memref
        pattern: StridePattern = pattern

        input_stream_factory = StridedMemrefInputStream(
            strided_pointer_offset_iter(pattern.strides, pattern.ub),
            memref,
        )
        return (input_stream_factory,)

    @impl(stream.StridedWriteOp)
    def run_strided_write(
        self, interpreter: Interpreter, op: stream.StridedWriteOp, args: tuple[Any, ...]
    ) -> PythonValues:
        (memref, pattern) = args
        memref: ShapedArray[Any] = memref
        pattern: StridePattern = pattern

        output_stream_factory = StridedMemrefOutputStream(
            strided_pointer_offset_iter(pattern.strides, pattern.ub),
            memref,
        )
        return (output_stream_factory,)

    @impl(stream.ReadOp)
    def run_read(
        self, interpreter: Interpreter, op: stream.ReadOp, args: tuple[Any, ...]
    ) -> PythonValues:
        (stream,) = args
        stream: InputStream[Any] = stream
        return (stream.read(),)

    @impl(stream.WriteOp)
    def run_write(
        self, interpreter: Interpreter, op: stream.WriteOp, args: tuple[Any, ...]
    ) -> PythonValues:
        (stream, value) = args
        stream: OutputStream[Any] = stream
        stream.write(value)
        return ()

    @impl_terminator(stream.YieldOp)
    def run_br(
        self, interpreter: Interpreter, op: stream.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
