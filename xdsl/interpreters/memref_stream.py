from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any, Generic, TypeVar

from xdsl.dialects import memref_stream
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
from xdsl.interpreters.stream import ReadableStream, WritableStream

T = TypeVar("T")
TCov = TypeVar("TCov", covariant=True)
TCon = TypeVar("TCon", contravariant=True)


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
class MemrefStreamFunctions(InterpreterFunctions):
    @impl(memref_stream.GenericOp)
    def run_generic(
        self,
        interpreter: Interpreter,
        op: memref_stream.GenericOp,
        args: tuple[Any, ...],
    ) -> PythonValues:
        input_count = len(op.inputs)
        output_count = len(op.outputs)
        total_count = input_count + output_count

        repeat_count = args[0]
        input_shaped_arrays: tuple[ShapedArray[Any], ...] = args[1 : input_count + 1]
        output_shaped_arrays: tuple[ShapedArray[Any], ...] = args[input_count + 1 :]

        stride_patterns: tuple[StridePattern, ...] = args[total_count + 1 :]

        if len(stride_patterns) == 1:
            stride_patterns = stride_patterns * total_count

        input_stride_patterns = stride_patterns[:input_count]
        output_stride_patterns = stride_patterns[input_count:]

        input_streams = tuple(
            StridedMemrefInputStream(
                strided_pointer_offset_iter(pat.strides, pat.ub), shaped_array
            )
            for pat, shaped_array in zip(input_stride_patterns, input_shaped_arrays)
        )

        output_streams = tuple(
            StridedMemrefOutputStream(
                strided_pointer_offset_iter(pat.strides, pat.ub), shaped_array
            )
            for pat, shaped_array in zip(output_stride_patterns, output_shaped_arrays)
        )

        for _ in range(repeat_count):
            loop_args = tuple(i.read() for i in input_streams)
            loop_results = interpreter.run_ssacfg_region(op.body, loop_args, "for_loop")
            for o, r in zip(output_streams, loop_results):
                o.write(r)

        return ()

    @impl(memref_stream.StridePatternOp)
    def run_stride_pattern(
        self,
        interpreter: Interpreter,
        op: memref_stream.StridePatternOp,
        args: PythonValues,
    ) -> PythonValues:
        return (StridePattern([b.data for b in op.ub], [s.data for s in op.strides]),)

    @impl_terminator(memref_stream.YieldOp)
    def run_br(
        self, interpreter: Interpreter, op: memref_stream.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
