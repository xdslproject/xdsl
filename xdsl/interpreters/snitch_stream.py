from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from xdsl.dialects import snitch_stream
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)
from xdsl.interpreters.utils import ptr
from xdsl.interpreters.utils.stream import ReadableStream, WritableStream


@dataclass
class StridedPointerInputStream(ReadableStream[float]):
    offset_iter: Iterator[int]
    pointer: ptr.RawPtr
    index = -1

    def read(self) -> float:
        self.index += 1
        offset = next(self.offset_iter)
        return ptr.TypedPtr((self.pointer + offset), xtype=ptr.float64)[0]


@dataclass
class StridedPointerOutputStream(WritableStream[float]):
    index = -1
    offset_iter: Iterator[int]
    pointer: ptr.RawPtr

    def write(self, value: float) -> None:
        self.index += 1
        offset = next(self.offset_iter)
        ptr.TypedPtr((self.pointer + offset), xtype=ptr.float64)[0] = value


@register_impls
class SnitchStreamFunctions(InterpreterFunctions):
    @impl(snitch_stream.StreamingRegionOp)
    def run_streaming_region(
        self,
        interpreter: Interpreter,
        op: snitch_stream.StreamingRegionOp,
        args: tuple[Any, ...],
    ) -> PythonValues:
        input_stream_count = len(op.inputs)
        output_stream_count = len(op.outputs)
        input_pointers: tuple[ptr.RawPtr, ...] = args[:input_stream_count]
        output_pointers: tuple[ptr.RawPtr, ...] = args[
            input_stream_count : input_stream_count + output_stream_count
        ]

        if len(op.stride_patterns) == 1:
            pattern = op.stride_patterns.data[0]
            input_stride_patterns = (pattern,) * input_stream_count
            output_stride_patterns = (pattern,) * output_stream_count
        else:
            input_stride_patterns = op.stride_patterns.data[:input_stream_count]
            output_stride_patterns = op.stride_patterns.data[input_stream_count:]

        input_streams = tuple(
            StridedPointerInputStream(pat.offset_iter(), ptr)
            for pat, ptr in zip(input_stride_patterns, input_pointers, strict=True)
        )

        output_streams = tuple(
            StridedPointerOutputStream(pat.offset_iter(), ptr)
            for pat, ptr in zip(output_stride_patterns, output_pointers, strict=True)
        )

        interpreter.run_ssacfg_region(
            op.body, (*input_streams, *output_streams), "steraming_region"
        )

        return ()
