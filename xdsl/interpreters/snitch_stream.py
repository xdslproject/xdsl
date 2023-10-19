from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any

from xdsl.dialects import snitch_stream
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.riscv import RawPtr, RiscvFunctions
from xdsl.interpreters.stream import (
    ReadableStream,
    StridePattern,
    WritableStream,
    strided_pointer_offset_iter,
)


@dataclass
class StridedPointerInputStream(ReadableStream[float]):
    offset_iter: Iterator[int]
    pointer: RawPtr

    def read(self) -> float:
        offset = next(self.offset_iter)
        return (self.pointer + offset).float64[0]


@dataclass
class StridedPointerOutputStream(WritableStream[float]):
    offset_iter: Iterator[int]
    pointer: RawPtr

    def write(self, value: float) -> None:
        offset = next(self.offset_iter)
        (self.pointer + offset).float64[0] = value


@register_impls
class SnitchStreamFunctions(InterpreterFunctions):
    @impl(snitch_stream.GenericOp)
    def run_generic(
        self,
        interpreter: Interpreter,
        op: snitch_stream.GenericOp,
        args: tuple[Any, ...],
    ) -> PythonValues:
        input_streams: tuple[ReadableStream[Any], ...] = interpreter.get_values(
            op.inputs
        )
        output_streams: tuple[WritableStream[Any], ...] = interpreter.get_values(
            op.outputs
        )

        loop_ranges = op.static_loop_ranges

        for _ in product(*(range(loop_range.data) for loop_range in loop_ranges)):
            loop_args = tuple(i.read() for i in input_streams)
            loop_args = RiscvFunctions.set_reg_values(
                interpreter, op.body.block.args, loop_args
            )
            loop_results = interpreter.run_ssacfg_region(op.body, loop_args, "for_loop")
            for o, r in zip(output_streams, loop_results):
                o.write(r)

        return ()

    @impl(snitch_stream.StridePatternOp)
    def run_stride_pattern(
        self,
        interpreter: Interpreter,
        op: snitch_stream.StridePatternOp,
        args: PythonValues,
    ) -> PythonValues:
        return (StridePattern([b.data for b in op.ub], [s.data for s in op.strides]),)

    @impl(snitch_stream.StridedReadOp)
    def run_strided_read(
        self,
        interpreter: Interpreter,
        op: snitch_stream.StridedReadOp,
        args: tuple[Any, ...],
    ) -> PythonValues:
        (memref, pattern) = args
        memref: RawPtr = memref
        pattern: StridePattern = pattern

        input_stream_factory = StridedPointerInputStream(
            strided_pointer_offset_iter(pattern.strides, pattern.ub),
            memref,
        )
        return (input_stream_factory,)

    @impl(snitch_stream.StridedWriteOp)
    def run_strided_write(
        self,
        interpreter: Interpreter,
        op: snitch_stream.StridedWriteOp,
        args: tuple[Any, ...],
    ) -> PythonValues:
        (memref, pattern) = args
        memref: RawPtr = memref
        pattern: StridePattern = pattern

        output_stream_factory = StridedPointerOutputStream(
            strided_pointer_offset_iter(pattern.strides, pattern.ub),
            memref,
        )
        return (output_stream_factory,)

    @impl(snitch_stream.ReadOp)
    def run_read(
        self, interpreter: Interpreter, op: snitch_stream.ReadOp, args: tuple[Any, ...]
    ) -> PythonValues:
        (stream,) = args
        stream: ReadableStream[Any] = stream
        return (stream.read(),)

    @impl(snitch_stream.WriteOp)
    def run_write(
        self, interpreter: Interpreter, op: snitch_stream.WriteOp, args: tuple[Any, ...]
    ) -> PythonValues:
        (stream, value) = args
        stream: WritableStream[Any] = stream
        stream.write(value)
        return ()

    @impl_terminator(snitch_stream.YieldOp)
    def run_br(
        self, interpreter: Interpreter, op: snitch_stream.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
