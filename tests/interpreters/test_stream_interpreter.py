from dataclasses import dataclass, field

from xdsl.dialects import stream
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.stream import (
    ReadableStream,
    StreamFunctions,
    WritableStream,
)
from xdsl.utils.test_value import TestSSAValue


@dataclass
class Nats(ReadableStream[int]):
    index = 0

    def read(self) -> int:
        self.index += 1
        return self.index


@dataclass
class Acc(WritableStream[int]):
    values: list[int] = field(default_factory=list)

    def write(self, value: int) -> None:
        return self.values.append(value)


def test_read_write():
    interpreter = Interpreter(ModuleOp([]), index_bitwidth=32)
    interpreter.register_implementations(StreamFunctions())

    input_stream = Nats()
    output_stream = Acc()

    index = IndexType()

    (value,) = interpreter.run_op(
        stream.ReadOp(TestSSAValue(stream.ReadableStreamType(index))), (input_stream,)
    )
    assert value == 1

    (value,) = interpreter.run_op(
        stream.ReadOp(TestSSAValue(stream.ReadableStreamType(index))), (input_stream,)
    )
    assert value == 2

    interpreter.run_op(
        stream.WriteOp(
            TestSSAValue(index), TestSSAValue(stream.ReadableStreamType(index))
        ),
        (
            1,
            output_stream,
        ),
    )
    assert output_stream.values == [1]

    interpreter.run_op(
        stream.WriteOp(
            TestSSAValue(index), TestSSAValue(stream.ReadableStreamType(index))
        ),
        (
            2,
            output_stream,
        ),
    )
    assert output_stream.values == [1, 2]
