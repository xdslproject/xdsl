from typing import Literal

import pytest

from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    TensorType,
    f32,
    i1,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.builtin import BuiltinFunctions
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils import ptr

interpreter = Interpreter(ModuleOp([]))
interpreter.register_implementations(BuiltinFunctions())


def test_values():
    assert interpreter.value_for_attribute(IntegerAttr(1, i32), i32) == 1
    assert interpreter.value_for_attribute(IntegerAttr(2, i64), i64) == 2
    assert interpreter.value_for_attribute(IntegerAttr(0, i1), i1) == 0

    assert interpreter.value_for_attribute(FloatAttr(3.0, f32), f32) == 3.0

    assert interpreter.value_for_attribute(
        DenseIntOrFPElementsAttr.from_list(TensorType(i32, [2, 3]), list(range(6))),
        MemRefType(i32, (2, 3)),
    ) == ShapedArray(ptr.TypedPtr.new_int32(list(range(6))), [2, 3])


@pytest.mark.parametrize(
    "index_bitwidth,value,expected",
    [
        (32, 1 << 31, -(1 << 31)),
        (32, (1 << 32) + 7, 7),
        (64, 1 << 31, 1 << 31),
        (64, 1 << 63, -(1 << 63)),
        (64, (1 << 64) + 7, 7),
        (32, -1, -1),
        (64, -1, -1),
    ],
)
def test_index_values(index_bitwidth: Literal[32, 64], value: int, expected: int):
    interpreter = Interpreter(ModuleOp([]), index_bitwidth=index_bitwidth)
    interpreter.register_implementations(BuiltinFunctions())
    interpreter.register_implementations(ArithFunctions())
    attr = IntegerAttr(value, IndexType())
    constant = ConstantOp(attr)
    constant.verify()

    assert interpreter.value_for_attribute(attr, attr.type) == expected
    assert interpreter.run_op(constant, ()) == (expected,)
