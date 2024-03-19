from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    ModuleOp,
    TensorType,
    f32,
    i32,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.builtin import BuiltinFunctions
from xdsl.interpreters.shaped_array import ShapedArray

interpreter = Interpreter(ModuleOp([]))
interpreter.register_implementations(BuiltinFunctions())


def test_values():
    assert interpreter.value_for_attribute(IntegerAttr(1, i32)) == 1
    assert interpreter.value_for_attribute(IntegerAttr(2, i64)) == 2

    assert interpreter.value_for_attribute(FloatAttr(3.0, f32)) == 3.0

    assert interpreter.value_for_attribute(
        DenseIntOrFPElementsAttr.create_dense_int(
            TensorType(i32, [2, 3]), list(range(6))
        )
    ) == ShapedArray(list(range(6)), [2, 3])
