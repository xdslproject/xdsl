from xdsl.dialects import tensor
from xdsl.dialects.builtin import ModuleOp, TensorType, f32, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.tensor import TensorFunctions
from xdsl.utils.test_value import TestSSAValue


def test_tensor_empty():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(TensorFunctions())
    op = tensor.EmptyOp((), TensorType(f32, [3, 2]))
    (c,) = interpreter.run_op(op, ())
    assert c == ShapedArray([0.0] * 6, [3, 2])


def test_tensor_reshape():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(TensorFunctions())
    op = tensor.ReshapeOp(
        TestSSAValue(TensorType(f32, [4, 1])),
        TestSSAValue(TensorType(i32, [1])),
        TensorType(f32, [4]),
    )
    a = ShapedArray([1, 2, 3, 4], [4, 1])
    b = ShapedArray([4], [1])
    (c,) = interpreter.run_op(op, (a, b))
    assert c == ShapedArray([1, 2, 3, 4], [4])
