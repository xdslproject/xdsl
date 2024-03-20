from xdsl.dialects import ml_program
from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.ml_program import MLProgramFunctions
from xdsl.interpreters.shaped_array import ShapedArray


def test_ml_program_global():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MLProgramFunctions())
    op = ml_program.Global(
        StringAttr("global_op"),
        TensorType(i32, [4]),
        None,
        None,
        StringAttr("private"),
    )
    print(op.value)
    c = interpreter.run_op(op, ())
    assert c == ShapedArray([0, 1, 2, 3], [4])


def test_ml_program_global_load_constant():
    pass
