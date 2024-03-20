from xdsl.dialects import ml_program
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    TensorType,
    i32,
    i64,
)
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
        DenseIntOrFPElementsAttr.tensor_from_list([4], i32, [4]),
        StringAttr("private"),
    )

    c = interpreter.run_op(op, ())
    assert c[0] == ShapedArray([4], [4])


def test_ml_program_global_ex2():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MLProgramFunctions())
    op = ml_program.Global(
        StringAttr("global_op_2"),
        TensorType(i64, [2]),
        None,
        DenseIntOrFPElementsAttr.tensor_from_list([1, 320], i64, [2]),
        StringAttr("private"),
    )

    c = interpreter.run_op(op, ())
    assert c[0] == ShapedArray([1, 320], [2])


def test_ml_program_global_load_constant():
    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(MLProgramFunctions())
    op = ml_program.GlobalLoadConstant(
        SymbolRefAttr("global_op"),
        TensorType(i32, [4]),
    )

    c = interpreter.run_op(op, ())
    assert c[0] == ShapedArray([None], [4])
