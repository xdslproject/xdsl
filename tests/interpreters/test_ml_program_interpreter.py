from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, ml_program
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
from xdsl.interpreters.utils.ptr import TypedPtr


def test_ml_program_global_load_constant():
    tensor_type = TensorType(i32, [4])
    module = ModuleOp([])
    with ImplicitBuilder(module.body):
        ml_program.GlobalOp(
            StringAttr("my_global"),
            tensor_type,
            None,
            DenseIntOrFPElementsAttr.from_list(tensor_type, [4]),
            StringAttr("private"),
        )
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            fetch = ml_program.GlobalLoadConstantOp(
                SymbolRefAttr("my_global"), tensor_type
            )

    interpreter = Interpreter(module)
    interpreter.register_implementations(MLProgramFunctions())

    (result,) = interpreter.run_op(fetch, ())
    assert result == ShapedArray(TypedPtr.new_int32([4] * 4), [4])


def test_ml_program_global_load_constant_ex2():
    tensor_type = TensorType(i64, [2])
    module = ModuleOp([])
    with ImplicitBuilder(module.body):
        ml_program.GlobalOp(
            StringAttr("my_global"),
            tensor_type,
            None,
            DenseIntOrFPElementsAttr.from_list(tensor_type, [1, 320]),
            StringAttr("private"),
        )
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            fetch = ml_program.GlobalLoadConstantOp(
                SymbolRefAttr("my_global"), tensor_type
            )

    interpreter = Interpreter(module)
    interpreter.register_implementations(MLProgramFunctions())

    (result,) = interpreter.run_op(fetch, ())
    assert result == ShapedArray(TypedPtr.new_int64([1, 320]), [2])
