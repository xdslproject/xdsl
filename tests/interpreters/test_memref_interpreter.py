import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import arith, func, memref
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IndexType,
    ModuleOp,
    StringAttr,
    TensorType,
    i32,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.memref import MemrefFunctions, MemrefValue
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.utils.exceptions import InterpretationError

interpreter = Interpreter(ModuleOp([]))
interpreter.register_implementations(ArithFunctions())
interpreter.register_implementations(MemrefFunctions())

index = IndexType()


def test_functions():
    alloc_op = memref.Alloc.get(i32, None, (2, 3))
    zero_op = arith.Constant.from_int_and_width(0, index)
    one_op = arith.Constant.from_int_and_width(1, index)
    two_op = arith.Constant.from_int_and_width(2, index)
    forty_two_op = arith.Constant.from_int_and_width(42, 32)
    store_op = memref.Store.get(forty_two_op, alloc_op, (zero_op, one_op))
    load_42_op = memref.Load.get(alloc_op, (zero_op, one_op))
    load_undef_op = memref.Load.get(alloc_op, (zero_op, two_op))
    dealloc_op = memref.Dealloc.get(alloc_op)

    (shaped_array,) = interpreter.run_op(alloc_op, ())
    v = MemrefValue.Uninitialized
    assert shaped_array == ShapedArray([v, v, v, v, v, v], [2, 3])
    (zero,) = interpreter.run_op(zero_op, ())
    (one,) = interpreter.run_op(one_op, ())
    (two,) = interpreter.run_op(two_op, ())
    (forty_two_0,) = interpreter.run_op(forty_two_op, ())
    store_res = interpreter.run_op(store_op, (forty_two_0, shaped_array, zero, one))
    assert store_res == ()
    (forty_two_1,) = interpreter.run_op(load_42_op, (shaped_array, zero, one))
    assert forty_two_1 == 42

    with pytest.raises(InterpretationError) as e:
        interpreter.run_op(load_undef_op, (shaped_array, zero, two))
        e.match("uninitialized")

    dealloc_res = interpreter.run_op(dealloc_op, (shaped_array,))
    assert dealloc_res == ()

    with pytest.raises(InterpretationError) as e:
        interpreter.run_op(load_undef_op, (shaped_array, zero, one))
        e.match("deallocated")


def test_memref_get_global():
    memref_type = memref.MemRefType(i32, (2, 2))
    tensor_type = TensorType(i32, (2, 2))
    module = ModuleOp([])
    with ImplicitBuilder(module.body):
        memref.Global.get(
            StringAttr("my_global"),
            memref_type,
            DenseIntOrFPElementsAttr.from_list(tensor_type, [1, 2, 3, 4]),
            sym_visibility=StringAttr("public"),
        )
        with ImplicitBuilder(func.FuncOp("main", ((), ())).body):
            fetch = memref.GetGlobal.get("my_global", memref_type)

    interpreter = Interpreter(module)
    interpreter.register_implementations(MemrefFunctions())

    (result,) = interpreter.run_op(fetch, ())
    assert result == ShapedArray([1, 2, 3, 4], [2, 2])
