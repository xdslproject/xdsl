import jax
import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.dialects import func, stablehlo
from xdsl.dialects.builtin import ModuleOp, StringAttr, TensorType, i32
from xdsl.irdl import IRDLOperation, attr_def, irdl_op_definition
from xdsl.traits import SymbolOpInterface

pytest.importorskip("jax")

from xdsl.backend.jax_executable import JaxExecutable, array  # noqa: E402


def test_abs():
    TI32 = TensorType(i32, ())

    main_op = func.FuncOp("main", ((TI32,), (TI32,)))
    with ImplicitBuilder(main_op.body) as (arg,):
        res = stablehlo.AbsOp(arg).result
        func.Return(res)

    module = ModuleOp([main_op])

    executable = JaxExecutable.compile(module)

    assert executable.execute([array(-2, dtype=jax.numpy.int32)])[0] == array(
        2, dtype=jax.numpy.int32
    )
    assert executable.execute([array(0, dtype=jax.numpy.int32)])[0] == array(
        0, dtype=jax.numpy.int32
    )
    assert executable.execute([array(2, dtype=jax.numpy.int32)])[0] == array(
        2, dtype=jax.numpy.int32
    )


def test_no_main():
    with pytest.raises(ValueError, match="No `main` function in module"):
        module = ModuleOp([])
        JaxExecutable.compile(module)

    TI32 = TensorType(i32, ())

    with pytest.raises(ValueError, match="No `main` function in module"):
        main_op = func.FuncOp("not_main", ((TI32,), (TI32,)))
        with ImplicitBuilder(main_op.body) as (arg,):
            res = stablehlo.AbsOp(arg).result
            func.Return(res)

        module = ModuleOp([main_op])

        JaxExecutable.compile(module)


def test_main_not_func():
    @irdl_op_definition
    class SymNameOp(IRDLOperation):
        name = "sym_name"

        sym_name = attr_def(StringAttr)
        traits = frozenset((SymbolOpInterface(),))

    module = ModuleOp([SymNameOp(attributes={"sym_name": StringAttr("main")})])

    with pytest.raises(ValueError, match="`main` operation is not a `func.func`"):
        JaxExecutable.compile(module)
