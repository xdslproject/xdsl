import pytest

from xdsl.builder import Builder
from xdsl.dialects import riscv, riscv_func
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv_func import RiscvFuncFunctions
from xdsl.ir import BlockArgument

index = IndexType()


@ModuleOp
@Builder.implicit_region
def my_module():
    a0 = riscv.Registers.A0

    @Builder.implicit_region((a0,))
    def body(args: tuple[BlockArgument, ...]) -> None:
        riscv_func.ReturnOp(*args)

    riscv_func.FuncOp("id", body, ((a0,), (a0,)))


def scf_interp(module_op: ModuleOp, func_name: str, n: int) -> int:
    module_op.verify()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(RiscvFuncFunctions())
    (result,) = interpreter.call_op(func_name, (n,))
    return result


@pytest.mark.parametrize("n,res", [(0, 0), (1, 1)])
def test_sum_to(n: int, res: int):
    assert res == scf_interp(my_module, "id", n)
