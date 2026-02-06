import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import func, riscv, riscv_scf, rv32
from xdsl.dialects.builtin import IndexType, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.func import FuncFunctions
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.riscv_scf import RiscvScfFunctions
from xdsl.ir import BlockArgument

index = IndexType()
register = riscv.Registers.UNALLOCATED_INT


def sum_to_for_fn(n: int) -> int:
    """
    Python implementation of sum_to_for_op
    """
    result = 0
    for i in range(0, n, 1):
        result += i
    return result


@ModuleOp
@Builder.implicit_region
def sum_to_for_op():
    with ImplicitBuilder(func.FuncOp("sum_to", ((register,), (register,))).body) as (
        ub,
    ):
        lb = rv32.LiOp(0)
        step = rv32.LiOp(1)
        initial = rv32.LiOp(0)

        @Builder.implicit_region((register, register))
        def for_loop_region(args: tuple[BlockArgument, ...]):
            (i, acc) = args
            res = riscv.AddOp(i, acc)
            riscv_scf.YieldOp(res)

        result = riscv_scf.ForOp(lb, ub, step, (initial,), for_loop_region)
        func.ReturnOp(result)


# Python implementation of `sum_to_while_op`
# def sum_to_while_fn(n: int) -> int:
#     result = 0
#     i = 0
#     while i < n:
#         result += 1
#         i += 1
#     return result


@ModuleOp
@Builder.implicit_region
def sum_to_while_op():
    with ImplicitBuilder(func.FuncOp("sum_to", ((register,), (register,))).body) as (
        ub,
    ):
        lb = rv32.LiOp(0)
        step = rv32.LiOp(1)
        initial = rv32.LiOp(0)

        @Builder.implicit_region((register, register, register, register))
        def before_region(args: tuple[BlockArgument, ...]):
            (acc0, i0, ub0, step0) = args
            cond = riscv.SltOp(i0, ub0).rd
            riscv_scf.ConditionOp(cond, acc0, i0, ub0, step0)

        @Builder.implicit_region((register, register, register, register))
        def after_region(args: tuple[BlockArgument, ...]):
            (acc1, i1, ub1, step1) = args
            res = riscv.AddOp(i1, acc1).rd
            i2 = riscv.AddOp(i1, step1).rd
            riscv_scf.YieldOp(res, i2, ub1, step1)

        result, *_ = riscv_scf.WhileOp(
            (initial, lb, ub, step),
            (register, register, register, register),
            before_region,
            after_region,
        ).res
        func.ReturnOp(result)


def interp(module_op: ModuleOp, func_name: str, n: int) -> int:
    module_op.verify()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(RiscvScfFunctions())
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(FuncFunctions())
    (result,) = interpreter.call_op(func_name, (n,))
    return result


@pytest.mark.parametrize("n,res", [(0, 0), (1, 0), (2, 1), (3, 3), (4, 6), (5, 10)])
def test_sum_to(n: int, res: int):
    assert res == interp(sum_to_for_op, "sum_to", n)
    assert res == interp(sum_to_while_op, "sum_to", n)
