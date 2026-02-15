import pytest

from xdsl.dialects import riscv, rv32
from xdsl.dialects.builtin import (
    ModuleOp,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.rv32 import Rv32Functions
from xdsl.interpreters.utils.ptr import TypedPtr
from xdsl.ir import Block, Region
from xdsl.utils.exceptions import InterpretationError


def test_riscv_interpreter():
    module_op = ModuleOp(
        [
            riscv.AssemblySectionOp(
                ".data",
                Region(
                    Block([riscv.LabelOp("label0"), riscv.DirectiveOp(".word", "2A")])
                ),
            ),
            riscv.AssemblySectionOp(
                ".data",
                Region(
                    Block([riscv.LabelOp("label1"), riscv.DirectiveOp(".word", "3B")])
                ),
            ),
        ]
    )

    rv32_functions = Rv32Functions()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(rv32_functions)

    assert interpreter.run_op(rv32.LiOp("label0"), ()) == (
        TypedPtr.new_int32((42,)).raw,
    )
    assert interpreter.run_op(rv32.LiOp("label1"), ()) == (
        TypedPtr.new_int32((59,)).raw,
    )

    assert interpreter.run_op(rv32.GetRegisterOp(riscv.Registers.ZERO), ()) == (0,)

    get_non_zero = rv32.GetRegisterOp(riscv.Registers.UNALLOCATED_INT)
    with pytest.raises(
        InterpretationError,
        match="Cannot get value for unallocated register !riscv.reg",
    ):
        interpreter.run_op(get_non_zero, ())


def test_register_contents():
    module_op = ModuleOp([])

    rv32_functions = Rv32Functions()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(rv32_functions)

    assert RiscvFunctions.set_reg_value(interpreter, riscv.Registers.T0, 1) == 1

    assert interpreter.run_op(rv32.GetRegisterOp(riscv.Registers.ZERO), ()) == (0,)
    assert interpreter.run_op(rv32.GetRegisterOp(riscv.Registers.T0), ()) == (1,)
