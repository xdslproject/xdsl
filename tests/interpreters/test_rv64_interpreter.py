import re

import pytest

from xdsl.dialects import riscv, rv64
from xdsl.dialects.builtin import (
    IntegerAttr,
    ModuleOp,
    i64,
)
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.rv64 import Rv64Functions
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

    rv64_functions = Rv64Functions()
    interpreter = Interpreter(module_op, index_bitwidth=64)
    interpreter.register_implementations(rv64_functions)

    assert interpreter.run_op(rv64.LiOp("label0"), ()) == (
        TypedPtr.new_int32((42,)).raw,
    )
    assert interpreter.run_op(rv64.LiOp("label1"), ()) == (
        TypedPtr.new_int32((59,)).raw,
    )

    assert interpreter.run_op(rv64.GetRegisterOp(riscv.Registers.ZERO), ()) == (
        IntegerAttr(0, i64),
    )

    get_non_zero = rv64.GetRegisterOp(riscv.Registers.UNALLOCATED_INT)
    with pytest.raises(
        InterpretationError,
        match=re.escape("Cannot get value for unallocated register !riscv.reg"),
    ):
        interpreter.run_op(get_non_zero, ())


def test_register_contents():
    module_op = ModuleOp([])

    rv64_functions = Rv64Functions()
    interpreter = Interpreter(module_op, index_bitwidth=64)
    interpreter.register_implementations(rv64_functions)

    one = IntegerAttr(1, i64)
    assert RiscvFunctions.set_reg_value(interpreter, riscv.Registers.T0, one) == one

    assert interpreter.run_op(rv64.GetRegisterOp(riscv.Registers.ZERO), ()) == (
        IntegerAttr(0, i64),
    )
    assert interpreter.run_op(rv64.GetRegisterOp(riscv.Registers.T0), ()) == (one,)
