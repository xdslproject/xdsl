from io import StringIO

from xdsl.dialects import riscv, riscv_debug
from xdsl.dialects.builtin import FloatAttr, IntegerAttr, ModuleOp, f64, i32
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.riscv_debug import RiscvDebugFunctions
from xdsl.utils.test_value import create_ssa_value


def test_riscv_interpreter():
    module_op = ModuleOp([])

    riscv_functions = RiscvFunctions()
    riscv_debug_functions = RiscvDebugFunctions()
    file = StringIO()
    interpreter = Interpreter(module_op, file=file, index_bitwidth=32)
    interpreter.register_implementations(riscv_functions)
    interpreter.register_implementations(riscv_debug_functions)

    assert (
        interpreter.run_op(
            riscv_debug.PrintfOp(
                "{} {} {} {}",
                (
                    create_ssa_value(riscv.Registers.UNALLOCATED_INT),
                    create_ssa_value(riscv.Registers.UNALLOCATED_INT),
                    create_ssa_value(riscv.Registers.UNALLOCATED_FLOAT),
                    create_ssa_value(riscv.Registers.UNALLOCATED_FLOAT),
                ),
            ),
            (
                IntegerAttr(1, i32),
                IntegerAttr(-1, i32),
                FloatAttr(2.0, f64),
                FloatAttr(-2.0, f64),
            ),
        )
        == ()
    )

    assert file.getvalue() == "1 -1 2.0 -2.0"
