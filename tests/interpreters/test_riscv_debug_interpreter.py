from io import StringIO

from xdsl.dialects import riscv, riscv_debug
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.riscv_debug import RiscvDebugFunctions
from xdsl.utils.test_value import TestSSAValue


def test_riscv_interpreter():
    module_op = ModuleOp([])
    register = riscv.IntRegisterType()
    fregister = riscv.FloatRegisterType()

    riscv_functions = RiscvFunctions()
    riscv_debug_functions = RiscvDebugFunctions()
    file = StringIO()
    interpreter = Interpreter(module_op, file=file)
    interpreter.register_implementations(riscv_functions)
    interpreter.register_implementations(riscv_debug_functions)

    assert (
        interpreter.run_op(
            riscv_debug.PrintfOp(
                "{} {} {} {}",
                (
                    TestSSAValue(register),
                    TestSSAValue(register),
                    TestSSAValue(fregister),
                    TestSSAValue(fregister),
                ),
            ),
            (1, -1, 2.0, -2.0),
        )
        == ()
    )

    assert file.getvalue() == "1 -1 2.0 -2.0"
