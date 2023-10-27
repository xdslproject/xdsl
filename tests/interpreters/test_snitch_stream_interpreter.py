from xdsl.dialects import snitch_stream
from xdsl.dialects.builtin import ArrayAttr, IntAttr, ModuleOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.snitch_stream import SnitchStreamFunctions, StridePattern


def test_riscv_interpreter():
    # register = riscv.IntRegisterType.unallocated()
    # fregister = riscv.FloatRegisterType.unallocated()

    interpreter = Interpreter(ModuleOp([]))
    interpreter.register_implementations(RiscvFunctions())
    interpreter.register_implementations(SnitchStreamFunctions())

    stride_pattern_op = snitch_stream.StridePatternOp(
        ArrayAttr((IntAttr(2), IntAttr(3))),
        ArrayAttr((IntAttr(12), IntAttr(4))),
        IntAttr(0),
    )

    assert interpreter.run_op(stride_pattern_op, ()) == (
        StridePattern([2, 3], [12, 4]),
    )

    # Generic
    # Strided read
    # Strided write
