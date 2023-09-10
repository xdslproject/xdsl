import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, PythonValues
from xdsl.interpreters.riscv import RawPtr, RiscvFunctions
from xdsl.ir.core import Block, Region
from xdsl.utils.bitwise_casts import convert_f32_to_u32
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import TestSSAValue


def test_riscv_interpreter():
    def my_custom_instruction(
        interpreter: Interpreter,
        op: riscv.CustomAssemblyInstructionOp,
        args: PythonValues,
    ) -> PythonValues:
        return args

    module_op = ModuleOp([])
    register = riscv.IntRegisterType.unallocated()
    fregister = riscv.FloatRegisterType.unallocated()

    riscv_functions = RiscvFunctions(
        module_op,
        data={"label0": RawPtr.new_int32([42])},
        custom_instructions={"my_custom_instruction": my_custom_instruction},
    )
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(riscv_functions)

    assert interpreter.run_op(riscv.LiOp("label0"), ()) == (RawPtr.new_int32((42,)),)
    assert interpreter.run_op(
        riscv.MVOp(TestSSAValue(register), rd=riscv.IntRegisterType.unallocated()),
        (42,),
    ) == (42,)

    assert interpreter.run_op(riscv.SltiuOp(TestSSAValue(register), 5), (0,)) == (1,)
    assert interpreter.run_op(riscv.SltiuOp(TestSSAValue(register), 5), (10,)) == (0,)
    assert interpreter.run_op(riscv.SltiuOp(TestSSAValue(register), 5), (-10,)) == (0,)

    assert interpreter.run_op(riscv.SltiuOp(TestSSAValue(register), 0), (5,)) == (0,)
    assert interpreter.run_op(riscv.SltiuOp(TestSSAValue(register), 10), (5,)) == (1,)
    assert interpreter.run_op(riscv.SltiuOp(TestSSAValue(register), -10), (5,)) == (1,)

    assert interpreter.run_op(
        riscv.AddOp(
            TestSSAValue(register),
            TestSSAValue(register),
            rd=riscv.IntRegisterType.unallocated(),
        ),
        (1, 2),
    ) == (3,)

    assert interpreter.run_op(
        riscv.MulOp(
            TestSSAValue(register),
            TestSSAValue(register),
            rd=riscv.IntRegisterType.unallocated(),
        ),
        (2, 3),
    ) == (6,)

    # Buffer to be modified by the interpreter
    buffer = RawPtr.zeros(16)
    # Buffer to be modified by the test
    test_buffer = RawPtr.zeros(16)

    assert (
        interpreter.run_op(
            riscv.SwOp(TestSSAValue(register), TestSSAValue(register), 0), (buffer, 1)
        )
        == ()
    )

    test_buffer.int32[0] = 1
    assert buffer == test_buffer

    assert (
        interpreter.run_op(
            riscv.SwOp(TestSSAValue(register), TestSSAValue(register), 4), (buffer, 2)
        )
        == ()
    )

    test_buffer.int32[1] = 2
    assert buffer == test_buffer

    assert interpreter.run_op(riscv.LwOp(TestSSAValue(register), 0), (buffer,)) == (1,)
    assert interpreter.run_op(riscv.LabelOp("label"), ()) == ()

    custom_instruction_op = riscv.CustomAssemblyInstructionOp(
        "my_custom_instruction",
        (TestSSAValue(register), TestSSAValue(register)),
        (register, register),
    )

    assert interpreter.run_op(custom_instruction_op, (1, 2)) == (1, 2)

    assert interpreter.run_op(
        riscv.FMulSOp(
            TestSSAValue(fregister),
            TestSSAValue(fregister),
            rd=riscv.FloatRegisterType.unallocated(),
        ),
        (3.0, 4.0),
    ) == (12.0,)

    # same behaviour as riscemu currently, but incorrect
    # the top line is the one that should pass, the other is the same as riscemu
    # assert interpreter.run_op(riscv.FCvtSWOp(TestSSAValue(fregister)), (3,)) == (3.0,)
    assert interpreter.run_op(
        riscv.FCvtSWOp(
            TestSSAValue(fregister), rd=riscv.FloatRegisterType.unallocated()
        ),
        (convert_f32_to_u32(3.0),),
    ) == (3.0,)

    assert (
        interpreter.run_op(
            riscv.FSwOp(TestSSAValue(register), TestSSAValue(fregister), 8),
            (buffer, 3.0),
        )
        == ()
    )

    test_buffer.float32[2] = 3.0
    assert buffer == test_buffer

    assert interpreter.run_op(
        riscv.FLwOp(TestSSAValue(register), 8),
        (buffer,),
    ) == (3.0,)

    assert buffer == test_buffer

    assert interpreter.run_op(riscv.GetRegisterOp(riscv.Registers.ZERO), ()) == (0,)

    get_non_zero = riscv.GetRegisterOp(riscv.IntRegisterType.unallocated())
    with pytest.raises(
        InterpretationError,
        match="Cannot interpret riscv.get_register op with non-ZERO type",
    ):
        interpreter.run_op(get_non_zero, ())


def test_get_data():
    @ModuleOp
    @Builder.implicit_region
    def module():
        data = riscv.AssemblySectionOp(".data", Region(Block()))
        with ImplicitBuilder(data.data):
            riscv.LabelOp("one")
            riscv.DirectiveOp(".word", "1")
            riscv.LabelOp("two_three")
            riscv.DirectiveOp(".word", "2, 3")

    assert RiscvFunctions.get_data(module) == {
        "one": RawPtr.new_int32([1]),
        "two_three": RawPtr.new_int32([2, 3]),
    }
