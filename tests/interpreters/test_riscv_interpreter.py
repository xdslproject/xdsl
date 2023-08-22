from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import Interpreter, PythonValues
from xdsl.interpreters.riscv import Buffer, RiscvFunctions
from xdsl.ir.core import Block, Region
from xdsl.utils.bitwise_casts import convert_f32_to_u32
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
        data={"label0": [42]},
        custom_instructions={"my_custom_instruction": my_custom_instruction},
    )
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(riscv_functions)

    assert interpreter.run_op(riscv.LiOp("label0"), ()) == (
        Buffer(data=[42], offset=0),
    )
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
        riscv.AddOp(TestSSAValue(register), TestSSAValue(register)), (1, 2)
    ) == (3,)

    assert interpreter.run_op(
        riscv.MulOp(TestSSAValue(register), TestSSAValue(register)), (2, 3)
    ) == (6,)

    buffer = Buffer([0, 0, 0, 0])

    assert (
        interpreter.run_op(
            riscv.SwOp(TestSSAValue(register), TestSSAValue(register), 0), (buffer, 1)
        )
        == ()
    )

    assert buffer == Buffer([1, 0, 0, 0])

    assert (
        interpreter.run_op(
            riscv.SwOp(TestSSAValue(register), TestSSAValue(register), 1), (buffer, 2)
        )
        == ()
    )

    assert buffer == Buffer([1, 2, 0, 0])

    assert interpreter.run_op(riscv.LwOp(TestSSAValue(register), 0), (buffer,)) == (1,)
    assert interpreter.run_op(riscv.LabelOp("label"), ()) == ()

    custom_instruction_op = riscv.CustomAssemblyInstructionOp(
        "my_custom_instruction",
        (TestSSAValue(register), TestSSAValue(register)),
        (register, register),
    )

    assert interpreter.run_op(custom_instruction_op, (1, 2)) == (1, 2)

    assert interpreter.run_op(
        riscv.FMulSOp(TestSSAValue(fregister), TestSSAValue(fregister)), (3.0, 4.0)
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
            riscv.FSwOp(TestSSAValue(register), TestSSAValue(fregister), 2),
            (buffer, 3.0),
        )
        == ()
    )

    assert buffer == Buffer([1, 2, 3.0, 0])

    assert interpreter.run_op(
        riscv.FLwOp(TestSSAValue(register), 2),
        (buffer,),
    ) == (3.0,)

    assert buffer == Buffer([1, 2, 3.0, 0])


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

    assert RiscvFunctions.get_data(module) == {"one": [1], "two_three": [2, 3]}
