import struct

import pytest

from xdsl.builder import Builder, ImplicitBuilder
from xdsl.dialects import riscv
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    IntegerAttr,
    ModuleOp,
    TensorType,
    f64,
    i1,
    i32,
)
from xdsl.interpreter import Interpreter, PythonValues
from xdsl.interpreters.riscv import RiscvFunctions
from xdsl.interpreters.utils.ptr import RawPtr, TypedPtr
from xdsl.ir import Block, Region
from xdsl.utils.bitwise_casts import convert_f32_to_u32
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.test_value import create_ssa_value


def test_riscv_interpreter():
    def my_custom_instruction(
        interpreter: Interpreter,
        op: riscv.CustomAssemblyInstructionOp,
        args: PythonValues,
    ) -> PythonValues:
        return args

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
    register = riscv.Registers.UNALLOCATED_INT
    fregister = riscv.Registers.UNALLOCATED_FLOAT

    riscv_functions = RiscvFunctions(
        custom_instructions={"my_custom_instruction": my_custom_instruction},
    )

    interpreter = Interpreter(module_op)
    interpreter.register_implementations(riscv_functions)

    assert interpreter.run_op(riscv.MVOp(create_ssa_value(register)), (42,)) == (42,)

    assert interpreter.run_op(riscv.SltiuOp(create_ssa_value(register), 5), (0,)) == (
        1,
    )
    assert interpreter.run_op(riscv.SltiuOp(create_ssa_value(register), 5), (10,)) == (
        0,
    )
    assert interpreter.run_op(riscv.SltiuOp(create_ssa_value(register), 5), (-10,)) == (
        0,
    )

    assert interpreter.run_op(riscv.SltiuOp(create_ssa_value(register), 0), (5,)) == (
        0,
    )
    assert interpreter.run_op(riscv.SltiuOp(create_ssa_value(register), 10), (5,)) == (
        1,
    )
    assert interpreter.run_op(riscv.SltiuOp(create_ssa_value(register), -10), (5,)) == (
        1,
    )

    assert interpreter.run_op(
        riscv.AddOp(create_ssa_value(register), create_ssa_value(register)),
        (1, 2),
    ) == (3,)

    assert interpreter.run_op(
        riscv.AddiOp(create_ssa_value(register), 2),
        (1,),
    ) == (3,)

    assert interpreter.run_op(
        riscv.SubOp(
            create_ssa_value(register),
            create_ssa_value(register),
        ),
        (1, 2),
    ) == (-1,)

    assert interpreter.run_op(
        riscv.SllOp(
            create_ssa_value(register),
            create_ssa_value(register),
        ),
        (3, 2),
    ) == (12,)

    assert interpreter.run_op(
        riscv.MulOp(
            create_ssa_value(register),
            create_ssa_value(register),
        ),
        (2, 3),
    ) == (6,)

    assert interpreter.run_op(
        riscv.DivOp(
            create_ssa_value(register),
            create_ssa_value(register),
        ),
        (6, 3),
    ) == (2,)

    # Buffer to be modified by the interpreter
    buffer = RawPtr.zeros(16)
    # Buffer to be modified by the test
    test_buffer = RawPtr.zeros(16)

    assert (
        interpreter.run_op(
            riscv.SwOp(create_ssa_value(register), create_ssa_value(register), 0),
            (buffer, 1),
        )
        == ()
    )

    test_buffer.int32[0] = 1
    assert buffer == test_buffer

    assert (
        interpreter.run_op(
            riscv.SwOp(create_ssa_value(register), create_ssa_value(register), 4),
            (buffer, 2),
        )
        == ()
    )

    test_buffer.int32[1] = 2
    assert buffer == test_buffer

    assert interpreter.run_op(riscv.LwOp(create_ssa_value(register), 0), (buffer,)) == (
        1,
    )
    assert interpreter.run_op(riscv.LabelOp("label"), ()) == ()

    custom_instruction_op = riscv.CustomAssemblyInstructionOp(
        "my_custom_instruction",
        (create_ssa_value(register), create_ssa_value(register)),
        (register, register),
    )

    assert interpreter.run_op(custom_instruction_op, (1, 2)) == (1, 2)

    assert interpreter.run_op(
        riscv.FMulSOp(
            create_ssa_value(fregister),
            create_ssa_value(fregister),
        ),
        (3.0, 4.0),
    ) == (12.0,)

    # D extension arithmetic

    assert interpreter.run_op(
        riscv.FMAddDOp(
            create_ssa_value(fregister),
            create_ssa_value(fregister),
            create_ssa_value(fregister),
        ),
        (3.0, 4.0, 5.0),
    ) == (17.0,)

    assert interpreter.run_op(
        riscv.FAddDOp(
            create_ssa_value(fregister),
            create_ssa_value(fregister),
        ),
        (3.0, 4.0),
    ) == (7.0,)

    assert interpreter.run_op(
        riscv.FSubDOp(
            create_ssa_value(fregister),
            create_ssa_value(fregister),
        ),
        (3.0, 4.0),
    ) == (-1.0,)

    assert interpreter.run_op(
        riscv.FMulDOp(
            create_ssa_value(fregister),
            create_ssa_value(fregister),
        ),
        (3.0, 4.0),
    ) == (12.0,)

    assert interpreter.run_op(
        riscv.FDivDOp(
            create_ssa_value(fregister),
            create_ssa_value(fregister),
        ),
        (3.0, 4.0),
    ) == (0.75,)

    assert interpreter.run_op(
        riscv.FMinDOp(
            create_ssa_value(fregister),
            create_ssa_value(fregister),
        ),
        (1, 2),
    ) == (1,)

    assert interpreter.run_op(
        riscv.FMaxDOp(
            create_ssa_value(fregister),
            create_ssa_value(fregister),
        ),
        (1, 2),
    ) == (2,)

    assert interpreter.run_op(
        riscv.FMVOp(create_ssa_value(register)),
        (42.0,),
    ) == (42.0,)

    # same behaviour as riscemu currently, but incorrect
    # the top line is the one that should pass, the other is the same as riscemu
    # assert interpreter.run_op(riscv.FMvWXOp(test_ssa_value(fregister)), (3,)) == (3.0,)
    assert interpreter.run_op(
        riscv.FMvWXOp(create_ssa_value(fregister)),
        (convert_f32_to_u32(3.0),),
    ) == (3.0,)

    assert (
        interpreter.run_op(
            riscv.FSwOp(create_ssa_value(register), create_ssa_value(fregister), 8),
            (buffer, 3.0),
        )
        == ()
    )

    test_buffer.float32[2] = 3.0
    assert buffer == test_buffer

    assert interpreter.run_op(
        riscv.FLwOp(create_ssa_value(register), 8),
        (buffer,),
    ) == (3.0,)

    assert buffer == test_buffer

    # Store a second float for us to read
    buffer.float32[3] = 4.0
    test_buffer.float32[3] = 4.0

    assert interpreter.run_op(
        riscv.FLdOp(create_ssa_value(register), 8),
        (buffer,),
    ) == (struct.unpack("<d", struct.pack("<ff", 3.0, 4.0))[0],)

    assert buffer == test_buffer

    assert interpreter.run_op(
        riscv.FMvDOp(create_ssa_value(register), rd=fregister),
        (5.0,),
    ) == (5.0,)

    assert interpreter.run_op(
        riscv.FCvtDWOp(create_ssa_value(register)),
        (42,),
    ) == (42.0,)

    assert (
        interpreter.run_op(
            riscv.FSdOp(create_ssa_value(register), create_ssa_value(fregister), 8),
            (buffer, struct.unpack("<d", struct.pack("<ff", 5.0, 6.0))[0]),
        )
        == ()
    )

    test_buffer.float32[2] = 5.0
    test_buffer.float32[3] = 6.0
    assert buffer == test_buffer


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
        "one": TypedPtr.new_int32([1]).raw,
        "two_three": TypedPtr.new_int32([2, 3]).raw,
    }


def test_cast():
    module_op = ModuleOp([])

    riscv_functions = RiscvFunctions()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(riscv_functions)

    assert interpreter.cast_value(riscv.Registers.UNALLOCATED_FLOAT, f64, 42.0) == 42.0


def test_register_contents():
    module_op = ModuleOp([])

    riscv_functions = RiscvFunctions()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(riscv_functions)

    assert RiscvFunctions.registers(interpreter) == {
        riscv.Registers.ZERO.register_name: 0,
        riscv.Registers.SP.register_name: RawPtr(bytearray(1 << 20), offset=1 << 20),
    }

    with pytest.raises(
        InterpretationError, match="Unexpected type i1, expected register type"
    ):
        RiscvFunctions.get_reg_value(interpreter, i1, 2)

    with pytest.raises(
        InterpretationError, match="Unexpected type i1, expected register type"
    ):
        RiscvFunctions.set_reg_value(interpreter, i1, 2)

    assert RiscvFunctions.get_reg_value(interpreter, riscv.Registers.ZERO, 0) == 0
    assert RiscvFunctions.set_reg_value(interpreter, riscv.Registers.ZERO, 1) == 0
    assert RiscvFunctions.get_reg_value(interpreter, riscv.Registers.ZERO, 0) == 0

    with pytest.raises(
        InterpretationError, match="Runtime and stored value mismatch: 1 != 0"
    ):
        RiscvFunctions.get_reg_value(interpreter, riscv.Registers.ZERO, 1)

    with pytest.raises(
        InterpretationError, match="Value not found for register name t0"
    ):
        RiscvFunctions.get_reg_value(interpreter, riscv.Registers.T0, 2)

    assert RiscvFunctions.set_reg_value(interpreter, riscv.Registers.T0, 1) == 1

    assert RiscvFunctions.get_reg_value(interpreter, riscv.Registers.T0, 1) == 1

    with pytest.raises(
        InterpretationError, match="Runtime and stored value mismatch: 2 != 1"
    ):
        RiscvFunctions.get_reg_value(interpreter, riscv.Registers.T0, 2)


def test_values():
    module_op = ModuleOp([])

    riscv_functions = RiscvFunctions()
    interpreter = Interpreter(module_op)
    interpreter.register_implementations(riscv_functions)
    assert interpreter.value_for_attribute(IntegerAttr(1, i32), riscv.Registers.A0) == 1

    assert (
        interpreter.value_for_attribute(
            DenseIntOrFPElementsAttr.from_list(
                TensorType(i32, [2, 3]), tuple(range(6))
            ),
            riscv.Registers.A0,
        )
        == TypedPtr.new_int32(tuple(range(6))).raw
    )
