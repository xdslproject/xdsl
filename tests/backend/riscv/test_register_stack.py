import re

import pytest

from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.dialects import riscv
from xdsl.dialects.builtin import IntAttr


def test_default_reserved_registers():
    register_stack = RiscvRegisterStack.get()
    int_register_set = riscv.IntRegisterType.name

    for reg in (
        riscv.Registers.ZERO,
        riscv.Registers.SP,
        riscv.Registers.GP,
        riscv.Registers.TP,
        riscv.Registers.FP,
        riscv.Registers.S0,
    ):
        available_before = register_stack.available_registers[int_register_set].copy()
        register_stack.push(reg)
        assert available_before == register_stack.available_registers[int_register_set]


def test_push_j_register():
    register_stack = RiscvRegisterStack()

    j0 = riscv.IntRegisterType.infinite_register(0)
    register_stack.push(j0)
    assert register_stack.pop(riscv.IntRegisterType) == j0

    fj0 = riscv.FloatRegisterType.infinite_register(0)
    register_stack.push(fj0)
    assert register_stack.pop(riscv.FloatRegisterType) == fj0


def test_push_register():
    register_stack = RiscvRegisterStack.get()

    register_stack.push(riscv.Registers.A0)
    assert register_stack.pop(riscv.IntRegisterType) == riscv.Registers.A0

    register_stack.push(riscv.Registers.FA0)
    assert register_stack.pop(riscv.FloatRegisterType) == riscv.Registers.FA0


def test_reserve_register():
    register_stack = RiscvRegisterStack.get()

    j0 = riscv.IntRegisterType.infinite_register(0)
    assert isinstance(j0.index, IntAttr)

    reserved_int_registers = register_stack.reserved_registers[
        riscv.IntRegisterType.name
    ]

    register_stack.reserve_register(j0)
    assert reserved_int_registers[j0.index.data] == 1

    register_stack.reserve_register(j0)
    assert reserved_int_registers[j0.index.data] == 2

    register_stack.unreserve_register(j0)
    assert reserved_int_registers[j0.index.data] == 1

    register_stack.unreserve_register(j0)
    assert j0 not in reserved_int_registers
    assert j0 not in register_stack.available_registers[j0.name]

    # Check assertion error when reserving an available register
    reg = register_stack.pop(riscv.IntRegisterType)
    register_stack.push(reg)
    register_stack.reserve_register(reg)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Cannot pop a reserved register ({reg.register_name.data}), it must have been reserved while available."
        ),
    ):
        register_stack.pop(riscv.IntRegisterType)
