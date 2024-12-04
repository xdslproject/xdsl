import re

import pytest

from xdsl.backend.riscv.register_queue import RegisterQueue
from xdsl.dialects import riscv


def test_default_reserved_registers():
    register_queue = RegisterQueue()

    for reg in (
        riscv.Registers.ZERO,
        riscv.Registers.SP,
        riscv.Registers.GP,
        riscv.Registers.TP,
        riscv.Registers.FP,
        riscv.Registers.S0,
    ):
        available_before = register_queue.available_int_registers.copy()
        register_queue.push(reg)
        assert available_before == register_queue.available_int_registers


def test_push_j_register():
    register_queue = RegisterQueue()

    register_queue.push(riscv.IntRegisterType("j0"))
    assert riscv.IntRegisterType("j0") == register_queue.available_int_registers[-1]

    register_queue.push(riscv.FloatRegisterType("j0"))
    assert riscv.FloatRegisterType("j0") == register_queue.available_float_registers[-1]


def test_push_register():
    register_queue = RegisterQueue()

    register_queue.push(riscv.Registers.A0)
    assert riscv.Registers.A0 == register_queue.available_int_registers[-1]

    register_queue.push(riscv.Registers.FA0)
    assert riscv.Registers.FA0 == register_queue.available_float_registers[-1]


def test_reserve_register():
    register_queue = RegisterQueue()

    register_queue.reserve_register(riscv.IntRegisterType("j0"))
    assert register_queue.reserved_registers[riscv.IntRegisterType("j0")] == 1

    register_queue.reserve_register(riscv.IntRegisterType("j0"))
    assert register_queue.reserved_registers[riscv.IntRegisterType("j0")] == 2

    register_queue.unreserve_register(riscv.IntRegisterType("j0"))
    assert register_queue.reserved_registers[riscv.IntRegisterType("j0")] == 1

    register_queue.unreserve_register(riscv.IntRegisterType("j0"))
    assert riscv.IntRegisterType("j0") not in register_queue.reserved_registers
    assert riscv.IntRegisterType("j0") not in register_queue.available_int_registers

    # Check assertion error when reserving an available register
    reg = register_queue.pop(riscv.IntRegisterType)
    register_queue.push(reg)
    register_queue.reserve_register(reg)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Cannot pop a reserved register ({reg.register_name}), it must have been reserved while available."
        ),
    ):
        register_queue.pop(riscv.IntRegisterType)


def test_limit():
    register_queue = RegisterQueue()
    register_queue.limit_registers(1)

    assert not register_queue.pop(riscv.IntRegisterType).register_name.startswith("j")
    assert register_queue.pop(riscv.IntRegisterType).register_name.startswith("j")

    assert not register_queue.pop(riscv.FloatRegisterType).register_name.startswith("j")
    assert register_queue.pop(riscv.FloatRegisterType).register_name.startswith("j")
