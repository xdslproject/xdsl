import re

import pytest

from xdsl.backend.riscv.riscv_register_queue import RiscvRegisterQueue
from xdsl.dialects import riscv
from xdsl.dialects.builtin import IntAttr


def test_default_reserved_registers():
    register_queue = RiscvRegisterQueue.default()
    int_register_set = riscv.IntRegisterType.name

    for reg in (
        riscv.Registers.ZERO,
        riscv.Registers.SP,
        riscv.Registers.GP,
        riscv.Registers.TP,
        riscv.Registers.FP,
        riscv.Registers.S0,
    ):
        available_before = register_queue.available_registers[int_register_set].copy()
        register_queue.push(reg)
        assert available_before == register_queue.available_registers[int_register_set]


def test_push_j_register():
    register_queue = RiscvRegisterQueue()

    j0 = riscv.IntRegisterType.infinite_register(0)
    register_queue.push(j0)
    assert register_queue.pop(riscv.IntRegisterType) == j0

    fj0 = riscv.FloatRegisterType.infinite_register(0)
    register_queue.push(fj0)
    assert register_queue.pop(riscv.FloatRegisterType) == fj0


def test_push_register():
    register_queue = RiscvRegisterQueue()

    register_queue.push(riscv.Registers.A0)
    assert register_queue.pop(riscv.IntRegisterType) == riscv.Registers.A0

    register_queue.push(riscv.Registers.FA0)
    assert register_queue.pop(riscv.FloatRegisterType) == riscv.Registers.FA0


def test_reserve_register():
    register_queue = RiscvRegisterQueue()

    j0 = riscv.IntRegisterType.infinite_register(0)
    assert isinstance(j0.index, IntAttr)

    reserved_int_registers = register_queue.reserved_registers[
        riscv.IntRegisterType.name
    ]

    register_queue.reserve_register(j0)
    assert reserved_int_registers[j0.index.data] == 1

    register_queue.reserve_register(j0)
    assert reserved_int_registers[j0.index.data] == 2

    register_queue.unreserve_register(j0)
    assert reserved_int_registers[j0.index.data] == 1

    register_queue.unreserve_register(j0)
    assert j0 not in reserved_int_registers
    assert j0 not in register_queue.available_registers[j0.name]

    # Check assertion error when reserving an available register
    reg = register_queue.pop(riscv.IntRegisterType)
    register_queue.push(reg)
    register_queue.reserve_register(reg)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            f"Cannot pop a reserved register ({reg.register_name.data}), it must have been reserved while available."
        ),
    ):
        register_queue.pop(riscv.IntRegisterType)


def test_limit():
    register_queue = RiscvRegisterQueue.default()
    register_queue.limit_registers(1)

    assert not register_queue.pop(riscv.IntRegisterType).register_name.data.startswith(
        "j"
    )
    assert register_queue.pop(riscv.IntRegisterType).register_name.data.startswith("j")

    assert not register_queue.pop(
        riscv.FloatRegisterType
    ).register_name.data.startswith("fj")
    assert register_queue.pop(riscv.FloatRegisterType).register_name.data.startswith(
        "fj"
    )
