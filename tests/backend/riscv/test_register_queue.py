import re

import pytest

from xdsl.backend.riscv.riscv_register_queue import RiscvRegisterQueue
from xdsl.dialects import riscv


def test_default_reserved_registers():
    register_queue = RiscvRegisterQueue()

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
    register_queue = RiscvRegisterQueue()

    register_queue.push(riscv.IntRegisterType.infinite_register(0))
    assert (
        riscv.IntRegisterType.infinite_register(0)
        == register_queue.available_int_registers[-1]
    )

    register_queue.push(riscv.FloatRegisterType.infinite_register(0))
    assert (
        riscv.FloatRegisterType.infinite_register(0)
        == register_queue.available_float_registers[-1]
    )


def test_push_register():
    register_queue = RiscvRegisterQueue()

    register_queue.push(riscv.Registers.A0)
    assert riscv.Registers.A0 == register_queue.available_int_registers[-1]

    register_queue.push(riscv.Registers.FA0)
    assert riscv.Registers.FA0 == register_queue.available_float_registers[-1]


def test_reserve_register():
    register_queue = RiscvRegisterQueue()

    register_queue.reserve_register(riscv.IntRegisterType.infinite_register(0))
    assert (
        register_queue.reserved_int_registers[
            riscv.IntRegisterType.infinite_register(0)
        ]
        == 1
    )

    register_queue.reserve_register(riscv.IntRegisterType.infinite_register(0))
    assert (
        register_queue.reserved_int_registers[
            riscv.IntRegisterType.infinite_register(0)
        ]
        == 2
    )

    register_queue.unreserve_register(riscv.IntRegisterType.infinite_register(0))
    assert (
        register_queue.reserved_int_registers[
            riscv.IntRegisterType.infinite_register(0)
        ]
        == 1
    )

    register_queue.unreserve_register(riscv.IntRegisterType.infinite_register(0))
    assert (
        riscv.IntRegisterType.infinite_register(0)
        not in register_queue.reserved_int_registers
    )
    assert (
        riscv.IntRegisterType.infinite_register(0)
        not in register_queue.available_int_registers
    )

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
    register_queue = RiscvRegisterQueue()
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
