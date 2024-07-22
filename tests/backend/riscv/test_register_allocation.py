import re

import pytest

from xdsl.backend.riscv.register_allocation import RegisterAllocatorLivenessBlockNaive
from xdsl.backend.riscv.register_queue import RegisterQueue
from xdsl.dialects import riscv
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.test_value import TestSSAValue


def test_default_reserved_registers():
    register_queue = RegisterQueue(
        available_int_registers=[], available_float_registers=[]
    )

    unallocated = riscv.Registers.UNALLOCATED_INT

    def j(index: int):
        return riscv.IntRegisterType(f"j{index}")

    assert register_queue.pop(riscv.IntRegisterType) == j(0)

    register_allocator = RegisterAllocatorLivenessBlockNaive(register_queue)

    assert not register_allocator.allocate_same(())

    a = TestSSAValue(unallocated)
    register_allocator.allocate_same((a,))

    assert a.type == j(1)

    register_allocator.allocate_same((a,))

    assert a.type == j(1)

    b0 = TestSSAValue(unallocated)
    b1 = TestSSAValue(unallocated)

    register_allocator.allocate_same((b0, b1))

    assert b0.type == j(2)
    assert b1.type == j(2)

    c0 = TestSSAValue(j(2))
    c1 = TestSSAValue(unallocated)

    register_allocator.allocate_same((c0, c1))

    assert c0.type == j(2)
    assert c1.type == j(2)

    d0 = TestSSAValue(j(2))
    d1 = TestSSAValue(j(3))

    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Cannot allocate registers to the same register ['!riscv.reg<j2>', '!riscv.reg<j3>']"
        ),
    ):
        register_allocator.allocate_same((d0, d1))

    e0 = TestSSAValue(j(2))
    e1 = TestSSAValue(j(3))
    e2 = TestSSAValue(unallocated)

    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Cannot allocate registers to the same register ['!riscv.reg', '!riscv.reg<j2>', '!riscv.reg<j3>']"
        ),
    ):
        register_allocator.allocate_same((e0, e1, e2))
