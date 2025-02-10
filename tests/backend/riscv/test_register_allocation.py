import re

import pytest
from typing_extensions import Self

from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.backend.riscv.register_allocation import (
    RegisterAllocatorLivenessBlockNaive,
    reg_types,
)
from xdsl.backend.riscv.riscv_register_queue import RiscvRegisterQueue
from xdsl.dialects import riscv
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.test_value import TestSSAValue


def test_default_reserved_registers():
    register_queue = RiscvRegisterQueue(
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


def test_allocate_with_inout_constraints():
    @irdl_op_definition
    class MyInstructionOp(riscv.RISCVAsmOperation, IRDLOperation):
        name = "riscv.my_instruction"

        rs0 = operand_def()
        rs1 = operand_def()
        rd0 = result_def()
        rd1 = result_def()

        @classmethod
        def get(cls, rs0: str, rs1: str, rd0: str, rd1: str) -> Self:
            return cls.build(
                operands=(
                    TestSSAValue(riscv.IntRegisterType(rs0)),
                    TestSSAValue(riscv.IntRegisterType(rs1)),
                ),
                result_types=(
                    riscv.IntRegisterType(rd0),
                    riscv.IntRegisterType(rd1),
                ),
            )

        def get_register_constraints(self) -> RegisterConstraints:
            return RegisterConstraints(
                (self.rs0,), (self.rd0,), ((self.rs1, self.rd1),)
            )

    register_queue = RiscvRegisterQueue(
        available_int_registers=[], available_float_registers=[]
    )
    register_allocator = RegisterAllocatorLivenessBlockNaive(register_queue)

    # All new registers. The result register is reused by the allocator for the operand.
    op0 = MyInstructionOp.get("", "", "", "")
    register_allocator.process_riscv_op(op0)
    assert op0.rs0.type == riscv.IntRegisterType("j1")
    assert op0.rs1.type == riscv.IntRegisterType("j0")
    assert op0.rd0.type == riscv.IntRegisterType("j1")
    assert op0.rd1.type == riscv.IntRegisterType("j0")

    # One register reserved for inout parameter, the allocator should allocate the output
    # to the same register.
    op1 = MyInstructionOp.get("", "", "", "a0")
    register_allocator.process_riscv_op(op1)
    assert op1.rs0.type == riscv.IntRegisterType("j2")
    assert op1.rs1.type == riscv.IntRegisterType("a0")
    assert op1.rd0.type == riscv.IntRegisterType("j2")
    assert op1.rd1.type == riscv.IntRegisterType("a0")


def test_count_reg_types():
    a0 = riscv.Registers.A0
    a1 = riscv.Registers.A1

    fa0 = riscv.Registers.FA0

    assert reg_types([a0, a0, a1, fa0, fa0]) == ({"a0", "a1"}, {"fa0"})
