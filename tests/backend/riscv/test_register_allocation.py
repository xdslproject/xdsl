import re

import pytest
from typing_extensions import Self

from xdsl.backend.register_allocatable import RegisterConstraints
from xdsl.backend.riscv.register_allocation import (
    RegisterAllocatorLivenessBlockNaive,
    reg_types_by_name,
)
from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.dialects import riscv
from xdsl.dialects.test import TestOp
from xdsl.ir import SSAValue
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, result_def
from xdsl.utils.exceptions import DiagnosticException


def test_default_reserved_registers():
    available_registers = RiscvRegisterStack(allow_infinite=True)

    unallocated = riscv.Registers.UNALLOCATED_INT

    j = riscv.IntRegisterType.infinite_register

    assert available_registers.pop(riscv.IntRegisterType) == j(0)

    register_allocator = RegisterAllocatorLivenessBlockNaive(available_registers)

    assert not register_allocator.allocate_values_same_reg(())

    op_a = TestOp(result_types=[unallocated])
    register_allocator.allocate_values_same_reg(op_a.results)

    assert op_a.results[0].type == j(1)

    register_allocator.allocate_values_same_reg(op_a.results)

    assert op_a.results[0].type == j(1)

    op_b = TestOp(result_types=[unallocated, unallocated])

    register_allocator.allocate_values_same_reg(op_b.results)

    assert tuple(op_b.result_types) == (j(2), j(2))

    op_c = TestOp(result_types=[j(2), unallocated])

    register_allocator.allocate_values_same_reg(op_c.results)

    assert tuple(op_c.result_types) == (j(2), j(2))

    op_d = TestOp(result_types=[j(2), j(3)])

    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Cannot allocate registers to the same register ['!riscv.reg<j_2>', '!riscv.reg<j_3>']"
        ),
    ):
        register_allocator.allocate_values_same_reg(op_d.results)

    op_e = TestOp(result_types=[j(2), j(3), unallocated])

    with pytest.raises(
        DiagnosticException,
        match=re.escape(
            "Cannot allocate registers to the same register ['!riscv.reg', '!riscv.reg<j_2>', '!riscv.reg<j_3>']"
        ),
    ):
        register_allocator.allocate_values_same_reg(op_e.results)


def test_allocate_with_inout_constraints():
    @irdl_op_definition
    class MyInstructionOp(
        riscv.RISCVAsmOperation, riscv.RISCVRegallocOperation, IRDLOperation
    ):
        name = "riscv.my_instruction"

        rs0 = operand_def()
        rs1 = operand_def()
        rd0 = result_def()
        rd1 = result_def()

        @classmethod
        def get(cls, rs0: SSAValue, rs1: SSAValue, rd0: str, rd1: str) -> Self:
            return cls.build(
                operands=(rs0, rs1),
                result_types=(
                    riscv.IntRegisterType.from_name(rd0),
                    riscv.IntRegisterType.from_name(rd1),
                ),
            )

        def get_register_constraints(self) -> RegisterConstraints:
            return RegisterConstraints(
                (self.rs0,), (self.rd0,), ((self.rs1, self.rd1),)
            )

    available_registers = RiscvRegisterStack(allow_infinite=True)
    register_allocator = RegisterAllocatorLivenessBlockNaive(available_registers)

    # All new registers. The result register is reused by the allocator for the operand.
    rs0, rs1 = TestOp(
        result_types=[
            riscv.IntRegisterType.unallocated(),
            riscv.IntRegisterType.unallocated(),
        ]
    ).results
    op0 = MyInstructionOp.get(rs0, rs1, "", "")
    op0.allocate_registers(register_allocator)
    assert op0.rs0.type == riscv.IntRegisterType.infinite_register(1)
    assert op0.rs1.type == riscv.IntRegisterType.infinite_register(0)
    assert op0.rd0.type == riscv.IntRegisterType.infinite_register(1)
    assert op0.rd1.type == riscv.IntRegisterType.infinite_register(0)

    # One register reserved for inout parameter, the allocator should allocate the output
    # to the same register.
    rs0, rs1 = TestOp(
        result_types=[
            riscv.IntRegisterType.unallocated(),
            riscv.IntRegisterType.unallocated(),
        ]
    ).results
    op1 = MyInstructionOp.get(rs0, rs1, "", "a0")
    op1.allocate_registers(register_allocator)
    assert op1.rs0.type == riscv.IntRegisterType.infinite_register(2)
    assert op1.rs1.type == riscv.IntRegisterType.from_name("a0")
    assert op1.rd0.type == riscv.IntRegisterType.infinite_register(2)
    assert op1.rd1.type == riscv.IntRegisterType.from_name("a0")


def test_count_reg_types():
    a0 = riscv.Registers.A0
    a1 = riscv.Registers.A1

    fa0 = riscv.Registers.FA0

    assert reg_types_by_name([a0, a0, a1, fa0, fa0]) == {
        "riscv.reg": {"a0", "a1"},
        "riscv.freg": {"fa0"},
    }


def test_multiple_outputs():
    class AllocatableTestOp(TestOp, riscv.RISCVRegallocOperation):
        pass

    available_registers = RiscvRegisterStack(allow_infinite=True)
    register_allocator = RegisterAllocatorLivenessBlockNaive(available_registers)

    op = AllocatableTestOp(
        result_types=(
            riscv.IntRegisterType.unallocated(),
            riscv.IntRegisterType.unallocated(),
        )
    )

    op.allocate_registers(register_allocator)

    # Check allocated registers are unique
    assert len(op.result_types) == len(set(op.result_types))
