from abc import ABC
from typing import Annotated, Generic, TypeAlias, TypeVar

import pytest

from xdsl.backend.riscv.register_allocation_constraints import OpTieConstraints
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    IntRegisterType,
    Registers,
    RISCVInstruction,
    RISCVRegisterType,
)
from xdsl.ir import OpResult
from xdsl.irdl import (
    ConstraintVar,
    Operand,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.test_value import TestSSAValue

RD1InvT = TypeVar("RD1InvT", bound=RISCVRegisterType)
RD2InvT = TypeVar("RD2InvT", bound=RISCVRegisterType)
RS1InvT = TypeVar("RS1InvT", bound=RISCVRegisterType)
RS2InvT = TypeVar("RS2InvT", bound=RISCVRegisterType)


class TestOp4(Generic[RD1InvT, RD2InvT, RS1InvT, RS2InvT], RISCVInstruction, ABC):

    rd1: OpResult = result_def(RD1InvT)
    rd2: OpResult = result_def(RD2InvT)
    rs1: Operand = operand_def(RS1InvT)
    rs2: Operand = operand_def(RS2InvT)

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd1, self.rd2, self.rs1, self.rs2


SameIntRegisterType: TypeAlias = Annotated[IntRegisterType, ConstraintVar("T")]


def test_constrained_single():
    @irdl_op_definition
    class ConstrainedOp(
        TestOp4[SameIntRegisterType, IntRegisterType, IntRegisterType, IntRegisterType]
    ):
        name = "constrained_op"

    op = ConstrainedOp(
        result_types=[
            IntRegisterType.unallocated(),  # rd1
            IntRegisterType.unallocated(),  # rd2
        ],
        operands=[
            TestSSAValue(IntRegisterType.unallocated()),  # rs1
            TestSSAValue(IntRegisterType.unallocated()),  # rs2
        ],
    )
    op.verify()

    constr = OpTieConstraints.from_op(op)
    assert constr.result_has_constraints(0)
    assert not constr.result_has_constraints(1)
    assert not constr.operand_has_constraints(0)
    assert not constr.operand_has_constraints(1)

    assert constr.result_is_constrained_to(0) is None
    assert constr.result_is_constrained_to(1) is None
    assert constr.operand_is_constrained_to(2) is None
    assert constr.operand_is_constrained_to(3) is None

    # Allocate rd1
    op.rd1.type = Registers.A1
    constr.result_satisfy_constraint(0, Registers.A1)

    assert constr.result_is_constrained_to(0) == Registers.A1
    assert constr.result_is_constrained_to(1) is None
    assert constr.operand_is_constrained_to(2) is None
    assert constr.operand_is_constrained_to(3) is None


def test_constrained_all_unallocated():
    @irdl_op_definition
    class ConstrainedOp(
        TestOp4[
            SameIntRegisterType,
            SameIntRegisterType,
            SameIntRegisterType,
            SameIntRegisterType,
        ]
    ):
        name = "constrained_op"

    op = ConstrainedOp(
        result_types=[
            IntRegisterType.unallocated(),  # rd1
            IntRegisterType.unallocated(),  # rd2
        ],
        operands=[
            TestSSAValue(IntRegisterType.unallocated()),  # rs1
            TestSSAValue(IntRegisterType.unallocated()),  # rs2
        ],
    )
    op.verify()

    constr = OpTieConstraints.from_op(op)
    assert constr.result_has_constraints(0)
    assert constr.result_has_constraints(1)
    assert constr.operand_has_constraints(0)
    assert constr.operand_has_constraints(1)

    assert constr.result_is_constrained_to(0) is None
    assert constr.result_is_constrained_to(1) is None
    assert constr.operand_is_constrained_to(0) is None
    assert constr.operand_is_constrained_to(1) is None

    # Allocate rs2
    op.rs2.type = Registers.A1
    constr.operand_satisfy_constraint(1, Registers.A1)

    assert constr.result_is_constrained_to(0) == Registers.A1
    assert constr.result_is_constrained_to(1) == Registers.A1
    assert constr.operand_is_constrained_to(0) == Registers.A1
    assert constr.operand_is_constrained_to(1) == Registers.A1


def test_constrained_all_allocated():
    @irdl_op_definition
    class ConstrainedOp(
        TestOp4[
            SameIntRegisterType,
            SameIntRegisterType,
            SameIntRegisterType,
            SameIntRegisterType,
        ]
    ):
        name = "constrained_op"

    op = ConstrainedOp(
        result_types=[
            IntRegisterType.unallocated(),  # rd1
            IntRegisterType.unallocated(),  # rd2
        ],
        operands=[
            TestSSAValue(Registers.A1),  # rs1
            TestSSAValue(IntRegisterType.unallocated()),  # rs2
        ],
    )
    with pytest.raises(VerifyException):
        op.verify()

    constr = OpTieConstraints.from_op(op)
    assert constr.result_has_constraints(0)
    assert constr.result_has_constraints(1)
    assert constr.operand_has_constraints(0)
    assert constr.operand_has_constraints(1)

    assert constr.result_is_constrained_to(0) == Registers.A1
    assert constr.result_is_constrained_to(1) == Registers.A1
    assert constr.operand_is_constrained_to(0) == Registers.A1
    assert constr.operand_is_constrained_to(1) == Registers.A1

    op.rd1.type = Registers.A1
    op.rd2.type = Registers.A1
    op.rs1.type = Registers.A1
    op.rs2.type = Registers.A1
    op.verify()

    op.rs2.type = Registers.A2
    with pytest.raises(VerifyException):
        op.verify()
    with pytest.raises(ValueError):
        constr.operand_satisfy_constraint(0, Registers.A2)
