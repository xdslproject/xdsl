from typing import Annotated, TypeAlias

import pytest

from xdsl.backend.riscv.register_allocation import RegisterAllocatorLivenessBlockNaive
from xdsl.builder import Builder
from xdsl.dialects import builtin, riscv, riscv_func
from xdsl.dialects.riscv import (
    AssemblyInstructionArg,
    IntRegisterType,
    RISCVInstruction,
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


# We use the infamous post-increment load instruction
# provided by several architectures (ARM, PULP, ...) that presents
# the classical form of a 2-address instruction: the first operand
# is incremented and returned in a brand new SSA value as the second result.
# Both tied SSA values must be allocated on the same register, otherwise the
# generated code is broken.
@irdl_op_definition
class PostIncrementLoad(RISCVInstruction):
    name = "postload"

    SameIntRegisterType: TypeAlias = Annotated[IntRegisterType, ConstraintVar("T")]

    rd: OpResult = result_def(IntRegisterType)  # loaded value
    off: OpResult = result_def(
        SameIntRegisterType
    )  # new value of the incremented offset
    rs1: Operand = operand_def(SameIntRegisterType)  # offset
    rs2: Operand = operand_def(IntRegisterType)  # base adddress

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg, ...]:
        return self.rd, self.rs1, self.rs2


def test_allocate_2address():
    # riscv_func.func @function() {
    #     %0 = riscv.li 1 : !riscv.reg<t1>
    #     %1 = riscv.li 2 : !riscv.reg<t0>
    #        tied -------- tied                  tied ---------------------------------------------- tied
    #         |             |                     |                                                   |
    #     %2, %3 = postload %0, %1 : (!riscv.reg<t1>, !riscv.reg<t0>) -> (!riscv.reg<t0>, !riscv.reg<t1>)
    #
    #                this add forces all results of the postload to be allocated first
    #                 |
    #     %4 = riscv.add %2, %3 : (!riscv.reg<t0>, !riscv.reg<t1>) -> !riscv.reg<t0>
    #     %5 = builtin.unrealized_conversion_cast %4 : !riscv.reg<t0> to i32
    #     riscv_func.return
    # }
    @Builder.implicit_region
    def region():
        base = riscv.LiOp(1)
        offset = riscv.LiOp(2)
        load = PostIncrementLoad(
            operands=[base.rd, offset.rd],
            result_types=[
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            ],
        )
        add = riscv.AddOp(load.rd, load.off, rd=IntRegisterType.unallocated())
        builtin.UnrealizedConversionCastOp.get((add.rd,), (builtin.i32,))
        riscv_func.ReturnOp()

    func = riscv_func.FuncOp("function", region, ((), ()))
    func.verify()

    allocator = RegisterAllocatorLivenessBlockNaive()
    allocator.allocate_func(func)
    func.verify()

    body = list(func.body.block.ops)
    # load.rs1 == load.off
    assert body[2].operands[0].type == body[2].results[1].type
    # load.rs1 != load.rd
    assert body[2].operands[0].type != body[2].results[0].type
    # load.rs1 != load.rs2
    assert body[2].operands[0].type != body[2].operands[1].type


def test_allocate_2address_no_riscv_successor():
    # riscv_func.func @function() {
    #     %0 = riscv.li 1 : !riscv.reg<t1>
    #     %1 = riscv.li 2 : !riscv.reg<t2>
    #        tied -------- tied                  tied ---------------------------------------------- tied
    #         |             |                     |                                                   |
    #     %2, %3 = postload %0, %1 : (!riscv.reg<t1>, !riscv.reg<t2>) -> (!riscv.reg<t0>, !riscv.reg<t1>)
    #     %4 = builtin.unrealized_conversion_cast %2 : !riscv.reg<t0> to i32
    #     %5 = builtin.unrealized_conversion_cast %3 : !riscv.reg<t1> to i32
    #     riscv_func.return
    # }
    @Builder.implicit_region
    def region():
        base = riscv.LiOp(1)
        offset = riscv.LiOp(2)
        load = PostIncrementLoad(
            operands=[base.rd, offset.rd],
            result_types=[
                IntRegisterType.unallocated(),
                IntRegisterType.unallocated(),
            ],
        )
        builtin.UnrealizedConversionCastOp.get((load.rd,), (builtin.i32,))
        builtin.UnrealizedConversionCastOp.get((load.off,), (builtin.i32,))
        riscv_func.ReturnOp()

    func = riscv_func.FuncOp("function", region, ((), ()))
    func.verify()

    allocator = RegisterAllocatorLivenessBlockNaive()
    allocator.allocate_func(func)
    func.verify()

    body = list(func.body.block.ops)
    # load.rs1 == load.off
    assert body[2].operands[0].type == body[2].results[1].type
    # load.rs1 != load.rd
    assert body[2].operands[0].type != body[2].results[0].type
    # load.rs1 != load.rs2
    assert body[2].operands[0].type != body[2].operands[1].type


def test_allocate_2address_preallocated():
    @Builder.implicit_region
    def region():
        base = riscv.LiOp(1)
        offset = riscv.LiOp(2)
        load = PostIncrementLoad(
            operands=[base.rd, offset.rd],
            result_types=[
                IntRegisterType.unallocated(),
                riscv.Registers.A7,
            ],
        )
        builtin.UnrealizedConversionCastOp.get((load.rd,), (builtin.i32,))
        builtin.UnrealizedConversionCastOp.get((load.off,), (builtin.i32,))
        riscv_func.ReturnOp()

    func = riscv_func.FuncOp("function", region, ((), ()))
    with pytest.raises(VerifyException):
        func.verify()

    allocator = RegisterAllocatorLivenessBlockNaive()
    allocator.exclude_preallocated = True
    allocator.allocate_func(func)
    func.verify()

    body = list(func.body.block.ops)
    # load.rs1 == load.off
    assert body[2].operands[0].type == riscv.Registers.A7
    assert body[2].results[1].type == riscv.Registers.A7
    # load.rs1 != load.rd
    assert body[2].operands[0].type != body[2].results[0].type
    # load.rs1 != load.rs2
    assert body[2].operands[0].type != body[2].operands[1].type
