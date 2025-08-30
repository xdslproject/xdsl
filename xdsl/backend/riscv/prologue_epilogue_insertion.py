from dataclasses import dataclass, field

from ordered_set import OrderedSet

from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_func
from xdsl.dialects.riscv import (
    FloatRegisterType,
    IntRegisterType,
    Registers,
    RISCVRegisterType,
)
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class PrologueEpilogueInsertion(ModulePass):
    """
    Pass inserting a prologue and epilogue according to the RISC-V ABI.
    The prologues and epilogues are responsible for saving any callee-preserved
    registers.
    In RISC-V these are 's0' to 's11' and 'fs0' to `fs11'.
    The stack pointer 'sp' must also be restored to its original value.

    This pass should be run late in the pipeline after register allocation.
    It does not itself require register allocation nor invalidate the result of the
    register allocator.
    """

    name = "riscv-prologue-epilogue-insertion"
    xlen: int = field(default=4)
    flen: int = field(default=8)

    def _process_function(self, func: riscv_func.FuncOp) -> None:
        # Find all callee-preserved registers that are clobbered. We define clobbered
        # as it being the result of some operation and therefore written to.
        used_callee_preserved_registers = OrderedSet(
            res.type
            for op in func.walk()
            if not isinstance(op, riscv.GetRegisterOp | riscv.GetFloatRegisterOp)
            for res in op.results
            if isinstance(res.type, IntRegisterType | FloatRegisterType)
            if res.type in Registers.S or res.type in Registers.FS
        )

        if not used_callee_preserved_registers:
            return

        def get_register_size(r: RISCVRegisterType):
            if isinstance(r, IntRegisterType):
                return self.xlen
            return self.flen

        # Build the prologue at the beginning of the function.
        builder = Builder(InsertPoint.at_start(func.body.blocks[0]))
        sp_register = builder.insert_op(riscv.GetRegisterOp(Registers.SP))
        stack_size = sum(get_register_size(r) for r in used_callee_preserved_registers)
        builder.insert_op(riscv.AddiOp(sp_register, -stack_size, rd=Registers.SP))
        offset = 0
        for reg in used_callee_preserved_registers:
            if isinstance(reg, IntRegisterType):
                reg_op = builder.insert_op(riscv.GetRegisterOp(reg))
                op = riscv.SwOp(rs1=sp_register, rs2=reg_op, immediate=offset)
            else:
                reg_op = builder.insert_op(riscv.GetFloatRegisterOp(reg))
                op = riscv.FSdOp(rs1=sp_register, rs2=reg_op, immediate=offset)

            builder.insert_op(op)
            offset += get_register_size(reg)

        # Now build the epilogue right before every return operation.
        for block in func.body.blocks:
            ret_op = block.last_op
            if not isinstance(ret_op, riscv_func.ReturnOp):
                continue

            builder = Builder(InsertPoint.before(ret_op))
            offset = 0
            for reg in used_callee_preserved_registers:
                if isinstance(reg, IntRegisterType):
                    op = riscv.LwOp(rs1=sp_register, rd=reg, immediate=offset)
                else:
                    op = riscv.FLdOp(rs1=sp_register, rd=reg, immediate=offset)
                builder.insert_op(op)
                offset += get_register_size(reg)

            builder.insert_op(riscv.AddiOp(sp_register, stack_size, rd=Registers.SP))

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in op.walk():
            if not isinstance(func, riscv_func.FuncOp):
                continue

            if len(func.body.blocks) == 0:
                continue

            self._process_function(func)
