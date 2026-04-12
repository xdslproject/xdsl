from dataclasses import dataclass, field

from ordered_set import OrderedSet

from xdsl.builder import Builder, InsertPoint
from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_func, rv32
from xdsl.dialects.riscv import (
    FloatRegisterType,
    IntRegisterType,
    Registers,
)
from xdsl.dialects.riscv.stack import AllocaOp, LoadOp, StoreOp
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
            if not isinstance(op, rv32.GetRegisterOp | riscv.GetFloatRegisterOp)
            for res in op.results
            if isinstance(res.type, IntRegisterType | FloatRegisterType)
            if res.type in Registers.S or res.type in Registers.FS
        )

        if not used_callee_preserved_registers:
            return

        # Build the prologue at the beginning of the function.
        builder = Builder(InsertPoint.at_start(func.body.blocks[0]))
        memrefs = []
        for reg in used_callee_preserved_registers:
            if isinstance(reg, IntRegisterType):
                alloca_op = builder.insert(AllocaOp(builtin.i32))
                reg_op = builder.insert(rv32.GetRegisterOp(reg))
            else:
                match self.flen:
                    case 8:
                        float_size = builtin.f64
                    case 4:
                        float_size = builtin.f32
                    case _:
                        raise NotImplementedError("flen must be 4 or 8")

                alloca_op = builder.insert(AllocaOp(float_size))
                reg_op = builder.insert(riscv.GetFloatRegisterOp(reg))
            builder.insert(StoreOp(alloca_op, reg_op))
            memrefs.append(alloca_op)

        # Now build the epilogue right before every return operation.
        for block in func.body.blocks:
            ret_op = block.last_op
            if not isinstance(ret_op, riscv_func.ReturnOp):
                continue

            builder = Builder(InsertPoint.before(ret_op))
            for memref, reg in zip(memrefs, used_callee_preserved_registers):
                builder.insert(LoadOp(memref, rd=reg))

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in op.walk():
            if not isinstance(func, riscv_func.FuncOp):
                continue

            if len(func.body.blocks) == 0:
                continue

            self._process_function(func)
