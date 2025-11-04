from dataclasses import dataclass

from ordered_set import OrderedSet

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, x86, x86_func
from xdsl.dialects.x86.registers import (
    R12,
    R13,
    R14,
    R15,
    RBP,
    RBX,
    RSP,
    GeneralRegisterType,
)
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint

X86_CALLEE_SAVED_REGISTERS = [RBX, RBP, R12, R13, R14, R15]
"""
Registers that should be the same after the called function returns to the caller, see
[external documentation](https://refspecs.linuxbase.org/elf/x86_64-abi-0.21.pdf).
"""


@dataclass(frozen=True)
class X86PrologueEpilogueInsertion(ModulePass):
    name = "x86-prologue-epilogue-insertion"

    def _process_function(self, func: x86_func.FuncOp) -> None:
        used_callee_preserved_registers = OrderedSet(
            res.type
            for op in func.walk()
            if not isinstance(op, x86.GetRegisterOp)
            for res in op.results
            if isinstance(res.type, GeneralRegisterType)
            if res.type in X86_CALLEE_SAVED_REGISTERS
        )

        if not used_callee_preserved_registers:
            return

        builder = Builder(InsertPoint.at_start(func.body.blocks[0]))
        sp_register = builder.insert(x86.GetRegisterOp(RSP))

        # Build the prologue at the beginning of the function.
        for reg in used_callee_preserved_registers:
            reg_op = builder.insert(x86.GetRegisterOp(reg))
            builder.insert(x86.S_PushOp(rsp_in=sp_register, source=reg_op))

        # Now build the epilogue right before every return operation.
        for block in func.body.blocks:
            ret_op = block.last_op
            if not isinstance(ret_op, x86_func.RetOp):
                continue
            builder = Builder(InsertPoint.before(ret_op))
            for reg in reversed(used_callee_preserved_registers):
                builder.insert(x86.D_PopOp(rsp_in=sp_register, destination=reg))

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func in op.walk():
            if not isinstance(func, x86_func.FuncOp):
                continue

            if not func.body.blocks:
                continue

            self._process_function(func)
