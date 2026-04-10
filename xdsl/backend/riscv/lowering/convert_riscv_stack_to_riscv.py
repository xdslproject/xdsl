from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import riscv_func, rv32
from xdsl.dialects.builtin import IntegerType, ModuleOp
from xdsl.dialects.riscv.ops import AddiOp, LwOp, SwOp
from xdsl.dialects.riscv.registers import IntRegisterType, Registers
from xdsl.dialects.riscv.stack import AllocaOp, LoadOp, StoreOp
from xdsl.ir import Attribute, Use
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.utils.exceptions import PassFailedException


def get_type_size(value_type: Attribute):
    return 4


class ConvertRiscvStackToRiscvPass(ModulePass):
    name = "convert-riscv-stack-to-riscv"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for func_op in op.walk():
            if not isinstance(func_op, riscv_func.FuncOp):
                continue

            cur_offset = 0
            # Need to instantiate generator so erasing op doesn't stop iteration
            alloca_ops = tuple(op for op in func_op.walk() if isinstance(op, AllocaOp))
            # For each function, give each alloca its stack offset
            for alloca_op in alloca_ops:
                stack_slot = alloca_op.ref.type
                assert isinstance(stack_slot.value_type, IntegerType)
                value_size = get_type_size(stack_slot.value_type)

                for use in alloca_op.ref.uses:
                    self.rewrite_stack_slot_use(use, cur_offset)

                Rewriter.erase_op(alloca_op)
                cur_offset += value_size

            # Adjust stack pointer
            self.insert_prologue_epilogue(cur_offset, func_op)

    def rewrite_stack_slot_use(self, use: Use, stack_offset: int):
        if isinstance(use.operation, StoreOp):
            Rewriter.replace_op(
                use.operation,
                (
                    stack_ptr := rv32.GetRegisterOp(Registers.SP),
                    SwOp(stack_ptr, use.operation.rs, stack_offset),
                ),
            )
        elif isinstance(use.operation, LoadOp):
            assert isinstance(use.operation.rd.type, IntRegisterType)
            Rewriter.replace_op(
                use.operation,
                (
                    stack_ptr := rv32.GetRegisterOp(Registers.SP),
                    LwOp(stack_ptr, stack_offset, rd=use.operation.rd.type),
                ),
            )
        else:
            raise PassFailedException("Invalid use of StackSlotType: ", use.operation)

    def insert_prologue_epilogue(self, total_offset: int, func_op: riscv_func.FuncOp):
        # Align SP to 16 bytes (from RISC-V calling convention)
        total_offset = (total_offset + 15) & ~15

        if total_offset > 0:
            # prologue
            builder = Builder(InsertPoint.at_start(func_op.body.blocks[0]))
            builder.insert(stack_ptr := rv32.GetRegisterOp(Registers.SP))
            builder.insert(AddiOp(stack_ptr, -total_offset, rd=Registers.SP))
            # epilogue
            # using logic from prologue_epilogue_insertion.py
            for block in func_op.body.blocks:
                ret_op = block.last_op
                if not isinstance(ret_op, riscv_func.ReturnOp):
                    continue

                builder = Builder(InsertPoint.before(ret_op))
                builder.insert(AddiOp(stack_ptr, total_offset, rd=Registers.SP))
