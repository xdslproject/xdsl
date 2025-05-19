import json
from collections import defaultdict
from collections.abc import Iterable
from typing import cast

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_allocator import BlockAllocator, live_ins_per_block
from xdsl.backend.register_queue import RegisterQueue
from xdsl.backend.register_type import RegisterType
from xdsl.dialects import riscv, riscv_func, riscv_scf, riscv_snitch
from xdsl.dialects.riscv import Registers, RISCVAsmOperation, RISCVRegisterType
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value


def reg_types_by_name(regs: Iterable[RISCVRegisterType]) -> dict[str, set[str]]:
    """
    Groups register types by name.
    """
    res = defaultdict[str, set[str]](set)
    for reg in regs:
        res[reg.name].add(reg.register_name.data)
    return res


class RegisterAllocatorLivenessBlockNaive(BlockAllocator):
    """
    It traverses the use-def SSA chain backwards (i.e., from uses to defs) and:
      1. allocates registers for operands
      2. frees registers for results (since that will be the last time they appear when
      going backwards)
    It currently operates on blocks.

    This is a simplified version of the standard bottom-up local register allocation
    algorithm.

    A relevant reference in "Engineering a Compiler, Edition 3" ISBN: 9780128154120.

    ```
    for op in block.walk_reverse():
    for operand in op.operands:
        if operand is not allocated:
            allocate(operand)

    for result in op.results:
    if result is not allocated:
        allocate(result)
        free_before_next_instruction.append(result)
    else:
        free(result)
    ```
    """

    def __init__(self, available_registers: RegisterQueue) -> None:
        super().__init__(available_registers, RISCVRegisterType)

    def new_type_for_value(self, reg: SSAValue) -> RegisterType | None:
        if (
            isinstance(reg.type, self.register_base_class)
            and not reg.type.is_allocated
            and (val := get_constant_value(reg)) is not None
            and val.value.data == 0
        ):
            return Registers.ZERO
        return super().new_type_for_value(reg)

    def process_operation(self, op: Operation) -> None:
        """
        Allocate registers for one operation.
        """
        match op:
            case riscv_scf.ForOp():
                self.allocate_for_loop(op)
            case riscv_snitch.FRepOperation():
                self.allocate_frep_loop(op)
            case RISCVAsmOperation():
                self.process_riscv_op(op)
            case _:
                # Ignore non-riscv operations
                return

    def process_riscv_op(self, op: RISCVAsmOperation) -> None:
        """
        Allocate registers for RISC-V Instruction.
        """
        ins, outs, inouts = op.get_register_constraints()

        # Allocate registers to inout operand groups since they are defined further up
        # in the use-def SSA chain
        for operand_group in inouts:
            self.allocate_values_same_reg(operand_group)

        for result in outs:
            # Allocate registers to result if not already allocated
            if (new_result := self.allocate_value(result)) is not None:
                result = new_result
            self.free_value(result)

        # Allocate registers to operands since they are defined further up
        # in the use-def SSA chain
        for operand in ins:
            self.allocate_value(operand)

    def allocate_for_loop(self, loop: riscv_scf.ForOp) -> None:
        """
        Allocate registers for riscv_scf for loop, recursively calling process_operation
        for operations in the loop.
        """
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = self.live_ins_per_block[loop.body.block]
        for live_in in live_ins:
            self.allocate_value(live_in)

        yield_op = loop.body.block.last_op
        assert yield_op is not None, (
            "last op of riscv_scf.ForOp is guaranteed to be riscv_scf.Yield"
        )
        block_args = loop.body.block.args

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for block_arg, operand, yield_operand, op_result in zip(
            block_args[1:], loop.iter_args, yield_op.operands, loop.results
        ):
            self.allocate_values_same_reg(
                (block_arg, operand, yield_operand, op_result)
            )

        # Induction variable
        self.allocate_value(block_args[0])

        # Step and ub are used throughout loop
        self.allocate_value(loop.ub)
        self.allocate_value(loop.step)

        # Reserve the loop carried variables for allocation within the body
        regs = loop.iter_args.types
        assert all(isinstance(reg, RISCVRegisterType) for reg in regs)
        regs = cast(tuple[RISCVRegisterType, ...], regs)
        with self.available_registers.reserve_registers(regs):
            self.allocate_block(loop.body.block)

        # lb is only used as an input to the loop, so free induction variable before
        # allocating lb to it in case it's not yet allocated
        self.free_value(loop.body.block.args[0])
        self.allocate_value(loop.lb)

    def allocate_frep_loop(self, loop: riscv_snitch.FRepOperation) -> None:
        """
        Allocate registers for riscv_snitch frep_outer or frep_inner loop, recursively
        calling process_operation for operations in the loop.
        """
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = self.live_ins_per_block[loop.body.block]
        for live_in in live_ins:
            self.allocate_value(live_in)

        yield_op = loop.body.block.last_op
        assert yield_op is not None, (
            "last op of riscv_snitch.frep_outer and riscv_snitch.frep_inner is guaranteed"
            " to be riscv_scf.Yield"
        )
        block_args = loop.body.block.args

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for block_arg, operand, yield_operand, op_result in zip(
            block_args, loop.iter_args, yield_op.operands, loop.results
        ):
            self.allocate_values_same_reg(
                (block_arg, operand, yield_operand, op_result)
            )

        self.allocate_value(loop.max_rep)

        # Reserve the loop carried variables for allocation within the body
        regs = loop.iter_args.types
        assert all(isinstance(reg, RISCVRegisterType) for reg in regs)
        regs = cast(tuple[RISCVRegisterType, ...], regs)
        with self.available_registers.reserve_registers(regs):
            self.allocate_block(loop.body.block)

    def allocate_block(self, block: Block):
        for op in reversed(block.ops):
            self.process_operation(op)

    def allocate_func(
        self, func: riscv_func.FuncOp, *, add_regalloc_stats: bool = False
    ) -> None:
        """
        Allocates values in function passed in to registers.
        The whole function must have been lowered to the relevant riscv dialects
        and it must contain no unrealized casts.
        If `add_regalloc_stats` is set to `True`, then a comment op will be inserted
        before the function op passed in with a json containing the relevant data.
        """
        if not func.body.blocks:
            # External function declaration
            return

        if len(func.body.blocks) != 1:
            raise NotImplementedError(
                f"Cannot register allocate func with {len(func.body.blocks)} blocks."
            )

        preallocated = {
            reg
            for reg in RegisterAllocatableOperation.iter_all_used_registers(func.body)
            if isinstance(reg, RISCVRegisterType)
        }

        for pa_reg in preallocated:
            self.available_registers.reserve_register(pa_reg)
            self.available_registers.exclude_register(pa_reg)

        block = func.body.block

        self.live_ins_per_block = live_ins_per_block(block)
        assert not self.live_ins_per_block[block]

        self.allocate_block(block)

        if add_regalloc_stats:
            preallocated_stats = reg_types_by_name(preallocated)
            allocated_stats = reg_types_by_name(
                val.type
                for op in block.walk()
                for vals in (op.results, op.operands)
                for val in vals
                if isinstance(val.type, RISCVRegisterType)
            )
            stats = {
                "preallocated_float": sorted(preallocated_stats["riscv.freg"]),
                "preallocated_int": sorted(preallocated_stats["riscv.reg"]),
                "allocated_float": sorted(allocated_stats["riscv.freg"]),
                "allocated_int": sorted(allocated_stats["riscv.reg"]),
            }

            stats_str = json.dumps(stats)

            Rewriter.insert_op(
                riscv.CommentOp(f"Regalloc stats: {stats_str}"),
                InsertPoint.before(func),
            )
