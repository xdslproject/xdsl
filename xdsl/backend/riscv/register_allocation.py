import abc
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import cast

from ordered_set import OrderedSet

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_queue import RegisterQueue
from xdsl.dialects import riscv, riscv_func, riscv_scf, riscv_snitch
from xdsl.dialects.riscv import Registers, RISCVAsmOperation, RISCVRegisterType
from xdsl.ir import Attribute, Block, Operation, SSAValue
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value
from xdsl.utils.exceptions import DiagnosticException


class RegisterAllocator(abc.ABC):
    """
    Base class for register allocation strategies.
    """

    @abc.abstractmethod
    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        raise NotImplementedError()


def reg_types_by_name(regs: Iterable[RISCVRegisterType]) -> dict[str, set[str]]:
    """
    Groups register types by name.
    """
    res = defaultdict[str, set[str]](set)
    for reg in regs:
        res[reg.name].add(reg.register_name.data)
    return res


class RegisterAllocatorLivenessBlockNaive(RegisterAllocator):
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

    available_registers: RegisterQueue
    live_ins_per_block: dict[Block, OrderedSet[SSAValue]]
    new_value_by_old_value: dict[SSAValue, SSAValue]

    def __init__(self, available_registers: RegisterQueue) -> None:
        self.available_registers = available_registers
        self.live_ins_per_block = {}
        self.new_value_by_old_value = {}

    def _replace_value_with_new_type(
        self, val: SSAValue, new_type: Attribute
    ) -> SSAValue:
        new_val = Rewriter.replace_value_with_new_type(val, new_type)
        self.new_value_by_old_value[val] = new_val
        return new_val

    def allocate(self, reg: SSAValue) -> SSAValue | None:
        """
        Allocate a register if not already allocated.
        """
        if reg in self.new_value_by_old_value:
            reg = self.new_value_by_old_value[reg]
        if isinstance(reg.type, RISCVRegisterType) and not reg.type.is_allocated:
            if (val := get_constant_value(reg)) is not None and val.value.data == 0:
                new_reg = self._replace_value_with_new_type(reg, Registers.ZERO)
            else:
                new_reg = self._replace_value_with_new_type(
                    reg, self.available_registers.pop(type(reg.type))
                )
            return new_reg

    def allocate_same(self, vals: Sequence[SSAValue]) -> bool:
        """
        Allocates the values passed in to the same register.
        If some of the values are already allocated, they must be allocated to the same
        register, and unallocated values are then allocated to this register.
        If the values passed in are already allocated to differing registers, a
        `DiagnosticException` is raised.
        """
        reg_types = set(val.type for val in vals)
        assert all(isinstance(reg_type, RISCVRegisterType) for reg_type in reg_types)
        reg_types = cast(set[RISCVRegisterType], reg_types)

        match len(reg_types):
            case 0:
                # No inputs, nothing to do
                return False
            case 1:
                # Single input, may already be allocated
                reg_type = next(iter(reg_types))
                if reg_type.is_allocated:
                    return False
                else:
                    reg_type = self.available_registers.pop(type(reg_type))
            case 2:
                # Two inputs, either one is allocated or two
                reg_type_0, reg_type_1 = reg_types
                if reg_type_0.is_allocated:
                    if reg_type_1.is_allocated:
                        reg_names = [f"{reg_type}" for reg_type in reg_types]
                        reg_names.sort()
                        raise DiagnosticException(
                            f"Cannot allocate registers to the same register {reg_names}"
                        )
                    else:
                        reg_type = reg_type_0
                else:
                    reg_type = reg_type_1
            case _:
                # More than one input is allocated, meaning we can't allocate them to be
                # the same, error.
                reg_names = [f"{reg_type}" for reg_type in reg_types]
                reg_names.sort()
                raise DiagnosticException(
                    f"Cannot allocate registers to the same register {reg_names}"
                )

        did_allocate = False

        for val in vals:
            if val.type != reg_type:
                self._replace_value_with_new_type(val, reg_type)
                did_allocate = True

        return did_allocate

    def _free(self, reg: SSAValue) -> None:
        if reg in self.new_value_by_old_value:
            reg = self.new_value_by_old_value[reg]
        if isinstance(reg.type, RISCVRegisterType) and reg.type.is_allocated:
            self.available_registers.push(reg.type)

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
            self.allocate_same(operand_group)

        for result in outs:
            # Allocate registers to result if not already allocated
            if (new_result := self.allocate(result)) is not None:
                result = new_result
            self._free(result)

        # Allocate registers to operands since they are defined further up
        # in the use-def SSA chain
        for operand in ins:
            self.allocate(operand)

    def allocate_for_loop(self, loop: riscv_scf.ForOp) -> None:
        """
        Allocate registers for riscv_scf for loop, recursively calling process_operation
        for operations in the loop.
        """
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = self.live_ins_per_block[loop.body.block]
        for live_in in live_ins:
            self.allocate(live_in)

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
            self.allocate_same((block_arg, operand, yield_operand, op_result))

        # Induction variable
        self.allocate(block_args[0])

        # Step and ub are used throughout loop
        self.allocate(loop.ub)
        self.allocate(loop.step)

        # Reserve the loop carried variables for allocation within the body
        regs = loop.iter_args.types
        assert all(isinstance(reg, RISCVRegisterType) for reg in regs)
        regs = cast(tuple[RISCVRegisterType, ...], regs)
        with self.available_registers.reserve_registers(regs):
            for op in reversed(loop.body.block.ops):
                self.process_operation(op)

        # lb is only used as an input to the loop, so free induction variable before
        # allocating lb to it in case it's not yet allocated
        self._free(loop.body.block.args[0])
        self.allocate(loop.lb)

    def allocate_frep_loop(self, loop: riscv_snitch.FRepOperation) -> None:
        """
        Allocate registers for riscv_snitch frep_outer or frep_inner loop, recursively
        calling process_operation for operations in the loop.
        """
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = self.live_ins_per_block[loop.body.block]
        for live_in in live_ins:
            self.allocate(live_in)

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
            self.allocate_same((block_arg, operand, yield_operand, op_result))

        self.allocate(loop.max_rep)

        # Reserve the loop carried variables for allocation within the body
        regs = loop.iter_args.types
        assert all(isinstance(reg, RISCVRegisterType) for reg in regs)
        regs = cast(tuple[RISCVRegisterType, ...], regs)
        with self.available_registers.reserve_registers(regs):
            for op in reversed(loop.body.block.ops):
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
        for op in reversed(block.ops):
            self.process_operation(op)

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


def _live_ins_per_block(
    block: Block, acc: dict[Block, OrderedSet[SSAValue]]
) -> OrderedSet[SSAValue]:
    res = OrderedSet[SSAValue]([])

    for op in reversed(block.ops):
        # Remove values defined in the block
        # We are traversing backwards, so cannot use the value removed here again
        res.difference_update(op.results)
        # Add values used in the block
        res.update(op.operands)

        # Process inner blocks
        for region in op.regions:
            for inner in region.blocks:
                # Add the values used in the inner block
                res.update(_live_ins_per_block(inner, acc))

    # Remove the block arguments
    res.difference_update(block.args)

    acc[block] = res

    return res


def live_ins_per_block(block: Block) -> dict[Block, OrderedSet[SSAValue]]:
    """
    Returns a mapping from a block to the set of values used in it but defined outside of
    it.
    """
    res: dict[Block, OrderedSet[SSAValue]] = {}
    _ = _live_ins_per_block(block, res)
    return res
