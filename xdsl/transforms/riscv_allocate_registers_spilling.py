from dataclasses import dataclass
from itertools import islice

from ordered_set import OrderedSet

from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.context import Context
from xdsl.dialects import builtin, riscv_func
from xdsl.dialects.riscv.ops import AddiOp, LwOp, SwOp
from xdsl.dialects.riscv.registers import IntRegisterType, Registers
from xdsl.dialects.rv32 import GetRegisterOp
from xdsl.ir import Operation, ParametrizedAttribute, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter


@irdl_attr_definition
class SpillType(ParametrizedAttribute):
    name = "spilling.type"


@irdl_op_definition
class SpillOp(IRDLOperation):
    name = "spilling.spill_op"
    value = operand_def()
    result = result_def(SpillType())

    def __init__(self, val):
        super().__init__(operands=[val], result_types=[SpillType()])


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "spilling.load_op"
    value = operand_def(SpillType())
    result = result_def()

    def __init__(self, val):
        super().__init__(operands=[val], result_types=[Registers.UNALLOCATED_INT])


@dataclass(frozen=True)
class SpillPass(ModulePass):
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func_op in op.walk():
            if not isinstance(func_op, riscv_func.FuncOp):
                continue
            # Assume for now we only have integer regs
            total_regs = len(
                RiscvRegisterStack.get().available_registers[IntRegisterType.name]
            )
            loaded_values = OrderedSet[SSAValue]([])

            die = self.get_die_set(func_op)

            for inner_op in func_op.walk():
                uses = OrderedSet(inner_op.operands)
                free_values = iter(
                    loaded_values - uses
                )  # values that we can use to spill

                # Process uses
                if len(loaded_values | uses) > total_regs:
                    # spill excess regs
                    num_regs_to_spill = len(uses | loaded_values) - total_regs
                    # get registers not used by current op
                    # TODO: use heuristic to select
                    regs_to_spill = OrderedSet(islice(free_values, num_regs_to_spill))

                    self.insert_spill(inner_op, regs_to_spill)
                    loaded_values -= regs_to_spill
                loaded_uses = self.insert_load(inner_op, uses - loaded_values)
                loaded_values |= loaded_uses

                # Remove dead values from live set
                loaded_values -= die[inner_op]

                # Process definitions
                defns = OrderedSet(inner_op.results)
                if len(loaded_values | defns) > total_regs:
                    # spill excess regs
                    num_regs_to_spill = len(loaded_values | defns) - total_regs
                    # get registers not used by current op
                    # TODO: use heuristic to select
                    regs_to_spill = OrderedSet(islice(free_values, num_regs_to_spill))

                    self.insert_spill(inner_op, regs_to_spill)
                    loaded_values -= regs_to_spill
                loaded_values |= defns

    def insert_spill(self, inner_op: Operation, spills: OrderedSet[SSAValue]):
        """Insert spills before inner_op."""
        if not spills:
            return

        spill_ops = tuple(SpillOp(spill_val) for spill_val in spills)
        for spill_op in spill_ops:
            Rewriter.insert_op(spill_op, InsertPoint.before(inner_op))
        # change all uses after pmov_op to results of pmov
        for src, spill_op in zip(spills, spill_ops):
            dst = spill_op.result
            src.replace_uses_with_if(
                dst, lambda use: spill_op.is_before_in_block(use.operation)
            )

    def insert_load(
        self, inner_op: Operation, loads: OrderedSet[SSAValue]
    ) -> OrderedSet[SSAValue]:
        """Insert loads before inner_op."""
        if not loads:
            return OrderedSet([])

        load_ops = tuple(LoadOp(load_val) for load_val in loads)
        for load_op in load_ops:
            Rewriter.insert_op(load_op, InsertPoint.before(inner_op))
        # change all uses after pmov_op to results of pmov
        for src, load_op in zip(loads, load_ops):
            dst = load_op.result
            src.replace_uses_with_if(
                dst, lambda use: load_op.is_before_in_block(use.operation)
            )
        return OrderedSet(i.result for i in load_ops)

    def get_die_set(self, func_op: riscv_func.FuncOp):
        """Create set of values that die at each operation."""
        die: dict[Operation, set[SSAValue]] = {}
        seen_vals = set[SSAValue]()
        for inner_op in func_op.walk(reverse=True):
            # by SSA, we can just check if this is the first seen use
            uses = set(inner_op.operands)
            die[inner_op] = uses - seen_vals
            seen_vals |= uses
        return die


class ResolveSpillingOps(ModulePass):
    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func_op in op.walk():
            if not isinstance(func_op, riscv_func.FuncOp):
                continue
            spill_ops = tuple(op for op in func_op.walk() if isinstance(op, SpillOp))
            load_ops = tuple(op for op in func_op.walk() if isinstance(op, LoadOp))

            # Map each spill value with its own stack offset
            offset_by_value = {}
            offset = 0
            for spill_op in spill_ops:
                offset_by_value[spill_op.result] = offset
                offset += 4  # assume 32 bit values for now

            # Get stack pointer for Lw/Sw to reference
            stack_pointer = GetRegisterOp(Registers.SP)
            Rewriter.insert_op(stack_pointer, InsertPoint.at_start(func_op.body.block))

            # Resolve loads
            for load_op in load_ops:
                Rewriter.replace_op(
                    load_op,
                    LwOp(stack_pointer, offset_by_value[load_op.value]),
                )

            # Resolve spills
            for spill_op in spill_ops:
                store_op = SwOp(
                    stack_pointer, spill_op.value, offset_by_value[spill_op.result]
                )
                # Use insert/erase since we are dropping the spill_op's result
                Rewriter.insert_op(store_op, InsertPoint.before(spill_op))
                Rewriter.erase_op(spill_op)

            # Manage stack pointer
            # Align stack to 16 bytes
            stack_size = (offset + 15) & ~15
            if stack_size > 0:
                Rewriter.insert_op(
                    AddiOp(stack_pointer, -stack_size, rd=Registers.SP),
                    InsertPoint.after(stack_pointer),
                )

                for block in func_op.body.blocks:
                    ret_op = block.last_op
                    if not isinstance(ret_op, riscv_func.ReturnOp):
                        continue

                    Rewriter.insert_op(
                        AddiOp(stack_pointer, stack_size, rd=Registers.SP),
                        InsertPoint.before(ret_op),
                    )
