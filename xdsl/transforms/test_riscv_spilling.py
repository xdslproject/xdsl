from dataclasses import dataclass
from itertools import islice

from ordered_set import OrderedSet

from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_func
from xdsl.dialects.riscv.registers import IntRegisterType, Registers
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


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

        builder = Builder(InsertPoint.before(inner_op))
        for spill_val in spills:
            alloca_op = riscv.stack.AllocaOp(builtin.i32)
            spill_op = riscv.stack.StoreOp(alloca_op, spill_val)
            builder.insert_op(alloca_op)
            builder.insert_op(spill_op)

            # replace all uses after spill to the ref
            spill_val.replace_uses_with_if(
                alloca_op.ref, lambda use: spill_op.is_before_in_block(use.operation)
            )

    def insert_load(
        self, inner_op: Operation, loads: OrderedSet[SSAValue]
    ) -> OrderedSet[SSAValue]:
        """Insert loads before inner_op."""
        if not loads:
            return OrderedSet([])

        assert isa(loads, OrderedSet[SSAValue[riscv.stack.StackSlotType]])

        load_results = OrderedSet[SSAValue]([])
        builder = Builder(InsertPoint.before(inner_op))

        for load_val in loads:
            load_op = riscv.stack.LoadOp(load_val)
            builder.insert_op(load_op)
            # replace all uses of the ref after load to the loaded result
            load_val.replace_uses_with_if(
                load_op.rd,
                lambda use: load_op.is_before_in_block(use.operation),
            )
            load_results.add(load_op.rd)

        return load_results

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


@dataclass(frozen=True)
class TestRiscvSpillingPass(ModulePass):
    name = "test-riscv-spilling"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        defaults = RiscvRegisterStack.DEFAULT_ALLOCATABLE_REGISTERS
        # Use reduced register set for testing
        RiscvRegisterStack.DEFAULT_ALLOCATABLE_REGISTERS = (
            Registers.T0,
            Registers.T1,
            Registers.T2,
        )
        SpillPass().apply(ctx, op)

        RiscvRegisterStack.DEFAULT_ALLOCATABLE_REGISTERS = defaults
