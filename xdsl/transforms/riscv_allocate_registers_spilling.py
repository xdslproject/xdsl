import typing as t
from dataclasses import dataclass
from itertools import islice

from ordered_set import OrderedSet

from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.context import Context
from xdsl.dialects import builtin, riscv_func
from xdsl.dialects.riscv.registers import IntRegisterType
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class RISCVSpillingPass(ModulePass):
    name = "riscv-allocate-infinite-registers-spilling"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func_op in op.walk():
            if isinstance(func_op, riscv_func.FuncOp):
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
                        regs_to_spill = OrderedSet(
                            islice(free_values, num_regs_to_spill)
                        )

                        self.insert_spill(inner_op, regs_to_spill)
                        loaded_values -= regs_to_spill
                    loaded_values |= uses

                    # Remove dead values from live set
                    loaded_values -= die[inner_op]

                    # Process definitions
                    defns = OrderedSet(inner_op.results)
                    if len(loaded_values | defns) > total_regs:
                        # spill excess regs
                        num_regs_to_spill = len(loaded_values | defns) - total_regs
                        # get registers not used by current op
                        # TODO: use heuristic to select
                        regs_to_spill = OrderedSet(
                            islice(free_values, num_regs_to_spill)
                        )

                        self.insert_spill(inner_op, regs_to_spill)
                        loaded_values -= regs_to_spill
                    loaded_values |= defns

    def insert_spill(self, inner_op: Operation, regs_to_spill: t.Iterable[SSAValue]):
        pass

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
