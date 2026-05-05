from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from itertools import islice

from ordered_set import OrderedSet

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_func
from xdsl.dialects.riscv.registers import (
    FloatRegisterType,
    IntRegisterType,
    RISCVRegisterType,
)
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
            self.spill_values(func_op, IntRegisterType)
            self.spill_values(func_op, FloatRegisterType)

    def spill_values(
        self, func_op: riscv_func.FuncOp, base_reg_type: type[RISCVRegisterType]
    ):
        used_regs = {
            reg
            for reg in RegisterAllocatableOperation.iter_all_used_registers(
                func_op.body
            )
            if isinstance(reg, RISCVRegisterType)
        }
        total_regs = len(
            {
                i
                for i in RiscvRegisterStack.default_allocatable_registers()
                if i not in used_regs and isinstance(i, base_reg_type)
            }
        )

        loaded_values = OrderedSet[SSAValue]([])

        die = self.get_die_set(func_op)

        for inner_op in func_op.walk():
            if not isinstance(inner_op, RegisterAllocatableOperation):
                continue
            uses = OrderedSet(
                i for i in inner_op.operands if isinstance(i.type, base_reg_type)
            )
            free_values = iter(loaded_values - uses)  # values that we can use to spill

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

            # TODO remove this assert
            assert loaded_values.issuperset(uses)

            # Remove dead values from live set
            for use in uses:
                if die[use] is inner_op:
                    loaded_values.remove(use)

            # Process definitions
            defns = OrderedSet(
                i for i in inner_op.results if isinstance(i.type, base_reg_type)
            )
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
            if isinstance(spill_val.type, IntRegisterType):
                alloca_op = riscv.stack.AllocaOp(builtin.i32)
            elif isinstance(spill_val.type, FloatRegisterType):
                alloca_op = riscv.stack.AllocaOp(builtin.f32)
            else:
                raise NotImplementedError(
                    f"Unsupported register type: {spill_val.type}."
                    "Only int and float regs supported"
                )
            spill_op = riscv.stack.StoreOp(alloca_op, spill_val)
            builder.insert_op(alloca_op)
            builder.insert_op(spill_op)

            # replace all uses after spill to the ref
            spill_val.replace_uses_with_if(
                alloca_op.ref, lambda use: spill_op.is_before_in_block(use.operation)
            )

    def insert_load(
        self, inner_op: Operation, loads: AbstractSet[SSAValue]
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
        """Create the operation where every values dies at."""
        die: dict[SSAValue, Operation] = {}
        seen_vals = set[SSAValue]()
        for inner_op in func_op.walk(reverse=True):
            # by SSA, we can just check if this is the first seen use
            uses = set(inner_op.operands)
            for dead_val in uses - seen_vals:
                die[dead_val] = inner_op
            seen_vals |= uses
        return die


@dataclass(frozen=True)
class TestRiscvSpillingPass(ModulePass):
    name = "test-riscv-spilling"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        SpillPass().apply(ctx, op)
