from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from ordered_set import OrderedSet

from xdsl.context import Context
from xdsl.dialects import riscv_func
from xdsl.dialects.builtin import DenseArrayBase, ModuleOp, i32
from xdsl.dialects.riscv import RISCVRegisterType
from xdsl.dialects.riscv.ops import ParallelMovOp
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter


def extend_parallel_mov_op(
    pmov_op: ParallelMovOp, new_values: Sequence[SSAValue]
) -> ParallelMovOp:
    """Extend a ParallelMovOp with additional inputs/outputs that must cross it."""

    new_input_types: list[RISCVRegisterType] = []
    new_widths: list[int] = []

    for value in new_values:
        input_type = cast(RISCVRegisterType, value.type)
        new_input_types.append(type(input_type).unallocated())  # unallocated output

        # Assume 32 bit bitwidth
        new_widths.append(32)

    # Extend arrays
    all_inputs = list(pmov_op.inputs) + list(new_values)
    all_outputs = list(i.type for i in pmov_op.outputs) + new_input_types
    all_widths = list(pmov_op.input_widths.get_values()) + new_widths

    return ParallelMovOp(
        all_inputs,
        all_outputs,
        DenseArrayBase.from_list(i32, all_widths),
        pmov_op.free_registers,
    )


@dataclass(frozen=True)
class RISCVLegalizeParallelMovPass(ModulePass):
    """Legalizes ParallelMovOp such that no live ranges cross it."""

    name = "riscv-legalize-parallel-mov"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        for func_op in op.walk():
            if not isinstance(func_op, riscv_func.FuncOp):
                continue
            self.process_func(func_op)

    def process_func(self, func_op: riscv_func.FuncOp):
        live_values: OrderedSet[SSAValue] = OrderedSet([])
        for inner_op in func_op.walk(reverse=True):
            if isinstance(inner_op, ParallelMovOp):
                self.fill_live_values(inner_op, live_values)

            # Liveness analysis
            for defn in inner_op.results:
                live_values.discard(defn)
            for use in inner_op.operands:
                live_values.add(use)

    def fill_live_values(
        self, pmov_op: ParallelMovOp, live_values: OrderedSet[SSAValue]
    ):
        new_values: list[SSAValue] = []
        for value in live_values:
            if value not in pmov_op.results:
                # This value's live range crosses the pmov.
                # Add the value to the pmov
                new_values.append(value)

        if not new_values:
            return

        new_pmov = extend_parallel_mov_op(pmov_op, new_values)

        Rewriter.replace_op(pmov_op, new_pmov, new_pmov.results[: len(pmov_op.results)])

        for old_val, new_val in zip(
            new_values, new_pmov.outputs[len(pmov_op.outputs) :], strict=True
        ):
            old_val.replace_uses_with_if(
                new_val, lambda use: new_pmov.is_before_in_block(use.operation)
            )
