from typing import Sequence

from xdsl.dialects import builtin, riscv
from xdsl.ir.core import SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def cast_operands_to_int_regs(
    vals: Sequence[SSAValue], rewriter: PatternRewriter
) -> list[SSAValue]:
    # convert all values to int registers if they are not already
    not_regs = filter(lambda v: isinstance(v.type, riscv.IntRegisterType), vals)
    mapped_vals = [
        builtin.UnrealizedConversionCastOp.get(
            [not_reg], (riscv.IntRegisterType.unallocated(),)
        )
        for not_reg in not_regs
    ]
    rewriter.insert_op_before_matched_op(mapped_vals)

    # return the original values if they were already int registers, otherwise return the casted values
    return [
        mapped_vals.pop(0).results[0]
        if isinstance(v.type, riscv.IntRegisterType)
        else v
        for v in vals
    ]


def cast_results_to_int_regs(
    vals: Sequence[SSAValue], rewriter: PatternRewriter
) -> list[SSAValue]:
    results = [
        builtin.UnrealizedConversionCastOp.get(
            [val], (riscv.IntRegisterType.unallocated(),)
        )
        for val in vals
    ]
    rewriter.insert_op_after_matched_op(results)
    return [result.results[0] for result in results]
