from typing import Sequence

from xdsl.dialects import builtin, riscv
from xdsl.ir import Block, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def cast_operands_to_int_regs(
    vals: Sequence[SSAValue], rewriter: PatternRewriter
) -> list[SSAValue]:
    # convert all values to int registers if they are not already
    not_regs = filter(lambda v: not isinstance(v.type, riscv.IntRegisterType), vals)
    mapped_vals = [
        builtin.UnrealizedConversionCastOp.get(
            [not_reg], (riscv.IntRegisterType.unallocated(),)
        )
        for not_reg in not_regs
    ]
    rewriter.insert_op_before_matched_op(mapped_vals)

    # return the original values if they were already int registers, otherwise
    # return the casted values
    return [
        mapped_vals.pop(0).results[0]
        if not isinstance(v.type, riscv.IntRegisterType)
        else v
        for v in vals
    ]


def cast_results_to_int_regs(
    vals: Sequence[SSAValue], rewriter: PatternRewriter
) -> list[SSAValue]:
    results = [
        builtin.UnrealizedConversionCastOp.get([val], (val.type,)) for val in vals
    ]

    for val, result in zip(vals, results):
        for use in set(val.uses):
            # avoid recursion on the casts we just inserted
            if use.operation != result:
                use.operation.operands[use.index] = result.results[0]

    rewriter.insert_op_after_matched_op(results)
    return [result.results[0] for result in results]


def cast_block_args_to_int_regs(block: Block, rewriter: PatternRewriter):
    if first_op := block.first_op:
        unallocated_reg = riscv.IntRegisterType.unallocated()

        for arg in block.args:
            rewriter.insert_op_before(
                new_val := builtin.UnrealizedConversionCastOp([arg], [arg.type]),
                first_op,
            )

            arg.type = unallocated_reg
            arg.replace_by(new_val.results[0])
            new_val.operands[new_val.results[0].index] = arg
