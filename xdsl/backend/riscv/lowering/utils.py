from xdsl.dialects import builtin, riscv
from xdsl.ir import Block, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def cast_operands_to_int_regs(rewriter: PatternRewriter) -> list[SSAValue]:
    """
    Add cast operations just before the targeted operation
    if the operands were not already int registers
    """

    # convert all values to int registers if they are not already
    casted_ops = [
        builtin.UnrealizedConversionCastOp.get(
            [not_reg], (riscv.IntRegisterType.unallocated(),)
        )
        for not_reg in filter(
            lambda v: not isinstance(v.type, riscv.IntRegisterType),
            rewriter.current_operation.operands,
        )
    ]

    rewriter.insert_op_before_matched_op(casted_ops)

    # return the original value if it's already an int register
    # otherwise return the casted value
    return [
        casted_ops.pop(0).results[0]
        if not isinstance(operand.type, riscv.IntRegisterType)
        else operand
        for operand in rewriter.current_operation.operands
    ]


def cast_results_from_int_regs(rewriter: PatternRewriter) -> list[SSAValue]:
    """
    Add cast operations just after the targeted operation from int registers
    to the original type.
    """

    results = [
        builtin.UnrealizedConversionCastOp.get([val], (val.type,))
        for val in rewriter.current_operation.results
    ]

    for res, result in zip(rewriter.current_operation.results, results):
        for use in set(res.uses):
            # avoid recursion on the casts we just inserted
            if use.operation != result:
                use.operation.operands[use.index] = result.results[0]

    rewriter.insert_op_after_matched_op(results)
    return [result.results[0] for result in results]


def cast_block_args_to_int_regs(block: Block, rewriter: PatternRewriter):
    """
    Change the type of the block arguments to int registers and
    add cast operations just after the block entry.
    """

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
