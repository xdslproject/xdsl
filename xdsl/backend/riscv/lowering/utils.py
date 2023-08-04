from xdsl.dialects import builtin, riscv
from xdsl.ir import Block, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def cast_operands_to_int_regs(rewriter: PatternRewriter) -> list[SSAValue]:
    """
    Add cast operations just before the targeted operation
    if the operands were not already int registers
    """

    new_ops = list[Operation]()
    new_operands = list[SSAValue]()

    for operand in rewriter.current_operation.operands:
        if not isinstance(operand.type, riscv.IntRegisterType):
            cast_op = builtin.UnrealizedConversionCastOp.get(
                (operand,), (riscv.IntRegisterType.unallocated(),)
            )
            new_ops.append(cast_op)
            operand = cast_op.results[0]
        new_operands.append(operand)

    rewriter.insert_op_before_matched_op(new_ops)
    return new_operands


def cast_operands_to_float_regs(rewriter: PatternRewriter) -> list[SSAValue]:
    """
    Add cast operations just before the targeted operation
    if the operands were not already float registers
    """

    new_ops = list[Operation]()
    new_operands = list[SSAValue]()

    for operand in rewriter.current_operation.operands:
        if not isinstance(operand.type, riscv.FloatRegisterType):
            cast_op = builtin.UnrealizedConversionCastOp.get(
                (operand,), (riscv.FloatRegisterType.unallocated(),)
            )
            new_ops.append(cast_op)
            operand = cast_op.results[0]
        new_operands.append(operand)

    rewriter.insert_op_before_matched_op(new_ops)
    return new_operands


def cast_matched_op_results(rewriter: PatternRewriter) -> list[SSAValue]:
    """
    Add cast operations just after the matched operation, to preserve the type validity of
    arguments of uses of results.
    """

    results = [
        builtin.UnrealizedConversionCastOp.get((val,), (val.type,))
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

    unallocated_reg = riscv.IntRegisterType.unallocated()

    for arg in block.args:
        rewriter.insert_op_at_start(
            new_val := builtin.UnrealizedConversionCastOp([arg], [arg.type]), block
        )

        arg.type = unallocated_reg
        arg.replace_by(new_val.results[0])
        new_val.operands[new_val.results[0].index] = arg
