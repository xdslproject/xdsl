from collections.abc import Iterable, Iterator

from xdsl.dialects import builtin, riscv
from xdsl.ir import Attribute, Block, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter


def register_type_for_type(
    attr: Attribute,
) -> type[riscv.IntRegisterType] | type[riscv.FloatRegisterType]:
    """
    Returns the appropriate register fype for a given input type.
    """
    if isinstance(attr, riscv.IntRegisterType | riscv.FloatRegisterType):
        return type(attr)
    if isinstance(attr, builtin.AnyFloat):
        return riscv.FloatRegisterType
    return riscv.IntRegisterType


def cast_to_regs(values: Iterable[SSAValue]) -> tuple[list[Operation], list[SSAValue]]:
    """
    Return cast operations for operands that don't already have a register type, and
    the new list of values that are all guaranteed to have register types.
    """

    new_ops = list[Operation]()
    new_values = list[SSAValue]()

    for value in values:
        if not isinstance(value.type, riscv.RISCVRegisterType):
            register_type = register_type_for_type(value.type)
            cast_op = builtin.UnrealizedConversionCastOp.get(
                (value,), (register_type.unallocated(),)
            )
            new_ops.append(cast_op)
            value = cast_op.results[0]
        new_values.append(value)

    return new_ops, new_values


def move_ops_for_value(
    value: SSAValue, rd: riscv.RISCVRegisterType
) -> tuple[Operation, SSAValue]:
    """
    Returns the operation that moves the value from the input to a new register.
    """

    if isinstance(rd, riscv.IntRegisterType):
        mv_op = riscv.MVOp(value, rd=rd)
        return mv_op, mv_op.rd
    elif isinstance(rd, riscv.FloatRegisterType):
        mv_op = riscv.FMVOp(value, rd=rd)
        return mv_op, mv_op.rd
    else:
        raise NotImplementedError(f"Unsupported register type for move op: {rd}")


def move_to_regs(
    values: Iterable[SSAValue], reg_types: Iterable[riscv.RISCVRegisterType]
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Return move operations to `a` registers (a0, a1, ... | fa0, fa1, ...).
    """

    new_ops = list[Operation]()
    new_values = list[SSAValue]()

    for value, register_type in zip(values, reg_types):
        move_op, new_value = move_ops_for_value(value, register_type)
        new_ops.append(move_op)
        new_values.append(new_value)

    return new_ops, new_values


def a_regs(values: Iterable[SSAValue]) -> Iterator[riscv.RISCVRegisterType]:
    return (
        register_type_for_type(value.type).a_register(index)
        for index, value in enumerate(values)
    )


def move_to_a_regs(
    values: Iterable[SSAValue],
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Return move operations to `a` registers (a0, a1, ... | fa0, fa1, ...).
    """
    return move_to_regs(values, a_regs(values))


def move_to_unallocated_regs(
    values: Iterable[SSAValue],
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Return move operations to `a` registers (a0, a1, ... | fa0, fa1, ...).
    """

    new_ops = list[Operation]()
    new_values = list[SSAValue]()

    for value in values:
        register_type = register_type_for_type(value.type)
        move_op, new_value = move_ops_for_value(value, register_type.unallocated())
        new_ops.append(move_op)
        new_values.append(new_value)

    return new_ops, new_values


def cast_operands_to_regs(rewriter: PatternRewriter) -> list[SSAValue]:
    """
    Add cast operations just before the targeted operation
    if the operands were not already int registers
    """

    new_ops = list[Operation]()
    new_operands = list[SSAValue]()

    for operand in rewriter.current_operation.operands:
        if not isinstance(
            operand.type, riscv.IntRegisterType | riscv.FloatRegisterType
        ):
            new_type = register_type_for_type(operand.type)
            cast_op = builtin.UnrealizedConversionCastOp.get(
                (operand,), (new_type.unallocated(),)
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


def cast_block_args_from_a_regs(block: Block, rewriter: PatternRewriter):
    """
    Change the type of the block arguments to `a` registers and add cast operations just
    after the block entry.
    """

    new_ops: list[Operation] = []

    for index, arg in enumerate(block.args):
        register_type = register_type_for_type(arg.type)
        move_op, new_value = move_ops_for_value(arg, register_type.unallocated())
        cast_op = builtin.UnrealizedConversionCastOp.get((new_value,), (arg.type,))
        new_ops.append(move_op)
        new_ops.append(cast_op)

        arg.type = register_type.a_register(index)
        arg.replace_by(cast_op.results[0])
        move_op.operands[0] = arg

    rewriter.insert_op_at_start(new_ops, block)


def cast_block_args_to_regs(block: Block, rewriter: PatternRewriter):
    """
    Change the type of the block arguments to registers and add cast operations just after
    the block entry.
    """

    for arg in block.args:
        rewriter.insert_op_at_start(
            new_val := builtin.UnrealizedConversionCastOp(
                operands=[arg], result_types=[arg.type]
            ),
            block,
        )

        arg.type = register_type_for_type(arg.type).unallocated()
        arg.replace_by(new_val.results[0])
        new_val.operands[new_val.results[0].index] = arg
