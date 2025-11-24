from collections import Counter
from collections.abc import Iterable, Iterator

from xdsl.backend.utils import cast_to_regs
from xdsl.dialects import builtin, riscv
from xdsl.ir import Attribute, Block, Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint


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


def move_ops_for_value(
    value: SSAValue, value_type: Attribute, rd: riscv.RISCVRegisterType
) -> tuple[Operation, SSAValue]:
    """
    Returns the operation that moves the value from the input to a new register.
    In order to disambiguate which floating point move should be used (fmv.s vs fmv.d),
    the floating point type in question must be passed
    """

    if isinstance(rd, riscv.IntRegisterType):
        mv_op = riscv.MVOp(value, rd=rd)
        return mv_op, mv_op.rd
    elif isinstance(rd, riscv.FloatRegisterType):
        match value_type:
            case builtin.Float64Type():
                mv_op = riscv.FMvDOp(value, rd=rd)
            case builtin.Float32Type():
                mv_op = riscv.FMVOp(value, rd=rd)
            case _:
                raise NotImplementedError(
                    f"Move operation for float register containing value of type {value.type} is not implemented"
                )
        return mv_op, mv_op.rd
    else:
        raise NotImplementedError(f"Unsupported register type for move op: {rd}")


def move_to_regs(
    values: Iterable[SSAValue],
    value_types: Iterable[Attribute],
    reg_types: Iterable[riscv.RISCVRegisterType],
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Return move operations to `a` registers (a0, a1, ... | fa0, fa1, ...).
    """

    new_ops = list[Operation]()
    new_values = list[SSAValue]()

    for value, value_type, register_type in zip(
        values, value_types, reg_types, strict=True
    ):
        move_op, new_value = move_ops_for_value(value, value_type, register_type)
        new_ops.append(move_op)
        new_values.append(new_value)

    return new_ops, new_values


def a_regs_for_types(types: Iterable[Attribute]) -> Iterator[riscv.RISCVRegisterType]:
    """
    Returns the "a" registers in which to store types, i.e. `fa0`, `fa1`, etc for
    floating-point values and `a0`, `a1`, etc for integer values and pointers. The
    register index is separate for integer and floating-point registers according to the
    RISC-V ABI.
    """
    counter = Counter[type[riscv.RISCVRegisterType]]()
    for attr_type in types:
        register_type = register_type_for_type(attr_type)
        index = counter[register_type]
        yield register_type.a_register(index)
        counter[register_type] += 1


def a_regs(values: Iterable[SSAValue]) -> Iterator[riscv.RISCVRegisterType]:
    return a_regs_for_types(value.type for value in values)


def move_to_a_regs(
    values: Iterable[SSAValue],
    value_types: Iterable[Attribute],
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Return move operations to `a` registers (a0, a1, ... | fa0, fa1, ...).
    """
    return move_to_regs(values, value_types, a_regs(values))


def move_to_unallocated_regs(
    values: Iterable[SSAValue],
    value_types: Iterable[Attribute],
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Return move operations to unallocated registers.
    """

    new_ops = list[Operation]()
    new_values = list[SSAValue]()

    for value, value_type in zip(values, value_types, strict=True):
        register_type = register_type_for_type(value.type)
        move_op, new_value = move_ops_for_value(
            value, value_type, register_type.unallocated()
        )
        new_ops.append(move_op)
        new_values.append(new_value)

    return new_ops, new_values


def cast_operands_to_regs(rewriter: PatternRewriter) -> list[SSAValue]:
    """
    Add cast operations just before the targeted operation
    if the operands were not already int registers
    """
    return cast_to_regs(
        rewriter.current_operation.operands, register_type_for_type, rewriter
    )


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

    rewriter.insert_op(results, InsertPoint.after(rewriter.current_operation))
    return [result.results[0] for result in results]


def cast_block_args_from_a_regs(block: Block, rewriter: PatternRewriter):
    """
    Change the type of the block arguments to "a" registers and add cast operations just
    after the block entry. Use `fa0`, `fa1`, etc for floating-point values and `a0`, `a1`,
    etc for integer values and pointers. The register index is separate for integer and
    floating-point registers according to the RISC-V ABI.
    """

    new_ops: list[Operation] = []
    counter = Counter[type[riscv.RISCVRegisterType]]()

    for arg in block.args:
        register_type = register_type_for_type(arg.type)
        move_op, new_value = move_ops_for_value(
            arg, arg.type, register_type.unallocated()
        )
        cast_op = builtin.UnrealizedConversionCastOp.get((new_value,), (arg.type,))
        new_ops.append(move_op)
        new_ops.append(cast_op)

        index = counter[register_type]
        arg.replace_by_if(cast_op.results[0], lambda use: use.operation != move_op)
        rewriter.replace_value_with_new_type(arg, register_type.a_register(index))
        counter[register_type] += 1

    rewriter.insert_op(new_ops, InsertPoint.at_start(block))


def cast_block_args_to_regs(block: Block, rewriter: PatternRewriter):
    """
    Change the type of the block arguments to registers and add cast operations just after
    the block entry.
    """

    for arg in block.args:
        rewriter.insert_op(
            cast_op := builtin.UnrealizedConversionCastOp(
                operands=[arg], result_types=[arg.type]
            ),
            InsertPoint.at_start(block),
        )
        new_val = cast_op.results[0]

        new_type = register_type_for_type(arg.type).unallocated()
        arg.replace_by_if(new_val, lambda use: use.operation != cast_op)
        rewriter.replace_value_with_new_type(arg, new_type)
