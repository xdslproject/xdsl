from xdsl.backend.riscv.lowering.utils import (
    a_regs,
    cast_block_args_to_regs,
    cast_op_results,
    cast_operands_to_regs,
    register_type_for_type,
)
from xdsl.builder import Builder
from xdsl.dialects import builtin, memref, riscv, test
from xdsl.ir import BlockArgument
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.utils.test_value import create_ssa_value

INDEX_TYPE = builtin.IndexType()
REGISTER_TYPE = riscv.Registers.UNALLOCATED_INT


def test_register_type_for_type():
    assert register_type_for_type(builtin.i32) == riscv.IntRegisterType
    assert (
        register_type_for_type(memref.MemRefType(builtin.f32, [1, 2, 3]))
        == riscv.IntRegisterType
    )
    assert (
        register_type_for_type(memref.MemRefType(builtin.i32, [1, 2, 3]))
        == riscv.IntRegisterType
    )

    assert register_type_for_type(builtin.f32) == riscv.FloatRegisterType
    assert register_type_for_type(builtin.f64) == riscv.FloatRegisterType


def test_op_cast_utils():
    @builtin.ModuleOp
    @Builder.implicit_region
    def input():
        @Builder.implicit_region(
            (
                INDEX_TYPE,
                INDEX_TYPE,
            )
        )
        def inner(args: tuple[BlockArgument, ...]):
            v = test.TestOp.create(operands=args, result_types=[INDEX_TYPE, INDEX_TYPE])
            test.TestTermOp.create(
                operands=v.results, result_types=[INDEX_TYPE, INDEX_TYPE]
            )

        test.TestOp.create(regions=(inner,))

    @builtin.ModuleOp
    @Builder.implicit_region
    def expected():
        @Builder.implicit_region((INDEX_TYPE, INDEX_TYPE))
        def inner(args: tuple[BlockArgument, ...]):
            (first_arg, second_arg) = args
            first_arg_cast = builtin.UnrealizedConversionCastOp(
                operands=(first_arg,), result_types=(REGISTER_TYPE,)
            )
            second_arg_cast = builtin.UnrealizedConversionCastOp(
                operands=(second_arg,), result_types=(REGISTER_TYPE,)
            )
            v = riscv.CustomAssemblyInstructionOp(
                "foo",
                (first_arg_cast.results[0], second_arg_cast.results[0]),
                (REGISTER_TYPE, REGISTER_TYPE),
            )
            v1 = builtin.UnrealizedConversionCastOp.get((v.results[0],), (INDEX_TYPE,))
            v2 = builtin.UnrealizedConversionCastOp.get((v.results[1],), (INDEX_TYPE,))
            test.TestTermOp.create(
                operands=(
                    v1.results[0],
                    v2.results[0],
                ),
                result_types=[INDEX_TYPE, INDEX_TYPE],
            )

        test.TestOp.create(regions=(inner,))

    target_op = next(filter(lambda op: len(op.results) == 2, input.walk()))
    assert target_op is not None
    rewriter = PatternRewriter(target_op)
    (first_arg_cast, second_arg_cast) = cast_operands_to_regs(rewriter, target_op)
    cast_op_results(rewriter, target_op)
    lowered_op = riscv.CustomAssemblyInstructionOp(
        "foo", (first_arg_cast, second_arg_cast), (REGISTER_TYPE, REGISTER_TYPE)
    )
    rewriter.replace_op(target_op, lowered_op)

    # check that the lowered region is still valid
    input.verify()

    assert f"{input}" == f"{expected}"


def test_block_cast_utils():
    @builtin.ModuleOp
    @Builder.implicit_region
    def input():
        @Builder.implicit_region(
            (
                INDEX_TYPE,
                INDEX_TYPE,
            )
        )
        def inner(args: tuple[BlockArgument, ...]):
            v = test.TestOp.create(operands=args, result_types=[INDEX_TYPE, INDEX_TYPE])
            test.TestTermOp.create(
                operands=v.results, result_types=[INDEX_TYPE, INDEX_TYPE]
            )

        test.TestOp.create(regions=(inner,))

    @builtin.ModuleOp
    @Builder.implicit_region
    def expected():
        @Builder.implicit_region(
            (
                REGISTER_TYPE,
                REGISTER_TYPE,
            )
        )
        def inner(args: tuple[BlockArgument, ...]):
            (first_arg, second_arg) = args
            second_arg_cast = builtin.UnrealizedConversionCastOp(
                operands=(second_arg,), result_types=(INDEX_TYPE,)
            )
            first_arg_cast = builtin.UnrealizedConversionCastOp(
                operands=(first_arg,), result_types=(INDEX_TYPE,)
            )

            v = test.TestOp.create(
                operands=(
                    first_arg_cast.results[0],
                    second_arg_cast.results[0],
                ),
                result_types=[INDEX_TYPE, INDEX_TYPE],
            )
            test.TestTermOp.create(
                operands=v.results, result_types=[INDEX_TYPE, INDEX_TYPE]
            )

        test.TestOp.create(regions=(inner,))

    target_op = next(filter(lambda op: isinstance(op, test.TestOp), input.walk()))
    assert target_op is not None
    rewriter = PatternRewriter(target_op)
    target_block = target_op.regions[0].blocks[0]
    assert target_block is not None

    cast_block_args_to_regs(target_block, rewriter)
    input.verify()
    assert f"{input}" == f"{expected}"


def test_a_regs():
    assert list(a_regs([])) == []
    assert list(
        a_regs(
            (
                create_ssa_value(riscv.Registers.FT0),
                create_ssa_value(riscv.Registers.FT0),
                create_ssa_value(riscv.Registers.T0),
            )
        )
    ) == [riscv.Registers.FA0, riscv.Registers.FA1, riscv.Registers.A0]
