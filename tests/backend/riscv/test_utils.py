from xdsl.backend.riscv.lowering.utils import (
    cast_block_args_to_int_regs,
    cast_operands_to_int_regs,
    cast_results_from_int_regs,
)
from xdsl.builder import Builder
from xdsl.dialects import builtin, riscv, test
from xdsl.ir import BlockArgument
from xdsl.pattern_rewriter import PatternRewriter

INDEX_TYPE = builtin.IndexType()
REGISTER_TYPE = riscv.IntRegisterType.unallocated()


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
            v = test.TestOp.create(args, [INDEX_TYPE, INDEX_TYPE])
            test.TestTermOp.create(v.results, [INDEX_TYPE, INDEX_TYPE])

        test.TestOp.create((), (), regions=(inner,))

    @builtin.ModuleOp
    @Builder.implicit_region
    def expected():
        @Builder.implicit_region((INDEX_TYPE, INDEX_TYPE))
        def inner(args: tuple[BlockArgument, ...]):
            (first_arg, second_arg) = args
            first_arg_cast = builtin.UnrealizedConversionCastOp(
                (first_arg,), (REGISTER_TYPE,)
            )
            second_arg_cast = builtin.UnrealizedConversionCastOp(
                (second_arg,), (REGISTER_TYPE,)
            )
            v = riscv.CustomAssemblyInstructionOp(
                "foo",
                (first_arg_cast.results[0], second_arg_cast.results[0]),
                (REGISTER_TYPE, REGISTER_TYPE),
            )
            v1 = builtin.UnrealizedConversionCastOp.get((v.results[0],), (INDEX_TYPE,))
            v2 = builtin.UnrealizedConversionCastOp.get((v.results[1],), (INDEX_TYPE,))
            test.TestTermOp.create(
                (
                    v1.results[0],
                    v2.results[0],
                ),
                [INDEX_TYPE, INDEX_TYPE],
            )

        test.TestOp.create((), (), regions=(inner,))

    target_op = next(filter(lambda op: len(op.results) == 2, input.walk()))
    assert target_op is not None
    rewriter = PatternRewriter(target_op)
    (first_arg_cast, second_arg_cast) = cast_operands_to_int_regs(rewriter)
    cast_results_from_int_regs(rewriter)
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
            v = test.TestOp.create(args, [INDEX_TYPE, INDEX_TYPE])
            test.TestTermOp.create(v.results, [INDEX_TYPE, INDEX_TYPE])

        test.TestOp.create((), (), regions=(inner,))

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
                (second_arg,), (INDEX_TYPE,)
            )
            first_arg_cast = builtin.UnrealizedConversionCastOp(
                (first_arg,), (INDEX_TYPE,)
            )

            v = test.TestOp.create(
                (
                    first_arg_cast.results[0],
                    second_arg_cast.results[0],
                ),
                [INDEX_TYPE, INDEX_TYPE],
            )
            test.TestTermOp.create(v.results, [INDEX_TYPE, INDEX_TYPE])

        test.TestOp.create((), (), regions=(inner,))

    target_op = next(filter(lambda op: isinstance(op, test.TestOp), input.walk()))
    assert target_op is not None
    rewriter = PatternRewriter(target_op)
    target_block = target_op.regions[0].blocks[0]
    assert target_block is not None

    cast_block_args_to_int_regs(target_block, rewriter)
    input.verify()
    assert f"{input}" == f"{expected}"
