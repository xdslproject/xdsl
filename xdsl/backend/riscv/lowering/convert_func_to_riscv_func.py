from xdsl.backend.riscv.lowering.utils import (
    a_regs,
    a_regs_for_types,
    cast_block_args_from_a_regs,
    cast_to_regs,
    move_to_a_regs,
    move_to_unallocated_regs,
    register_type_for_type,
)
from xdsl.context import Context
from xdsl.dialects import func, riscv_func
from xdsl.dialects.builtin import ModuleOp, StringAttr, UnrealizedConversionCastOp
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if len(op.function_type.inputs.data) > 8:
            raise ValueError("Cannot lower func.func with more than 8 inputs")
        if len(op.function_type.outputs.data) > 2:
            raise ValueError("Cannot lower func.func with more than 2 outputs")

        if (first_block := op.body.blocks.first) is not None:
            cast_block_args_from_a_regs(first_block, rewriter)

            input_types = first_block.arg_types
        else:
            input_types = tuple(a_regs_for_types(op.function_type.inputs.data))
        result_types = list(a_regs_for_types(op.function_type.outputs.data))

        # TODO we should ask the target for alignment, this works for rv32
        p2align = 2

        # C-like: default is public
        sym_visibility = (
            StringAttr("public") if op.sym_visibility is None else op.sym_visibility
        )

        new_func = riscv_func.FuncOp(
            op.sym_name.data,
            rewriter.move_region_contents_to_new_regions(op.body),
            (input_types, result_types),
            sym_visibility,
            p2align=p2align,
        )

        rewriter.replace_op(op, new_func)


class LowerFuncCallOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter) -> None:
        if len(op.arguments) > 8:
            raise ValueError("Cannot lower func.call with more than 8 operands")
        if len(op.res) > 2:
            raise ValueError("Cannot lower func.call with more than 2 results")

        if len(op.results) == 1:
            rewriter.name_hint = op.results[0].name_hint

        register_operands = cast_to_regs(op.arguments, register_type_for_type, rewriter)
        operand_types = op.arguments.types
        move_operand_ops, moved_operands = move_to_a_regs(
            register_operands, operand_types
        )

        new_result_types = list(a_regs(op.results))
        new_op = riscv_func.CallOp(op.callee, moved_operands, new_result_types)

        move_result_ops, moved_results = move_to_unallocated_regs(
            new_op.results, op.result_types
        )
        cast_result_ops = [
            UnrealizedConversionCastOp.get((moved_result,), (old_result.type,))
            for moved_result, old_result in zip(moved_results, op.results)
        ]
        rewriter.replace_op(
            op,
            [
                op
                for ops in (
                    move_operand_ops,
                    (new_op,),
                    move_result_ops,
                    cast_result_ops,
                )
                for op in ops
            ],
            [op.results[-1] for op in cast_result_ops],
        )


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter):
        if len(op.arguments) > 2:
            raise ValueError("Cannot lower func.return with more than 2 arguments")

        register_values = cast_to_regs(op.arguments, register_type_for_type, rewriter)
        move_ops, moved_values = move_to_a_regs(register_values, op.arguments.types)

        rewriter.insert_op(move_ops)

        rewriter.replace_op(op, riscv_func.ReturnOp(*moved_values))


class ConvertFuncToRiscvFuncPass(ModulePass):
    name = "convert-func-to-riscv-func"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerFuncOp(),
                    LowerFuncCallOp(),
                    LowerReturnOp(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
