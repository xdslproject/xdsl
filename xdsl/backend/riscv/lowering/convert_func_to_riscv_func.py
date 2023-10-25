from xdsl.backend.riscv.lowering.utils import (
    a_regs,
    cast_block_args_from_a_regs,
    cast_to_regs,
    move_to_a_regs,
    move_to_unallocated_regs,
    register_type_for_type,
)
from xdsl.dialects import func, riscv, riscv_func
from xdsl.dialects.builtin import ModuleOp, UnrealizedConversionCastOp
from xdsl.ir import Block, MLContext, Operation, Region
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

        first_block = op.body.blocks[0]
        cast_block_args_from_a_regs(first_block, rewriter)

        input_types = [arg.type for arg in first_block.args]
        result_types = [
            register_type_for_type(result_type).a_register(index)
            for index, result_type in enumerate(op.function_type.outputs.data)
        ]

        new_func = riscv_func.FuncOp(
            op.sym_name.data,
            rewriter.move_region_contents_to_new_regions(op.body),
            (input_types, result_types),
        )

        new_ops: list[Operation] = []

        if (visibility := op.sym_visibility) is None or visibility.data == "public":
            # C-like: default is public
            new_ops.append(riscv.DirectiveOp(".globl", op.sym_name.data))

        new_ops.append(
            # FIXME we should ask the target for alignment, this works for rv32
            riscv.DirectiveOp(".p2align", "2"),
        )

        new_ops.append(new_func)

        # Each function has its own .text: this will tell the assembler to emit
        # a .text section (if not present) and make it the current one
        # section = riscv.AssemblySectionOp(".text", Region(Block(result)))

        text_section = riscv.AssemblySectionOp(".text", Region(Block(new_ops)))

        rewriter.replace_matched_op(text_section)


class LowerFuncCallOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.Call, rewriter: PatternRewriter) -> None:
        if len(op.arguments) > 8:
            raise ValueError("Cannot lower func.call with more than 8 operands")
        if len(op.res) > 2:
            raise ValueError("Cannot lower func.call with more than 2 results")

        cast_operand_ops, register_operands = cast_to_regs(op.arguments)
        move_operand_ops, moved_operands = move_to_a_regs(register_operands)
        new_result_types = list(a_regs(op.results))
        new_op = riscv_func.CallOp(op.callee, moved_operands, new_result_types)
        move_result_ops, moved_results = move_to_unallocated_regs(new_op.results)
        cast_result_ops = [
            UnrealizedConversionCastOp.get((moved_result,), (old_result.type,))
            for moved_result, old_result in zip(moved_results, op.results)
        ]
        rewriter.replace_matched_op(
            [
                op
                for ops in (
                    cast_operand_ops,
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
    def match_and_rewrite(self, op: func.Return, rewriter: PatternRewriter):
        if len(op.arguments) > 2:
            raise ValueError("Cannot lower func.return with more than 2 arguments")

        cast_ops, register_values = cast_to_regs(op.arguments)
        move_ops, moved_values = move_to_a_regs(register_values)

        rewriter.insert_op_before_matched_op(cast_ops)
        rewriter.insert_op_before_matched_op(move_ops)

        rewriter.replace_matched_op(riscv_func.ReturnOp(*moved_values))


class ConvertFuncToRiscvFuncPass(ModulePass):
    name = "convert-func-to-riscv-func"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
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
