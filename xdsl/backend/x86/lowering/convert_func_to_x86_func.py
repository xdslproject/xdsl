from xdsl.context import Context
from xdsl.dialects import builtin, func, x86, x86_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

arg_passing_registers = [
    x86.register.RDI,
    x86.register.RSI,
    x86.register.RDX,
    x86.register.RCX,
    x86.register.R8,
    x86.register.R9,
]

return_passing_register = x86.register.RAX


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        num_inputs = len(op.function_type.inputs.data)
        if num_inputs > 6:
            raise ValueError(
                "Cannot lower func.func with more than 6 inputs (not implemented)"
            )
        if op.body.blocks.first is None:
            raise ValueError("Cannot lower external functions (not implemented)")

        reg_args_types = arg_passing_registers[0:num_inputs]
        first_block = op.body.blocks.first

        insertion_point = InsertPoint.at_start(first_block)

        actual_registers: list[SSAValue] = []
        for register_type in reg_args_types:
            get_reg_op = x86.ops.GetRegisterOp(register_type)
            rewriter.insert_op(get_reg_op, insertion_point)
            actual_registers.append(get_reg_op.result)

        for arg, register in zip(first_block.args, actual_registers):
            if isinstance(arg.type, builtin.ShapedType):
                raise ValueError(
                    "Cannot lower shaped function parameters (not implemented)"
                )
            arg.replace_by(register)
            cast_op = builtin.UnrealizedConversionCastOp.get((register,), (arg.type,))
            rewriter.insert_op(cast_op, insertion_point)

        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        block = new_region.blocks.first
        assert isinstance(block, Block)
        for a in block.args:
            block.erase_arg(a)

        new_func = x86_func.FuncOp(
            op.sym_name.data,
            new_region,
            ([], []),
            visibility=op.sym_visibility,
        )

        rewriter.replace_matched_op(new_func)


class LowerFuncCallOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.CallOp, rewriter: PatternRewriter) -> None:
        raise NotImplementedError


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter):
        if len(op.arguments) == 0:
            return
        elif len(op.arguments) > 1:
            raise ValueError(
                "Cannot lower func.return with more than 1 argument (not implemented)"
            )

        return_value = op.arguments[0]

        if isinstance(return_value, builtin.ShapedType):
            raise ValueError("Cannot lower shaped function output (not implemented)")

        cast_op = builtin.UnrealizedConversionCastOp.get(
            (return_value,), (x86.register.GeneralRegisterType(""),)
        )
        get_reg_op = x86.ops.GetRegisterOp(return_passing_register)
        mov_op = x86.ops.RR_MovOp(cast_op, get_reg_op, result=return_passing_register)
        ret_op = x86_func.RetOp()

        rewriter.replace_matched_op([cast_op, get_reg_op, mov_op, ret_op])


class ConvertFuncToX86FuncPass(ModulePass):
    name = "convert-func-to-x86-func"

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
