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
from xdsl.utils.exceptions import DiagnosticException

arg_passing_registers = [
    x86.register.RDI,
    x86.register.RSI,
    x86.register.RDX,
    x86.register.RCX,
    x86.register.R8,
    x86.register.R9,
]

return_passing_register = x86.register.RAX

stack_entry_size_in_bytes = 8


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        first_block = op.body.blocks.first

        if first_block is None:
            raise DiagnosticException(
                "Cannot lower external functions (not implemented)"
            )

        for arg in first_block.args:
            if isinstance(arg.type, builtin.ShapedType):
                raise DiagnosticException(
                    "Cannot lower shaped function parameters (not implemented)"
                )
            elif (
                isinstance(arg.type, builtin.FixedBitwidthType)
                and arg.type.bitwidth > 64
            ):
                raise DiagnosticException(
                    "Cannot lower function parameters bigger than 64 bits (not implemented)"
                )

        num_inputs = len(op.function_type.inputs.data)
        num_passing_args = num_inputs if num_inputs <= 6 else 6

        insertion_point = InsertPoint.at_start(first_block)

        # Get the 6 first parameters (if any) from general registers
        reg_args_types = arg_passing_registers[0:num_passing_args]
        actual_registers: list[SSAValue] = []
        for register_type in reg_args_types:
            get_reg_op = x86.ops.GetRegisterOp(register_type)
            rewriter.insert_op(get_reg_op, insertion_point)
            actual_registers.append(get_reg_op.result)

        # Get the other parameters (if any) from the stack
        if num_inputs > 6:
            get_sp_op = x86.ops.GetRegisterOp(x86.register.RSP)
            rewriter.insert_op(get_sp_op, insertion_point)
            for i in range(num_inputs - 6):
                get_reg_op = x86.ops.GetRegisterOp(x86.register.GeneralRegisterType(""))
                mov_op = x86.RM_MovOp(
                    r1=get_reg_op.result,
                    r2=get_sp_op.result,
                    offset=stack_entry_size_in_bytes * (i + 1),
                    result=x86.register.GeneralRegisterType(""),
                )
                actual_registers.append(mov_op.result)
                rewriter.insert_op([get_reg_op, mov_op], insertion_point)

        # Cast the registers to whatever type is needed
        for arg, register in zip(first_block.args, actual_registers):
            cast_op = builtin.UnrealizedConversionCastOp.get((register,), (arg.type,))
            arg.replace_by(cast_op.results[0])
            rewriter.insert_op(cast_op, insertion_point)

        # Create the new function
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


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter):
        if len(op.arguments) == 0:
            return
        elif len(op.arguments) > 1:
            raise DiagnosticException(
                "Cannot lower func.return with more than 1 argument (not implemented)"
            )

        return_value = op.arguments[0]

        if isinstance(return_value.type, builtin.ShapedType):
            raise DiagnosticException(
                "Cannot lower shaped function output (not implemented)"
            )
        elif (
            isinstance(return_value.type, builtin.FixedBitwidthType)
            and return_value.type.bitwidth > 64
        ):
            raise DiagnosticException(
                "Cannot lower function return values bigger than 64 bits (not implemented)"
            )

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
                    LowerReturnOp(),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
