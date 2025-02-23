from xdsl.context import Context
from xdsl.dialects import builtin, func, x86, x86_func
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Block, BlockArgument, SSAValue
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

STACK_SLOT_SIZE_BYTES = 8


class LowerFuncOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter):
        if op.body.blocks.first is None:
            raise DiagnosticException(
                "Cannot lower external functions (not implemented)"
            )

        for ty in op.function_type.inputs.data:
            if isinstance(ty, builtin.ShapedType):
                raise DiagnosticException(
                    "Cannot lower shaped function parameters (not implemented)"
                )
            elif isinstance(ty, builtin.FixedBitwidthType) and ty.bitwidth > 64:
                raise DiagnosticException(
                    "Cannot lower function parameters bigger than 64 bits (not implemented)"
                )

        num_inputs = len(op.function_type.inputs.data)
        reg_args_types = arg_passing_registers[:min(num_inputs,6)]

        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        first_block = new_region.blocks.first
        assert isinstance(first_block, Block)

        insertion_point = InsertPoint.at_start(first_block)

        params_mapping: list[tuple[BlockArgument, SSAValue]] = []

        # Build the basic block header
        for i, register_type in enumerate(reg_args_types):
            arg = first_block.args[i]
            register = first_block.insert_arg(register_type, i)
            params_mapping.append((arg, register))
        sp_tmp_index = num_inputs if num_inputs < 6 else 6
        sp = first_block.insert_arg(x86.register.RSP, sp_tmp_index)

        # Load the stack-carried parameters in registers
        for i in range(num_inputs - 6):
            arg = first_block.args[6 + i + 1]
            assert sp != arg
            get_reg_op = x86.ops.GetRegisterOp(x86.register.GeneralRegisterType(""))
            mov_op = x86.RM_MovOp(
                r1=get_reg_op.result,
                r2=sp,
                offset=STACK_SLOT_SIZE_BYTES * (i + 1),
                result=x86.register.GeneralRegisterType(""),
            )
            rewriter.insert_op([get_reg_op, mov_op], insertion_point)
            params_mapping.append((arg, mov_op.result))

        for old_param, new_param in params_mapping:
            cast_op = builtin.UnrealizedConversionCastOp.get(
                (new_param,), (old_param.type,)
            )
            rewriter.insert_op([cast_op], insertion_point)
            old_param.replace_by(cast_op.results[0])
            first_block.erase_arg(old_param)

        outputs_types: list[Attribute] = []
        if len(op.function_type.outputs.data) == 1:
            outputs_types.append(return_passing_register)

        new_func = x86_func.FuncOp(
            op.sym_name.data,
            new_region,
            (reg_args_types + [x86.register.RSP], outputs_types),
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
