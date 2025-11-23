from xdsl.context import Context
from xdsl.dialects import builtin, func, x86, x86_func
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.ir import Attribute, Block
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
    x86.registers.RDI,
    x86.registers.RSI,
    x86.registers.RDX,
    x86.registers.RCX,
    x86.registers.R8,
    x86.registers.R9,
]

return_passing_register = x86.registers.RAX


# According to x86 calling conventions, the maximum number of
# registers available for passing function arguments. Other function
# arguments (when the function have more arguments than
# MAX_REG_PASSING_INPUTS) are passed using the stack.
MAX_REG_PASSING_INPUTS = 6

# For now, we reserve a pre-defined number of bytes for each argument
# passed via the stack. Therefore, input variables requiring more than
# STACK_SLOT_SIZE_BYTES bytes are not allowed.
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

        if op.sym_visibility == StringAttr("public"):
            directive_op = x86.DirectiveOp(".global", op.sym_name)
            rewriter.insert_op(directive_op)

        num_inputs = len(op.function_type.inputs.data)
        reg_args_types = arg_passing_registers[
            : min(num_inputs, MAX_REG_PASSING_INPUTS)
        ]

        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        first_block = new_region.blocks.first
        assert isinstance(first_block, Block)

        insertion_point = InsertPoint.at_start(first_block)

        # Load the register-carried parameters
        for i, register_type in enumerate(reg_args_types):
            arg = first_block.args[i]
            register = first_block.insert_arg(register_type, i)
            mov_op = x86.DS_MovOp(
                source=register, destination=x86.registers.UNALLOCATED_GENERAL
            )
            cast_op, parameter = builtin.UnrealizedConversionCastOp.cast_one(
                mov_op.destination, arg.type
            )
            rewriter.insert_op([mov_op, cast_op], insertion_point)
            arg.replace_by(parameter)
            first_block.erase_arg(arg)

        # The last argument of the basic block should be the stack pointer
        sp = first_block.insert_arg(
            x86.registers.RSP, min(num_inputs, MAX_REG_PASSING_INPUTS)
        )

        # If needed, load the stack-carried parameters by iteratively
        # consuming the 7th argument of the basic block. Once the 7th argument
        # has been read from the stack, it is removed from the
        # basic block arguments, and the former 8th becomes the 7th.
        for i in range(num_inputs - MAX_REG_PASSING_INPUTS):
            arg = first_block.args[MAX_REG_PASSING_INPUTS + 1]
            assert sp != arg
            mov_op = x86.DM_MovOp(
                memory=sp,
                memory_offset=STACK_SLOT_SIZE_BYTES * (i + 1),
                destination=x86.registers.UNALLOCATED_GENERAL,
                comment=f"Load the {i + MAX_REG_PASSING_INPUTS + 1}th argument of the function",
            )
            cast_op = builtin.UnrealizedConversionCastOp.get(
                (mov_op.destination,), (arg.type,)
            )
            rewriter.insert_op([mov_op, cast_op], insertion_point)
            arg.replace_by(cast_op.results[0])
            first_block.erase_arg(arg)

        outputs_types: list[Attribute] = []
        if len(op.function_type.outputs.data) == 1:
            outputs_types.append(return_passing_register)

        new_func = x86_func.FuncOp(
            op.sym_name.data,
            new_region,
            (reg_args_types + [x86.registers.RSP], outputs_types),
            visibility=op.sym_visibility,
        )

        rewriter.replace_op(op, new_func)


class LowerReturnOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.ReturnOp, rewriter: PatternRewriter):
        if not op.arguments:
            rewriter.replace_op(op, [x86_func.RetOp()])
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
            (return_value,), (x86.registers.UNALLOCATED_GENERAL,)
        )
        mov_op = x86.ops.DS_MovOp(cast_op, destination=return_passing_register)
        ret_op = x86_func.RetOp()

        rewriter.replace_op(op, [cast_op, mov_op, ret_op])


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
