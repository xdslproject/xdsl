from dataclasses import dataclass

from xdsl.backend.x86.lowering.helpers import Arch
from xdsl.context import Context
from xdsl.dialects import builtin, func, x86, x86_func
from xdsl.dialects.builtin import ModuleOp, StringAttr
from xdsl.dialects.x86.registers import GeneralRegisterType
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

ARG_PASSING_REGISTER_INDICES = [
    7,
    6,
    2,
    1,
    8,
    9,
]
"""
ABI-specified function argument registers: RDI, RSI, RDX, RCX, R8, R9.
"""

RETURN_PASSING_REGISTER = 0
"""
ABI-specified return register: RAX.
"""


# According to x86 calling conventions, the maximum number of
# registers available for passing function arguments. Other function
# arguments (when the function have more arguments than
# MAX_REG_PASSING_INPUTS) are passed using the stack.
MAX_REG_PASSING_INPUTS = 6

# For now, we reserve a pre-defined number of bytes for each argument
# passed via the stack. Therefore, input variables requiring more than
# STACK_SLOT_SIZE_BYTES bytes are not allowed.
STACK_SLOT_SIZE_BYTES = 8


@dataclass
class LowerFuncOp(RewritePattern):
    arch: Arch

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
        new_region = rewriter.move_region_contents_to_new_regions(op.body)
        first_block = new_region.blocks.first
        assert isinstance(first_block, Block)

        reg_args_types = tuple(
            self.arch.register_type_for_type(arg.type).from_index(register_index)
            for register_index, arg in zip(
                ARG_PASSING_REGISTER_INDICES, first_block.args
            )
        )

        insertion_point = InsertPoint.at_start(first_block)

        # Load the register-carried parameters
        for i, register_type in enumerate(reg_args_types):
            arg = first_block.args[i]
            register = first_block.insert_arg(register_type, i)
            mov_op = x86.DS_MovOp(
                source=register, destination=register_type.unallocated()
            )
            cast_op, parameter = builtin.UnrealizedConversionCastOp.cast_one(
                mov_op.destination, arg.type
            )
            rewriter.insert_op([mov_op, cast_op], insertion_point)
            arg.replace_all_uses_with(parameter)
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
            destination_reg = self.arch.register_type_for_type(arg.type).unallocated()
            assert isinstance(destination_reg, GeneralRegisterType)
            mov_op = x86.DM_MovOp(
                memory=sp,
                memory_offset=STACK_SLOT_SIZE_BYTES * (i + 1),
                destination=destination_reg,
                comment=f"Load the {i + MAX_REG_PASSING_INPUTS + 1}th argument of the function",
            )
            cast_op = builtin.UnrealizedConversionCastOp.get(
                (mov_op.destination,), (arg.type,)
            )
            rewriter.insert_op([mov_op, cast_op], insertion_point)
            arg.replace_all_uses_with(cast_op.results[0])
            first_block.erase_arg(arg)

        outputs_types: list[Attribute] = []
        if len(op.function_type.outputs.data) == 1:
            output_reg = self.arch.register_type_for_type(
                op.function_type.outputs.data[0]
            ).from_index(RETURN_PASSING_REGISTER)
            outputs_types.append(output_reg)

        new_func = x86_func.FuncOp(
            op.sym_name.data,
            new_region,
            (reg_args_types + (x86.registers.RSP,), outputs_types),
            visibility=op.sym_visibility,
        )

        rewriter.replace_op(op, new_func)


@dataclass
class LowerReturnOp(RewritePattern):
    arch: Arch

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

        ret_unalloc = self.arch.register_type_for_type(return_value.type).unallocated()
        cast_op = builtin.UnrealizedConversionCastOp.get(
            (return_value,), (ret_unalloc,)
        )
        mov_op = x86.ops.DS_MovOp(
            cast_op, destination=ret_unalloc.from_index(RETURN_PASSING_REGISTER)
        )
        ret_op = x86_func.RetOp()

        rewriter.replace_op(op, [cast_op, mov_op, ret_op])


@dataclass(frozen=True)
class ConvertFuncToX86FuncPass(ModulePass):
    name = "convert-func-to-x86-func"
    arch: str | None = None

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        arch = Arch.arch_for_name(self.arch)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerFuncOp(arch),
                    LowerReturnOp(arch),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)
