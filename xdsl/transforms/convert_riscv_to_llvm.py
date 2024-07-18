from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import builtin, riscv
from xdsl.dialects.builtin import IntAttr, IntegerAttr, UnrealizedConversionCastOp
from xdsl.dialects.llvm import InlineAsmOp
from xdsl.dialects.riscv import IntRegisterType, RISCVInstruction
from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import HasInsnRepresentation


@dataclass(frozen=True)
class RiscvToLLVMPattern(RewritePattern):
    xlen: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RISCVInstruction, rewriter: PatternRewriter):
        ops_to_insert: list[Operation] = []

        # inputs for the llvm inline asm op
        assembly_args_str: list[str] = []
        constraints: list[str] = []
        inputs: list[SSAValue | OpResult] = []
        num_results: int = 0
        result_map: list[int | SSAValue] = []

        # populate assembly_args_str and constraints
        for arg in op.assembly_line_args():

            # invalid argument
            if not arg:
                return

            # ssa value used as an output operand
            elif (
                isinstance(arg, OpResult)
                and isinstance(arg.type, IntRegisterType)
                and arg.op is op
            ):
                # if we are storing to zero, we can't produce a result, so replace result by
                #
                if arg.type.is_allocated and arg.type.index == IntAttr(0):
                    assembly_args_str.append("x0")
                    ops_to_insert.append(zero := riscv.GetRegisterOp(arg.type))
                    result_map.append(zero.res)
                    continue
                assembly_args_str.append(f"${len(inputs) + num_results}")
                constraints.append("=r")
                result_map.append(num_results)
                num_results += 1

            # ssa value used as an input operand
            elif isinstance(arg, SSAValue) and isinstance(arg.type, IntRegisterType):
                # if the input is allocated to a register, use that register
                if arg.type.is_allocated and arg.type.index == IntAttr(0):
                    assembly_args_str.append("x0")
                # otherwise we need to get the value from the SSA value
                else:
                    conversion_op = UnrealizedConversionCastOp.get([arg], [builtin.i32])
                    ops_to_insert.append(conversion_op)
                    inputs.append(conversion_op.outputs[0])
                    constraints.append("r")
                    assembly_args_str.append(f"${len(inputs) + num_results - 1}")

            # constant value used as an immediate
            elif isinstance(arg, IntegerAttr):
                assembly_args_str.append(str(arg.value.data))

            # not supported argument
            else:
                return

        # construct asm_string
        iname = op.assembly_instruction_name()

        # check if the operation has a custom insn string (for comaptibility reasons)
        custom_insns = op.get_trait(HasInsnRepresentation)
        if custom_insns is not None:
            # generate custom insn inline assembly instruction
            # tahnk you pyright for making my code so much better I truly appreciate your presence
            # put the forbidden ~fruit~ method in a variable with a short name:
            n = custom_insns.get_insn
            # because if the name is too long, black will force the comment to be not in the same line as the call
            # making pyright not see the comment.
            # this continues to eat my sanity every day.
            insn_str = n(op)  # pyright: ignore[reportGeneralTypeIssues]
            asm_string = insn_str.format(*assembly_args_str)

        else:
            # generate generic riscv inline assembly instruction
            asm_string = iname + " " + ", ".join(assembly_args_str)

        # construct constraints_string
        constraints_string = ",".join(constraints)

        # construct llvm inline asm op
        register_width_int = builtin.IntegerType(self.xlen)
        ops_to_insert.append(
            new_op := InlineAsmOp(
                asm_string,
                constraints_string,
                inputs,
                [register_width_int] * num_results,
            )
        )
        op_results = new_op.results

        # cast output back to original type if necessary
        if num_results > 0:
            ops_to_insert.append(
                output_op := UnrealizedConversionCastOp.get(
                    new_op.results, [r.type for r in op.results]
                )
            )
            op_results = output_op.results

        rewriter.replace_matched_op(
            ops_to_insert,
            [op_results[i] if isinstance(i, int) else i for i in result_map],
        )


class ConvertRiscvToLLVMPass(ModulePass):
    """
    Convert RISC-V instructions to LLVM inline assembly. This allows for the use
    of an LLVM backend instead of direct RISC-V assembly generation. Additionally,
    custom ops are implemented using .insn directives, to avoid the need for a
    custom LLVM backend.

    Only integer register types are supported. Specify register width through the
    xlen pass argument.

    Due to the nature of inline assembly operations, this behaviour is very flaky
    for code that has been register allocated, and will most likely break for all
    non-trivial register allocated code.

    This pass handles register allocated operations by discarding allocated registers.
    This breaks as soon as the riscv dialect code has non-SSA def-use chains (e.g.
    through get_register ops).
    """

    name = "convert-riscv-to-llvm"

    xlen: int = 32

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RiscvToLLVMPattern(self.xlen)).rewrite_module(op)
