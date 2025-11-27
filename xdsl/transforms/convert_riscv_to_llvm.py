from dataclasses import dataclass

from xdsl.context import Context
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
from xdsl.utils.exceptions import DiagnosticException


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
        # number of results produced from the inline assembly op
        # all results are considered to be XLEN long integers
        num_results: int = 0
        # keep track which results are taken from the inline assembly op and which one are "other":
        #  int      -> index into the inline asm op result list
        #  ssa val  -> use this ssa value instead (e.g. when the op "returns" the zero register)
        result_map: list[int | SSAValue] = []

        # populate assembly_args_str and constraints
        for arg in op.assembly_line_args():
            # ssa value used as an output operand
            match arg:
                case OpResult() if arg.owner is op and isinstance(
                    arg.type, IntRegisterType
                ):
                    # if we are storing to zero, we can't produce a result, so replace result by
                    # a get_register for the zero registers.
                    if arg.type.is_allocated and arg.type.index == IntAttr(0):
                        assembly_args_str.append("x0")
                        ops_to_insert.append(zero := riscv.GetRegisterOp(arg.type))
                        # map final result to an existing SSA value
                        result_map.append(zero.res)
                        continue
                    # all other registers are treated as if they were unallocated
                    # meaning we cast them to i32 and pass values to the op
                    assembly_args_str.append(f"${len(inputs) + num_results}")
                    constraints.append("=r")
                    # map final result to a result of the inline asm op
                    result_map.append(num_results)
                    num_results += 1

                case SSAValue() if isinstance(arg.type, IntRegisterType):
                    # if the input is allocated to a zero register, use that register
                    # other allocated registers are treaded as if they were unallocated
                    if arg.type.is_allocated and arg.type.index == IntAttr(0):
                        assembly_args_str.append("x0")
                    # otherwise we need to get the value from the SSA value
                    else:
                        conversion_op = UnrealizedConversionCastOp.get(
                            [arg], [builtin.i32]
                        )
                        ops_to_insert.append(conversion_op)
                        inputs.append(conversion_op.outputs[0])
                        constraints.append("rI")
                        assembly_args_str.append(f"${len(inputs) + num_results - 1}")

                case IntegerAttr():
                    # constant value used as an immediate
                    assembly_args_str.append(str(arg.value.data))

                case _:
                    raise DiagnosticException(
                        "unsupported argument for conversion to an llvm inline assembly instruction"
                    )

        # construct asm_string
        instruction_name = op.assembly_instruction_name()

        # check if the operation has a custom insn string (for compatibility reasons)
        custom_insn = op.get_trait(HasInsnRepresentation)
        if custom_insn is not None:
            # generate custom insn inline assembly instruction
            insn_str = custom_insn.get_insn(op)
            asm_string = insn_str.format(*assembly_args_str)
        else:
            # generate generic riscv inline assembly instruction
            asm_string = instruction_name + " " + ", ".join(assembly_args_str)

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
        if num_results:
            ops_to_insert.append(
                output_op := UnrealizedConversionCastOp.get(
                    new_op.results, [r.type for r in op.results]
                )
            )
            op_results = output_op.results

        # map results back using result_map
        rewriter.replace_op(
            op,
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

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RiscvToLLVMPattern(self.xlen)).rewrite_module(op)
