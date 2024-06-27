from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.builtin import IntAttr, IntegerAttr, UnrealizedConversionCastOp
from xdsl.dialects.llvm import InlineAsmOp
from xdsl.dialects.riscv import IntRegisterType, RISCVInstruction
from xdsl.ir import Attribute, Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.traits import HasInsnRepresentation


class RiscvToLLVMPattern(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: RISCVInstruction, rewriter: PatternRewriter):

        ops_to_insert: list[Operation] = []

        # inputs for the llvm inline asm op
        assembly_args_str: list[str] = []
        constraints: list[str] = []
        inputs: list[SSAValue | OpResult] = []
        res_types: list[Attribute] = []

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
                # can't convert operations that load values into allocated registers
                if arg.type.is_allocated:
                    # if we are assigning to x0, just write to nowhere and don't produce
                    # a result
                    if arg.type.index == IntAttr(0):
                        assembly_args_str.append("x0")
                        continue
                    return
                res_types.append(builtin.i32)
                assembly_args_str.append(f"${len(inputs) + len(res_types) - 1}")
                constraints.append("=r")

            # ssa value used as an input operand
            elif isinstance(arg, SSAValue) and isinstance(arg.type, IntRegisterType):
                # if the input is allocated to a register, use that register
                if arg.type.is_allocated:
                    name = arg.type.register_name
                    if arg.type.index == IntAttr(0):
                        name = "x0"
                    assembly_args_str.append(name)
                # otherwise we need to get the value from the SSA value
                else:
                    conversion_op = UnrealizedConversionCastOp.get([arg], [builtin.i32])
                    ops_to_insert.append(conversion_op)
                    inputs.append(conversion_op.outputs[0])
                    constraints.append("r")
                    assembly_args_str.append(f"${len(inputs) + len(res_types) - 1}")

            # constant value used as an immediate
            elif isinstance(arg, IntegerAttr):
                assembly_args_str.append(str(arg.value.data))

            # not supported argument
            else:
                return

        # construct asm_string
        iname = op.assembly_instruction_name()

        # check if the operation has a custom insn string (for comaptibility reasons)
        custon_insns = op.get_trait(HasInsnRepresentation)
        if custon_insns is not None:
            # generate custom insn inline assembly instruction
            insn_str = custon_insns.get_insn(op)  # pyright: ignore [generalTypeIssue]
            asm_string = insn_str.format(*assembly_args_str)

        else:
            # generate generic riscv inline assembly instruction
            asm_string = iname + " " + ", ".join(assembly_args_str)

        # construct constraints_string
        constraints_string = ",".join(constraints)

        # construct llvm inline asm op
        ops_to_insert.append(
            new_op := InlineAsmOp(asm_string, constraints_string, inputs, res_types)
        )

        # cast output back to original type if necessary
        if res_types:
            ops_to_insert.append(
                output_op := UnrealizedConversionCastOp.get(
                    new_op.results, [r.type for r in op.results]
                )
            )
            new_results = output_op.outputs
        else:
            new_results = None

        rewriter.replace_matched_op(ops_to_insert, new_results)


class ConvertRiscvToLLVMPass(ModulePass):
    """
    Convert RISC-V instructions to LLVM inline assembly.
    This allows for the use of an LLVM backend instead of direct
    RISC-V assembly generation. Additionaly, custom ops are
    implemented using .insn directives, to avoid the need for
    a custom LLVM backend.

    Only integer register types are supported.
    """

    name = "convert-riscv-to-llvm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RiscvToLLVMPattern()).rewrite_module(op)
