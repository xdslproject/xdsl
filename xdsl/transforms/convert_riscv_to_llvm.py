from xdsl.context import MLContext
from xdsl.dialects import builtin
from xdsl.dialects.builtin import IntegerAttr, UnrealizedConversionCastOp
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

# for custom ops see
# https://pulp-platform.github.io/snitch_cluster/rm/custom_instructions.html
custom_ops = {
    "dmsrc": ".insn i 0x2b, 0, 0, x0, {0}, {1}",
    "dmdst": ".insn i 0x2b, 0, 1, x0, {0}, {1}",
    "dmcpyi": ".insn i 0x2b, 0, 2, {0}, {1}, {2}",
    "dmcpy": ".insn i 0x2b, 0, 3, {0}, {1}, {2}",
    "dmstati": ".insn i 0x2b, 0, 4, {0}, {1}, {2}",
    "dmstat": ".insn i 0x2b, 0, 5, {0}, {1}, {2}",
    "dmstr": ".insn i 0x2b, 0, 6, x0, {0}, {1}",
    "dmrep": ".insn i 0x2b, 0, 7, x0, {0}, {1}",
}


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
                res_types.append(builtin.i32)
                assembly_args_str.append(f"${len(inputs) + len(res_types) - 1}")
                constraints.append("=r")

            # ssa value used as an input operand
            elif isinstance(arg, SSAValue) and isinstance(arg.type, IntRegisterType):
                if arg.type == IntRegisterType("zero"):
                    assembly_args_str.append("x0")
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

        if iname in custom_ops:
            # generate custom insn inline assembly instruction
            asm_string = custom_ops[iname].format(*assembly_args_str)

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
