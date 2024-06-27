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
        assembly_args_str: list[str] = []
        constraints: list[str] = []
        inputs: list[SSAValue | OpResult] = []
        res_types: list[Attribute] = []

        # convert assembly args to strings for correct inline asms
        for arg in op.assembly_line_args():

            if not arg:
                continue

            elif (
                isinstance(arg, OpResult)
                and isinstance(arg.type, IntRegisterType)
                and arg.op is op
            ):
                # ssa value used as output
                res_types.append(builtin.i32)
                assembly_args_str.append(f"${len(inputs) + len(res_types) - 1}")
                constraints.append("=r")

            elif isinstance(arg, SSAValue) and isinstance(arg.type, IntRegisterType):
                # ssa value used as input
                if arg.type == IntRegisterType("zero"):
                    assembly_args_str.append("x0")
                else:
                    conversion_op = UnrealizedConversionCastOp.get([arg], [builtin.i32])
                    ops_to_insert.append(conversion_op)
                    inputs.append(conversion_op.outputs[0])
                    constraints.append("r")
                    assembly_args_str.append(f"${len(inputs) + len(res_types) - 1}")

            elif isinstance(arg, IntegerAttr):
                # immediate value
                assembly_args_str.append(str(arg.value.data))

        pass

        if op.assembly_instruction_name() in custom_ops:
            # generate custom insn inline assembly instruction
            asm_string = custom_ops[op.assembly_instruction_name()].format(
                *assembly_args_str
            )

        else:
            # generate riscv inline assembly instruction
            asm_string = (
                op.assembly_instruction_name() + " " + ", ".join(assembly_args_str)
            )

        constraints_string = ",".join(constraints)

        ops_to_insert.append(
            new_op := InlineAsmOp(asm_string, constraints_string, inputs, res_types)
        )

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
    name = "convert-riscv-to-llvm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(RiscvToLLVMPattern()).rewrite_module(op)
