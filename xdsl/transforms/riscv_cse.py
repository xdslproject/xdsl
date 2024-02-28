from dataclasses import dataclass

from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Attribute, Block, MLContext, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.traits import Pure


def cse(rewriter: Rewriter, block: Block) -> None:
    li_by_attr: dict[Attribute, riscv.LiOp] = {}
    op_by_operands_by_name: dict[str, dict[tuple[SSAValue, ...], Operation]] = {}
    for op in block.ops:
        if (
            isinstance(op, riscv.LiOp)
            and op.rd.type == riscv.IntRegisterType.unallocated()
        ):
            if op.immediate in li_by_attr:
                rewriter.replace_op(op, (), (li_by_attr[op.immediate].rd,))
            else:
                li_by_attr[op.immediate] = op
        elif op.has_trait(Pure) and not any(
            reg.is_allocated
            for operand in op.operands
            if isinstance(reg := operand, riscv.RISCVRegisterType)
        ):
            if op.name in op_by_operands_by_name:
                op_by_operands = op_by_operands_by_name[op.name]
            else:
                op_by_operands: dict[tuple[SSAValue, ...], Operation] = {}
                op_by_operands_by_name[op.name] = op_by_operands

            operands = tuple(op.operands)
            if operands in op_by_operands:
                rewriter.replace_op(op, (), op_by_operands[operands].results)
            else:
                op_by_operands[operands] = op


@dataclass(frozen=True)
class RiscvCommonSubexpressionElimination(ModulePass):
    """
    Eliminates common sub-expressions on unallocated registers per block.

    Currently, the following operations are reused:
    - `li` operations with the same attribute
    - `Pure` operations with the same operands
    """

    name = "riscv-cse"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        rewriter = Rewriter()

        for inner_op in op.walk():
            for region in inner_op.regions:
                for block in region.blocks:
                    cse(rewriter, block)
