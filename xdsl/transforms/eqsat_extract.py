from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, eqsat
from xdsl.ir import OpResult
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter


def eqsat_extract(module_op: builtin.ModuleOp, cleanup_costs: bool = True):
    eclass_ops = [op for op in module_op.walk() if isinstance(op, eqsat.AnyEClassOp)]

    while eclass_ops:
        op = eclass_ops.pop()
        if not op.result.uses:
            # Erase all operands and uses of operands
            ops_to_erase = [op] + [
                operand.owner
                for operand in op.operands
                if isinstance(operand, OpResult)
            ]
        elif (min_cost_index := op.min_cost_index) is not None:
            # Replace eclass result by operand
            operand = op.operands[min_cost_index.data]
            op.result.replace_by_if(operand, lambda use: use.operation is not op)
            # Erase eclass and all operand ops excluding min cost one
            ops_to_erase = [op] + [
                operand.owner
                for i, operand in enumerate(op.operands)
                if i != min_cost_index.data and isinstance(operand, OpResult)
            ]
            # Delete cost
            if (
                isinstance(operand, OpResult)
                and cleanup_costs
                and eqsat.EQSAT_COST_LABEL in operand.op.attributes
            ):
                del operand.op.attributes[eqsat.EQSAT_COST_LABEL]

        else:
            # Don't touch this eclass or its operands
            ops_to_erase = ()

        for op in ops_to_erase:
            Rewriter.erase_op(op)

    assert not eclass_ops


@dataclass(frozen=True)
class EqsatExtractPass(ModulePass):
    """
    Extracts the subprogram with the lowest cost, as specified by the `min_cost_index`
    """

    name = "eqsat-extract"

    cleanup_costs: bool = True

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        eqsat_extract(op)
