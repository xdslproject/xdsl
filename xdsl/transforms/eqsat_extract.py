from xdsl.context import Context
from xdsl.dialects import builtin, eqsat
from xdsl.ir import Block, Operation, OpResult
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter


def eqsat_extract(block: Block):
    ops_to_erase = set[Operation]()
    for op in reversed(block.ops):
        if isinstance(op, eqsat.EClassOp):
            if (min_cost_index := op.min_cost_index) is not None:
                min_cost_operand = op.operands[min_cost_index.data]
                has_uses = bool(op.result.uses)
                if has_uses:
                    ops_to_erase.update(
                        operand.op
                        for index, operand in enumerate(op.operands)
                        if index != min_cost_index.data
                        and isinstance(operand, OpResult)
                    )
                else:
                    ops_to_erase.update(
                        operand.op
                        for operand in op.operands
                        if isinstance(operand, OpResult)
                    )
                if isinstance(min_cost_operand, OpResult):
                    if eqsat.EQSAT_COST_LABEL in min_cost_operand.op.attributes:
                        del min_cost_operand.op.attributes[eqsat.EQSAT_COST_LABEL]
                Rewriter.replace_op(op, (), new_results=(min_cost_operand,))
            continue

        if op in ops_to_erase:
            ops_to_erase.remove(op)
            Rewriter.erase_op(op)

    assert not ops_to_erase


class EqsatExtractPass(ModulePass):
    """
    Extracts the subprogram with the lowest cost, as specified by the `min_cost_index`
    """

    name = "eqsat-extract"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        eclass_parent_blocks = set(
            o.parent
            for o in op.walk()
            if o.parent is not None and isinstance(o, eqsat.EClassOp)
        )
        for block in eclass_parent_blocks:
            eqsat_extract(block)
