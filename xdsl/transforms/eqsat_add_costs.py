from xdsl.context import Context
from xdsl.dialects import builtin, eqsat
from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Block, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import DiagnosticException


def get_eqsat_cost(value: SSAValue) -> int | None:
    if not isinstance(value, OpResult):
        return 0
    if isinstance(value.op, eqsat.EClassOp):
        if (min_cost_index := value.op.min_cost_index) is not None:
            return get_eqsat_cost(value.op.operands[min_cost_index.data])
    cost_attribute = value.op.attributes.get(eqsat.EQSAT_COST_LABEL)
    if cost_attribute is None:
        return None
    if not isinstance(cost_attribute, IntAttr):
        raise DiagnosticException(
            f"Unexpected value {cost_attribute} for key {eqsat.EQSAT_COST_LABEL} in {value.op}"
        )
    return cost_attribute.data


def add_eqsat_costs(block: Block):
    for op in block.ops:
        if not op.results:
            # No need to annotate ops without results
            continue

        if eqsat.EQSAT_COST_LABEL in op.attributes:
            continue

        if len(op.results) != 1:
            raise DiagnosticException(
                "Cannot compute cost of one result of operation with multiple "
                f"results: {op}"
            )

        costs = tuple(get_eqsat_cost(value) for value in op.operands)
        if None in costs:
            continue

        if isinstance(op, eqsat.EClassOp):
            cost = min(cost for cost in costs if cost is not None)
            index = costs.index(cost)
            op.min_cost_index = IntAttr(index)
        else:
            # For now, set the cost to the costs of the operands + 1
            cost = sum(cost for cost in costs if cost is not None) + 1
            op.attributes[eqsat.EQSAT_COST_LABEL] = IntAttr(cost)


class EqsatAddCostsPass(ModulePass):
    """
    Add costs to all operations in blocks that contain eqsat.eclass ops.
    The cost of an eclass operation is the minimum of all the costs of the operations of
    the operands, if these are all non-`None`, and `None` otherwise.
    The cost for all other operations is currently set to the costs of all the
    operations of the operands + 1, if these are all non-`None`, and `None` otherwise.
    The cost is stored as an `IntAttr`, and cannot be computed for operations with
    multiple results.
    """

    name = "eqsat-add-costs"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        eclass_parent_blocks = set(
            o.parent
            for o in op.walk()
            if o.parent is not None and isinstance(o, eqsat.EClassOp)
        )
        for block in eclass_parent_blocks:
            add_eqsat_costs(block)
