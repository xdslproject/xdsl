import json
import os
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import builtin, eqsat
from xdsl.dialects.builtin import IntAttr
from xdsl.ir import Block, Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import DiagnosticException
from xdsl.utils.hints import isa


def get_node_base_cost(op: Operation, default_cost: int | None) -> int | None:
    """
    Get the base cost of an operation (without considering dependencies).
    """
    cost_attr = op.attributes.get(eqsat.EQSAT_COST_LABEL)
    if cost_attr is None:
        return default_cost
    if not isa(cost_attr, IntAttr):
        raise DiagnosticException(
            f"Unexpected value {cost_attr} for key {eqsat.EQSAT_COST_LABEL} in {op}"
        )
    return cost_attr.data


def calculate_node_total_cost(
    value: SSAValue, eclass_costs: dict[OpResult, int], default_cost: int | None
) -> int | None:
    """
    Calculate the total cost of a node: its own cost plus the costs of all its
    e-class dependencies. This is equivalent to the Rust `node_sum_cost` function.

    Uses dictionary lookup (not recursion) to get child e-class costs.
    Returns None if any dependency cost is unknown.
    """
    # Block arguments are free
    if not isinstance(value, OpResult):
        return 0

    op = value.op

    # For e-classes, return their current best known cost (None if not set)
    if isinstance(op, eqsat.AnyEClassOp):
        return eclass_costs.get(value)

    # For regular operations, compute: own cost + sum of dependent e-class costs
    node_cost = get_node_base_cost(op, default_cost)
    if node_cost is None:
        return None

    total = node_cost

    # Add costs of all operands (non-recursive, just dictionary lookup)
    for operand in op.operands:
        if isinstance(operand, OpResult) and isinstance(operand.op, eqsat.AnyEClassOp):
            # Look up the e-class cost from the dictionary
            operand_cost = eclass_costs.get(operand)
            if operand_cost is None:
                return None
        else:
            # Block argument or non-eclass operation
            operand_cost = (
                0
                if not isinstance(operand, OpResult)
                else get_node_base_cost(operand.op, default_cost) or 0
            )

        total += operand_cost

    return total


def add_eqsat_costs(block: Block, default: int | None, cost_dict: dict[str, int]):
    """
    Add costs to all operations and perform bottom-up extraction to find the minimum
    cost node in each e-class using fixed-point iteration.
    """
    # First pass: assign base costs to operations from cost_dict or default
    for op in block.ops:
        if not op.results:
            continue

        if eqsat.EQSAT_COST_LABEL in op.attributes:
            continue

        if op.name in cost_dict:
            op.attributes[eqsat.EQSAT_COST_LABEL] = IntAttr(cost_dict[op.name])
            continue

        if len(op.results) != 1:
            raise DiagnosticException(
                "Cannot compute cost of one result of operation with multiple "
                f"results: {op}"
            )

        # For non-eclass operations without explicit costs, use default
        if not isinstance(op, eqsat.AnyEClassOp):
            if default is not None:
                op.attributes[eqsat.EQSAT_COST_LABEL] = IntAttr(default)

    # Track the minimum total cost for each e-class
    eclass_costs: dict[OpResult, int] = {}

    changed = True
    while changed:
        changed = False

        # Process all e-class operations
        for op in block.ops:
            if not isinstance(op, eqsat.AnyEClassOp):
                continue

            if not op.results:
                continue

            eclass_result = op.results[0]

            # For each operand (node) in this e-class, calculate its total cost
            for idx, operand in enumerate(op.operands):
                total_cost = calculate_node_total_cost(operand, eclass_costs, default)

                # Skip if cost cannot be determined yet
                if total_cost is None:
                    continue

                # Get current best cost for this e-class (None if not set)
                current_best = eclass_costs.get(eclass_result)

                # Update if this operand has lower cost (or if no cost is set yet)
                if current_best is None or total_cost < current_best:
                    eclass_costs[eclass_result] = total_cost
                    op.min_cost_index = IntAttr(idx)
                    changed = True


@dataclass(frozen=True)
class EqsatAddCostsPass(ModulePass):
    """
    Add costs to all operations in blocks that contain eqsat.eclass ops, and perform
    bottom-up extraction to find the minimum cost node in each e-class.

    The cost of an eclass operation is determined through fixed-point iteration:
    - Each operand's total cost is calculated (own cost + dependency costs)
    - The operand with minimum total cost is selected and stored in min_cost_index

    The cost for non-eclass operations is fetched from the cost_dict or set to the
    default value. The cost is stored as an IntAttr in the EQSAT_COST_LABEL attribute.

    If the cost cannot be calculated, the default value can be provided with the
    `default` optional parameter.
    """

    name = "eqsat-add-costs"

    cost_file: str | None = field(default=None)
    "Path to JSON file of cost values"
    default: int | None = field(default=None)
    "Default cost to assign if it cannot be calculated."

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        eclass_parent_blocks = set(
            o.parent
            for o in op.walk()
            if o.parent is not None and isinstance(o, eqsat.AnyEClassOp)
        )

        cost_dict: dict[str, int] = {}

        if self.cost_file is not None:
            assert os.path.exists(self.cost_file)
            with open(self.cost_file) as file:
                cost_dict = json.load(file)

        for block in eclass_parent_blocks:
            add_eqsat_costs(block, default=self.default, cost_dict=cost_dict)
