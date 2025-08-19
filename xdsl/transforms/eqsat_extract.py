import logging
import sys
from dataclasses import dataclass

from pulp import (  # pyright: ignore[reportMissingTypeStubs]
    HiGHS_CMD,
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpStatusNotSolved,
    LpStatusOptimal,
    LpVariable,
    lpSum,
)

from xdsl.context import Context
from xdsl.dialects import builtin, eqsat
from xdsl.ir import Block, Operation, OpResult, SSAValue
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.traits import IsTerminator

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def create_var_name(val: SSAValue | Operation) -> str:
    if isinstance(val, Operation):
        if len(val.results) == 1 and (val := val.results[0]).name_hint is not None:
            return val.name_hint
    elif val.name_hint is not None:
        return val.name_hint
    elif isinstance(val, OpResult):
        val = val.owner
    else:
        return str(val)
    return str(val).split("= ")[-1]


def ilp_extract(block: Block, timeout_seconds: int | None = None):
    """
    Extract optimal DAG from e-graph using ILP with PuLP and HiGHS solver.

    Args:
        block: The block containing the e-graph operations
        timeout_seconds: Optional timeout for the solver
    """

    # Collect all EClassOps and their operands
    eclass_ops: list[eqsat.EClassOp] = []
    all_operands: list[SSAValue] = []
    operand_to_eclass = {}  # Maps operand to its containing EClassOp
    operand_to_index = {}  # Maps operand to its index in the EClassOp

    for eclass in block.ops:
        if isinstance(eclass, eqsat.EClassOp):
            eclass_ops.append(eclass)
            for idx, operand in enumerate(eclass.operands):
                all_operands.append(operand)
                operand_to_eclass[operand] = eclass
                operand_to_index[operand] = idx

    if not eclass_ops:
        return  # Nothing to extract

    # Create the ILP model
    model = LpProblem("EGraph_Extraction", LpMinimize)

    # Decision variables
    # 1. Binary variables for each operand (node selection)
    operand_vars: dict[tuple[eqsat.EClassOp, int], LpVariable] = {}
    for eclass in eclass_ops:
        for idx, operand in enumerate(eclass.operands):
            var_name = f"node_{create_var_name(eclass)}_{idx}"
            operand_vars[(eclass, idx)] = LpVariable(var_name, cat=LpBinary)

    # 2. Binary variables for class activation
    class_active: dict[eqsat.EClassOp, LpVariable] = {}
    for eclass in eclass_ops:
        var_name = f"class_active_{create_var_name(eclass)}"
        class_active[eclass] = LpVariable(var_name, cat=LpBinary)

    # 3. Level variables for cycle prevention (continuous)
    class_levels = {}
    for eclass in eclass_ops:
        var_name = f"level_{create_var_name(eclass)}"
        class_levels[eclass] = LpVariable(var_name, lowBound=0, upBound=len(eclass_ops))

    # Helper function to get the cost of an operand
    def get_operand_cost(operand: SSAValue):
        if isinstance(operand, OpResult):
            if eqsat.EQSAT_COST_LABEL in operand.op.attributes:
                # Assuming cost is stored as an integer or float attribute
                cost_attr = operand.op.attributes[eqsat.EQSAT_COST_LABEL]
                assert isinstance(cost_attr, builtin.IntAttr)
                return float(cost_attr.data)
        return 0.0  # Default cost for constants or unknown operands

    # Objective function: minimize total cost
    objective: list[tuple[LpVariable, float]] = []
    for eclass in eclass_ops:
        for idx, operand in enumerate(eclass.operands):
            cost = get_operand_cost(operand)
            if cost > 0:
                objective.append((operand_vars[(eclass, idx)], cost))

    if objective:
        model += lpSum(objective), "Total_Cost"

    # Constraints

    # 1. Class-Node Consistency: exactly one node selected per active class
    for eclass in eclass_ops:
        node_sum: list[LpVariable] = []
        for idx in range(len(eclass.operands)):
            node_sum.append(operand_vars[(eclass, idx)])

        # Sum of selected nodes equals class activation
        model += (
            lpSum(node_sum) == class_active[eclass],
            f"class_node_consistency_{create_var_name(eclass)}",
        )

    # 2. Child Dependency: if a node is selected, its children classes must be active
    for eclass in eclass_ops:
        for idx, operand in enumerate(eclass.operands):
            if not isinstance(operand, OpResult):
                continue
            children = operand.owner.operands
            handled_children: set[eqsat.EClassOp] = set()
            for child in children:
                assert isinstance(child, OpResult)
                assert isinstance(eclass_child := child.owner, eqsat.EClassOp)
                if eclass_child in handled_children:
                    continue
                handled_children.add(eclass_child)
                model += (
                    class_active[eclass_child] >= operand_vars[(eclass, idx)],
                    f"child_dep_{create_var_name(eclass)}_{idx}_{str(eclass_child)}",
                )

    # 3. Root Requirements: identify and activate root classes
    terminator = block.last_op
    assert terminator
    assert terminator.has_trait(IsTerminator), (
        "Expected last operation in the block to be a terminator."
    )
    for i, root_val in enumerate(terminator.operands):
        root_class = root_val.owner
        assert isinstance(root_class, eqsat.EClassOp)
        model += (
            class_active[root_class] >= 1,
            f"root_requirement_{create_var_name(terminator)}[{i}]",
        )

    # 4. Cycle Prevention Constraints
    # For each node that has children, enforce level ordering
    M = len(eclass_ops) + 1  # Large constant for big-M method

    for eclass in eclass_ops:
        for idx, operand in enumerate(eclass.operands):
            opposite_var = None
            if isinstance(operand, OpResult):
                handled_children = set()
                for child in operand.owner.operands:
                    assert isinstance(child, OpResult)
                    assert isinstance(child_eclass := child.owner, eqsat.EClassOp)
                    if child_eclass in handled_children:
                        continue
                    else:
                        handled_children.add(child_eclass)
                    if child_eclass == eclass:
                        model += (
                            operand_vars[(eclass, idx)] == 0,
                            f"self_loop_{create_var_name(eclass)}_{idx}",
                        )
                        continue

                    # child_eclass != eclass:
                    if opposite_var is None:
                        opposite_var = (
                            LpVariable(
                                f"opposite_{create_var_name(eclass)}_{idx}",
                                cat=LpBinary,
                            )
                            if opposite_var is None
                            else opposite_var
                        )
                        model += (
                            opposite_var + operand_vars[(eclass, idx)] == 1,
                            f"opposite_def_{create_var_name(eclass)}_{idx}",
                        )
                    # Level ordering constraint with big-M
                    # If node is active (opposite = 0), then level[op] < level[child]
                    # level[op] - level[child] + M * opposite >= 1
                    model += (
                        class_levels[eclass]
                        - class_levels[child_eclass]
                        + M * opposite_var
                        >= 1,
                        f"cycle_prevent_{create_var_name(eclass)}_{idx}_{create_var_name(child_eclass)}",
                    )

    logger.info(model)

    # Solve the model
    if timeout_seconds:
        solver = HiGHS_CMD(msg=True, timeLimit=timeout_seconds)
    else:
        solver = HiGHS_CMD(msg=True)

    status = model.solve(solver)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    logger.info("variable solutions:")
    for var in model.variables():
        logger.info(f"  {var.name}: {var.value()}")

    # Check if solution is optimal or feasible
    if status not in [LpStatusOptimal, LpStatusNotSolved]:
        if timeout_seconds:
            raise RuntimeError("Could not solve ilp problem")
        else:
            raise RuntimeError(f"ILP extraction failed with status: {LpStatus[status]}")

    # Extract solution and apply to the e-graph
    ops_to_erase: set[Operation] = set()

    for eclass in reversed(eclass_ops):
        c = class_active[eclass].value()
        assert c is not None
        if c > 0.5:  # Class is active
            # Find which node was selected
            selected_idx = None
            for idx in range(len(eclass.operands)):
                n = operand_vars[(eclass, idx)].value()
                assert n is not None
                if n > 0.5:
                    selected_idx = idx
                    break

            if selected_idx is not None:
                selected_operand = eclass.operands[selected_idx]
                has_uses = bool(eclass.result.uses)

                if has_uses:
                    # Mark non-selected operands for erasure
                    ops_to_erase.update(
                        operand.op
                        for index, operand in enumerate(eclass.operands)
                        if index != selected_idx and isinstance(operand, OpResult)
                    )
                else:
                    # If no uses, mark all operands for erasure
                    ops_to_erase.update(
                        operand.op
                        for operand in eclass.operands
                        if isinstance(operand, OpResult)
                    )

                # Clean up cost attribute from selected operand
                if isinstance(selected_operand, OpResult):
                    if eqsat.EQSAT_COST_LABEL in selected_operand.op.attributes:
                        del selected_operand.op.attributes[eqsat.EQSAT_COST_LABEL]

                # Replace the EClassOp with the selected operand
                Rewriter.replace_op(eclass, (), new_results=(selected_operand,))
        else:
            # Inactive class - mark for erasure
            ops_to_erase.add(eclass)

    # Erase unused operations
    for eclass in list(block.ops):
        if eclass in ops_to_erase:
            ops_to_erase.remove(eclass)
            Rewriter.erase_op(eclass)

    assert not ops_to_erase, "Some operations marked for erasure were not found"

    if model.objective:
        logger.info(
            f"ILP extraction completed with objective value: {model.objective.value()}"
        )
    else:
        logger.info("ILP extraction completed with objective value (no objective)")


def eqsat_extract(block: Block):
    """
    Greedy extraction fallback.
    """
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
                    assert eqsat.EQSAT_COST_LABEL in min_cost_operand.op.attributes, (
                        min_cost_operand.op
                    )
                    del min_cost_operand.op.attributes[eqsat.EQSAT_COST_LABEL]
                Rewriter.replace_op(op, (), new_results=(min_cost_operand,))
            continue

        if op in ops_to_erase:
            ops_to_erase.remove(op)
            Rewriter.erase_op(op)

    assert not ops_to_erase


@dataclass(frozen=True)
class EqsatExtractPass(ModulePass):
    """
    Extracts the subprogram with the lowest cost, as specified by the `min_cost_index`
    """

    name = "eqsat-extract"

    ilp: bool = False

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        eclass_parent_blocks = set(
            o.parent
            for o in op.walk()
            if o.parent is not None and isinstance(o, eqsat.EClassOp)
        )
        for block in eclass_parent_blocks:
            if self.ilp:
                ilp_extract(block)
            else:
                eqsat_extract(block)
