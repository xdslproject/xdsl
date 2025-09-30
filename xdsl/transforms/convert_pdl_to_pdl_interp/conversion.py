"""
PDL to PDL_interp Transformation
"""

import sys
from abc import ABC
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Optional, cast

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import pdl, pdl_interp
from xdsl.dialects.builtin import (
    ArrayAttr,
    FunctionType,
    IntegerAttr,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
    TypeAttribute,
    UnitAttr,
)
from xdsl.ir import (
    Block,
    Operation,
    OpResult,
    Region,
    SSAValue,
)
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
    Answer,
    AttributeAnswer,
    AttributeConstraintQuestion,
    AttributeLiteralPosition,
    AttributePosition,
    ConstraintPosition,
    ConstraintQuestion,
    EqualToQuestion,
    ForEachPosition,
    IsNotNullQuestion,
    OperandCountAtLeastQuestion,
    OperandCountQuestion,
    OperandGroupPosition,
    OperandPosition,
    OperationNameQuestion,
    OperationPosition,
    Position,
    PositionalPredicate,
    Predicate,
    Question,
    ResultCountAtLeastQuestion,
    ResultCountQuestion,
    ResultGroupPosition,
    ResultPosition,
    StringAnswer,
    TrueAnswer,
    TypeAnswer,
    TypeConstraintQuestion,
    TypeLiteralPosition,
    TypePosition,
    UnsignedAnswer,
    UsersPosition,
    ValuePosition,
    get_position_cost,
    get_question_cost,
)
from xdsl.utils.scoped_dict import ScopedDict

# =============================================================================
# Matcher Tree Nodes
# =============================================================================


@dataclass
class MatcherNode(ABC):
    """Base class for matcher tree nodes"""

    position: Position | None = None
    question: Question | None = None
    failure_node: Optional["MatcherNode"] = None


@dataclass(kw_only=True)
class BoolNode(MatcherNode):
    """Boolean predicate node"""

    success_node: MatcherNode | None = None
    failure_node: MatcherNode | None = None

    answer: Answer


@dataclass
class SwitchNode(MatcherNode):
    """Multi-way switch node"""

    children: dict[Answer, MatcherNode | None] = field(default_factory=lambda: {})


@dataclass
class ChooseNode(MatcherNode):
    """Similar to a SwitchNode, but tries all choices by backtracking upon finalization."""

    choices: list[MatcherNode | None] = field(default_factory=lambda: [])


@dataclass(kw_only=True)
class SuccessNode(MatcherNode):
    """Successful pattern match"""

    pattern: pdl.PatternOp  # PDL pattern reference
    root: SSAValue | None = None  # Root value


@dataclass
class ExitNode(MatcherNode):
    """Exit/failure node"""

    pass


# =============================================================================
# Root Ordering and Cost Graph
# =============================================================================


@dataclass
class RootOrderingEntry:
    """Entry in the root ordering cost graph"""

    cost: tuple[int, int]  # (depth, tie_breaker)
    connector: SSAValue  # Value that connects the roots


class OptimalBranching:
    """Edmonds' optimal branching algorithm for minimum spanning arborescence"""

    graph: dict[SSAValue, dict[SSAValue, RootOrderingEntry]]
    root: SSAValue
    parents: dict[SSAValue, SSAValue | None]

    def __init__(
        self, graph: dict[SSAValue, dict[SSAValue, RootOrderingEntry]], root: SSAValue
    ):
        self.graph = graph
        self.root = root
        self.parents = {}

    def solve(self) -> int:
        """Solve for optimal branching, returns total cost"""
        self.parents.clear()
        self.parents[self.root] = None
        total_cost = 0
        parent_depths: dict[SSAValue, int] = {}

        nodes = list(self.graph.keys())
        for node in nodes:
            if node in self.parents:
                continue

            path: list[SSAValue] = []
            curr = node
            while curr not in self.parents:
                if curr in path:  # Cycle detected
                    cycle_start_index = path.index(curr)
                    cycle = path[cycle_start_index:]
                    self._contract_cycle(cycle)
                    # Restart solving on the contracted graph
                    return self.solve()

                path.append(curr)

                if curr not in self.graph or not self.graph[curr]:
                    # Node has no incoming edges, cannot be part of a solution unless it's the root
                    return sys.maxsize

                # Find best parent
                best_parent = min(
                    self.graph[curr].items(), key=lambda item: item[1].cost
                )
                self.parents[curr] = best_parent[0]
                cost = best_parent[1].cost[0]
                parent_depths[curr] = cost
                total_cost += cost
                curr = best_parent[0]

        return total_cost

    def _contract_cycle(self, cycle: list[SSAValue]) -> None:
        """Contract a cycle in the graph."""
        rep = cycle[0]
        cycle_set = set(cycle)

        # Cost of edges within the cycle
        cycle_costs: dict[SSAValue, int] = {}
        for node in cycle:
            parent = self.parents[node]
            assert parent is not None
            cycle_costs[node] = self.graph[node][parent].cost[0]

        # Create new graph with contracted node
        new_graph: dict[SSAValue, dict[SSAValue, RootOrderingEntry]] = {}
        new_graph[rep] = {}
        actual_targets: dict[SSAValue, SSAValue] = {}
        actual_sources: dict[SSAValue, SSAValue] = {}

        for target, sources in self.graph.items():
            if target in cycle_set:
                for source, entry in sources.items():
                    if source not in cycle_set:  # Edge entering the cycle
                        new_cost = entry.cost[0] - cycle_costs[target]
                        if (
                            source not in new_graph[rep]
                            or new_graph[rep][source].cost[0] > new_cost
                        ):
                            new_graph[rep][source] = RootOrderingEntry(
                                (new_cost, entry.cost[1]), entry.connector
                            )
                            actual_targets[source] = target
            else:  # Target is outside the cycle
                new_graph[target] = {}
                best_source_in_cycle = None
                best_entry_in_cycle = None
                for source, entry in sources.items():
                    if source not in cycle_set:
                        new_graph[target][source] = entry
                    else:  # Edge leaving the cycle
                        if (
                            best_entry_in_cycle is None
                            or best_entry_in_cycle.cost > entry.cost
                        ):
                            best_entry_in_cycle = entry
                            best_source_in_cycle = source
                if best_entry_in_cycle:
                    new_graph[target][rep] = best_entry_in_cycle
                    assert best_source_in_cycle is not None
                    actual_sources[target] = best_source_in_cycle

        self.graph = new_graph
        if self.root in cycle_set:
            self.root = rep

        # Solve recursively and expand the cycle
        sub_solver = OptimalBranching(self.graph, self.root)
        sub_solver.solve()
        self.parents = sub_solver.parents

        # Expand the cycle
        parent_of_rep = self.parents.get(rep)
        entry_node = actual_targets.get(parent_of_rep) if parent_of_rep else None

        for i, node in enumerate(cycle):
            if node == entry_node:
                self.parents[node] = parent_of_rep
            else:
                self.parents[node] = cycle[i - 1] if i > 0 else cycle[-1]

        for child, parent in list(self.parents.items()):
            if parent == rep:
                self.parents[child] = actual_sources.get(child)

    def pre_order_traversal(
        self, nodes: Sequence[SSAValue]
    ) -> list[tuple[SSAValue, SSAValue | None]]:
        """Returns the computed edges as visited in the preorder traversal."""
        children: dict[SSAValue, list[SSAValue]] = {node: [] for node in nodes}
        for node in nodes:
            if node != self.root:
                parent = self.parents.get(node)
                if parent is not None:
                    children[parent].append(node)

        result: list[tuple[SSAValue, SSAValue | None]] = []
        queue: list[tuple[SSAValue, SSAValue | None]] = [(self.root, None)]
        visited = {self.root}

        while queue:
            node, parent = queue.pop(0)
            result.append((node, parent))
            for child in sorted(children.get(node, []), key=lambda x: nodes.index(x)):
                if child not in visited:
                    visited.add(child)
                    queue.append((child, parent))
        return result


# =============================================================================
# Pattern Analysis
# =============================================================================


@dataclass(frozen=True)
class OpIndex:
    """An op accepting a value at an optional index."""

    parent: SSAValue
    index: int | None


class PatternAnalyzer:
    """Analyzes PDL patterns and extracts predicates"""

    def detect_roots(self, pattern: pdl.PatternOp) -> list[OpResult[pdl.OperationType]]:
        """Detect root operations in a pattern"""
        used = {
            operand.owner.parent_
            for operation_op in pattern.body.ops
            if isinstance(operation_op, pdl.OperationOp)
            for operand in operation_op.operand_values
            if isinstance(operand.owner, pdl.ResultOp | pdl.ResultsOp)
        }

        rewriter = pattern.body.block.last_op
        assert isinstance(rewriter, pdl.RewriteOp)
        if rewriter.root is not None:
            if rewriter.root in used:
                used.remove(rewriter.root)

        roots = [
            op.op
            for op in pattern.body.ops
            if isinstance(op, pdl.OperationOp) and op.op not in used
        ]
        return roots

    def build_cost_graph(
        self, roots: Sequence[SSAValue]
    ) -> tuple[
        dict[SSAValue, dict[SSAValue, RootOrderingEntry]],
        dict[SSAValue, dict[SSAValue, OpIndex]],
    ]:
        """Builds the cost graph for connecting candidate roots."""
        graph: dict[SSAValue, dict[SSAValue, RootOrderingEntry]] = {}
        parent_maps: dict[SSAValue, dict[SSAValue, OpIndex]] = {}
        connectors_roots_depths: dict[SSAValue, list[tuple[SSAValue, int]]] = {}

        @dataclass
        class Entry:
            value: SSAValue
            parent: SSAValue | None
            index: int | None
            depth: int

        for root in roots:
            to_visit = [Entry(root, None, None, 0)]
            parent_map: dict[SSAValue, OpIndex] = {}
            parent_maps[root] = parent_map
            visited_values: set[SSAValue] = set()

            while to_visit:
                entry = to_visit.pop(0)
                if entry.value in visited_values:
                    continue
                visited_values.add(entry.value)

                if entry.parent is not None:
                    parent_map[entry.value] = OpIndex(entry.parent, entry.index)

                if entry.value not in connectors_roots_depths:
                    connectors_roots_depths[entry.value] = []
                connectors_roots_depths[entry.value].append((root, entry.depth))

                defining_op = entry.value.owner
                if isinstance(defining_op, pdl.OperationOp):
                    operands = defining_op.operand_values
                    if len(operands) == 1 and isinstance(
                        operands[0].type, pdl.RangeType
                    ):
                        to_visit.append(
                            Entry(operands[0], entry.value, None, entry.depth + 1)
                        )
                    else:
                        for i, operand in enumerate(operands):
                            to_visit.append(
                                Entry(operand, entry.value, i, entry.depth + 1)
                            )
                elif isinstance(defining_op, pdl.ResultOp | pdl.ResultsOp):
                    to_visit.append(
                        Entry(
                            defining_op.parent_,
                            entry.value,
                            defining_op.index.value.data if defining_op.index else None,
                            entry.depth,
                        )
                    )

        next_id = 0
        for value, roots_depths in connectors_roots_depths.items():
            if len(roots_depths) <= 1:
                continue

            for source_root, _ in roots_depths:
                for target_root, depth in roots_depths:
                    if source_root == target_root:
                        continue
                    if target_root not in graph:
                        graph[target_root] = {}
                    entry = graph[target_root].get(source_root)

                    if entry is None or entry.cost[0] > depth:
                        tie_breaker = entry.cost[1] if entry else next_id
                        if entry is None:
                            next_id += 1
                        graph[target_root][source_root] = RootOrderingEntry(
                            (depth, tie_breaker), value
                        )
        return graph, parent_maps

    def _use_operand_group(self, op: pdl.OperationOp, index: int) -> bool:
        """Checks if an operand at an index should be queried as a group."""
        return any(
            isinstance(op.operand_values[i].type, pdl.RangeType)
            for i in range(index + 1)
        )

    def visit_upward(
        self,
        predicates: list[PositionalPredicate],
        op_index: OpIndex,
        inputs: dict[SSAValue, Position],
        pos: ValuePosition,
        root_id: int,
    ) -> Position:
        """Visit a node during upward traversal and generate predicates."""
        value = op_index.parent
        defining_op = value.owner

        if isinstance(defining_op, pdl.OperationOp):
            users_pos = pos.get_users(use_representative=True)
            for_each_pos = users_pos.get_for_each(root_id)
            op_pos = for_each_pos.get_passthrough_op()

            if op_index.index is None:
                # operand_pos = self.builder.get_all_operands(op_pos)
                operand_pos = op_pos.get_all_operands()
            elif self._use_operand_group(defining_op, op_index.index):
                is_variadic = isinstance(
                    defining_op.operand_values[op_index.index].type, pdl.RangeType
                )
                operand_pos = op_pos.get_operand_group(op_index.index, is_variadic)
            else:
                operand_pos = op_pos.get_operand(op_index.index)

            equal_to = Predicate.get_equal_to(pos)
            predicates.append(PositionalPredicate(equal_to.q, equal_to.a, operand_pos))

            assert value not in inputs, "Duplicate upward visit"
            inputs[value] = op_pos

            predicates.extend(
                self.extract_tree_predicates(
                    value, op_pos, inputs, ignore_operand=op_index.index
                )
            )
            return op_pos

        elif isinstance(defining_op, pdl.ResultOp | pdl.ResultsOp):
            assert isinstance(pos, OperationPosition)
            if isinstance(defining_op, pdl.ResultOp):
                assert op_index.index is not None
                new_pos = pos.get_result(op_index.index)
            else:
                is_variadic = isinstance(value.type, pdl.RangeType)
                new_pos = pos.get_result_group(op_index.index, is_variadic)
            inputs.setdefault(value, new_pos)
            return new_pos

        raise TypeError(f"Unexpected op type in upward traversal: {type(defining_op)}")

    def extract_tree_predicates(
        self,
        value: SSAValue,
        position: Position,
        inputs: dict[SSAValue, Position],
        ignore_operand: int | None = None,
    ) -> list[PositionalPredicate]:
        """Extract predicates by walking the operation tree"""
        predicates: list[PositionalPredicate] = []

        # Check if this value has been visited before
        existing_pos = inputs.get(value)
        if existing_pos is not None:
            # If this is an input value that has been visited in the tree,
            # add a constraint to ensure both instances refer to the same value
            defining_op = value.owner
            if isinstance(
                defining_op,
                pdl.AttributeOp
                | pdl.OperandOp
                | pdl.OperandsOp
                | pdl.OperationOp
                | pdl.TypeOp
                | pdl.TypesOp,
            ):
                # Order positions by depth (deeper position gets the equality predicate)
                if position.get_operation_depth() > existing_pos.get_operation_depth():
                    deeper_pos, shallower_pos = position, existing_pos
                else:
                    deeper_pos, shallower_pos = existing_pos, position

                equal_pred = Predicate.get_equal_to(shallower_pos)
                predicates.append(
                    PositionalPredicate(
                        q=equal_pred.q, a=equal_pred.a, position=deeper_pos
                    )
                )
            return predicates

        inputs[value] = position

        # Dispatch based on position type (not value type!)
        match position:
            case AttributePosition():
                assert isinstance(value, OpResult)
                predicates.extend(
                    self._extract_attribute_predicates(value.owner, position, inputs)
                )
            case OperationPosition():
                assert isinstance(value, OpResult)
                predicates.extend(
                    self._extract_operation_predicates(
                        value.owner, position, inputs, ignore_operand
                    )
                )
            case TypePosition():
                assert isinstance(value, OpResult)
                predicates.extend(
                    self._extract_type_predicates(value.owner, position, inputs)
                )
            case OperandPosition() | OperandGroupPosition():
                assert isinstance(value, SSAValue)
                predicates.extend(
                    self._extract_operand_tree_predicates(value, position, inputs)
                )
            case _:
                raise TypeError(f"Unexpected position kind: {type(position)}")

        return predicates

    def _get_num_non_range_values(self, values: Sequence[SSAValue]) -> int:
        """Returns the number of non-range elements within values"""
        return sum(1 for v in values if not isinstance(v.type, pdl.RangeType))

    def _extract_attribute_predicates(
        self,
        attr_op: Operation,
        attr_pos: AttributePosition,
        inputs: dict[SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates for an attribute"""
        predicates: list[PositionalPredicate] = []

        is_not_null = Predicate.get_is_not_null()
        predicates.append(
            PositionalPredicate(q=is_not_null.q, a=is_not_null.a, position=attr_pos)
        )

        if isinstance(attr_op, pdl.AttributeOp):
            if attr_op.value_type:
                type_pos = attr_pos.get_type()
                predicates.extend(
                    self.extract_tree_predicates(attr_op.value_type, type_pos, inputs)
                )

            elif attr_op.value:
                attr_constraint = Predicate.get_attribute_constraint(attr_op.value)
                predicates.append(
                    PositionalPredicate(
                        q=attr_constraint.q, a=attr_constraint.a, position=attr_pos
                    )
                )

        return predicates

    def _extract_operation_predicates(
        self,
        op_op: Operation,
        op_pos: OperationPosition,
        inputs: dict[SSAValue, Position],
        ignore_operand: int | None = None,
    ) -> list[PositionalPredicate]:
        """Extract predicates for an operation"""
        predicates: list[PositionalPredicate] = []

        if not isinstance(op_op, pdl.OperationOp):
            return predicates

        if not op_pos.is_root():
            is_not_null = Predicate.get_is_not_null()
            predicates.append(
                PositionalPredicate(q=is_not_null.q, a=is_not_null.a, position=op_pos)
            )

        # Operation name check
        if op_op.opName:
            op_name = op_op.opName.data
            op_name_pred = Predicate.get_operation_name(op_name)
            predicates.append(
                PositionalPredicate(q=op_name_pred.q, a=op_name_pred.a, position=op_pos)
            )

        operands = op_op.operand_values
        min_operands = self._get_num_non_range_values(operands)
        if min_operands != len(operands):
            # Has variadic operands - check minimum
            if min_operands > 0:
                operand_count_pred = Predicate.get_operand_count_at_least(min_operands)
                predicates.append(
                    PositionalPredicate(
                        q=operand_count_pred.q, a=operand_count_pred.a, position=op_pos
                    )
                )
        else:
            # All non-variadic - check exact count
            operand_count_pred = Predicate.get_operand_count(min_operands)
            predicates.append(
                PositionalPredicate(
                    q=operand_count_pred.q, a=operand_count_pred.a, position=op_pos
                )
            )

        types = op_op.type_values
        min_results = self._get_num_non_range_values(types)
        if min_results == len(types):
            # All non-variadic - check exact count
            result_count_pred = Predicate.get_result_count(len(types))
            predicates.append(
                PositionalPredicate(
                    q=result_count_pred.q, a=result_count_pred.a, position=op_pos
                )
            )
        elif min_results > 0:
            # Has variadic results - check minimum
            result_count_pred = Predicate.get_result_count_at_least(min_results)
            predicates.append(
                PositionalPredicate(
                    q=result_count_pred.q, a=result_count_pred.a, position=op_pos
                )
            )

        # Process attributes
        for attr_name, attr in zip(
            op_op.attributeValueNames, op_op.attribute_values, strict=True
        ):
            attr_pos = op_pos.get_attribute(attr_name.data)
            predicates.extend(self.extract_tree_predicates(attr, attr_pos, inputs))

        if len(operands) == 1 and isinstance(operands[0].type, pdl.RangeType):
            # Special case: single variadic operand represents all operands
            if op_pos.is_root() or op_pos.is_operand_defining_op():
                all_operands_pos = op_pos.get_all_operands()
                predicates.extend(
                    self.extract_tree_predicates(operands[0], all_operands_pos, inputs)
                )
        else:
            # Process individual operands
            found_variable_length = False
            for i, operand in enumerate(operands):
                is_variadic = isinstance(operand.type, pdl.RangeType)
                found_variable_length = found_variable_length or is_variadic

                if ignore_operand is not None and i == ignore_operand:
                    continue

                # Switch to group-based positioning after first variadic
                if found_variable_length:
                    operand_pos = op_pos.get_operand_group(i, is_variadic)
                else:
                    operand_pos = op_pos.get_operand(i)

                predicates.extend(
                    self.extract_tree_predicates(operand, operand_pos, inputs)
                )

        if len(types) == 1 and isinstance(types[0].type, pdl.RangeType):
            # Single variadic result represents all results
            all_results_pos = op_pos.get_all_results()
            type_pos = all_results_pos.get_type()
            predicates.extend(self.extract_tree_predicates(types[0], type_pos, inputs))
        else:
            # Process individual results
            found_variable_length = False
            for i, type_value in enumerate(types):
                is_variadic = isinstance(type_value.type, pdl.RangeType)
                found_variable_length = found_variable_length or is_variadic

                # Switch to group-based positioning after first variadic
                if found_variable_length:
                    result_pos = op_pos.get_result_group(i, is_variadic)
                else:
                    result_pos = op_pos.get_result(i)

                # Add not-null check for each result
                is_not_null = Predicate.get_is_not_null()
                predicates.append(
                    PositionalPredicate(
                        q=is_not_null.q, a=is_not_null.a, position=result_pos
                    )
                )

                # Process the result type
                type_pos = result_pos.get_type()
                predicates.extend(
                    self.extract_tree_predicates(type_value, type_pos, inputs)
                )

        return predicates

    def _extract_operand_tree_predicates(
        self,
        operand_value: SSAValue,
        operand_pos: OperandPosition | OperandGroupPosition,
        inputs: dict[SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates for an operand or operand group"""
        predicates: list[PositionalPredicate] = []

        # Get the defining operation
        defining_op = operand_value.owner
        is_variadic = isinstance(operand_value.type, pdl.RangeType)

        match defining_op:
            case pdl.OperandOp() | pdl.OperandsOp():
                match defining_op:
                    case pdl.OperandOp():
                        is_not_null = Predicate.get_is_not_null()
                        predicates.append(
                            PositionalPredicate(
                                q=is_not_null.q, a=is_not_null.a, position=operand_pos
                            )
                        )
                    case pdl.OperandsOp() if (
                        isinstance(operand_pos, OperandGroupPosition)
                        and operand_pos.group_number is not None
                    ):
                        is_not_null = Predicate.get_is_not_null()
                        predicates.append(
                            PositionalPredicate(
                                q=is_not_null.q, a=is_not_null.a, position=operand_pos
                            )
                        )
                    case _:
                        pass

                if defining_op.value_type:
                    type_pos = operand_pos.get_type()
                    predicates.extend(
                        self.extract_tree_predicates(
                            defining_op.value_type, type_pos, inputs
                        )
                    )

            case pdl.ResultOp() | pdl.ResultsOp():
                index_attr = defining_op.index
                index = index_attr.value.data if index_attr is not None else None

                if index is not None:
                    is_not_null = Predicate.get_is_not_null()
                    predicates.append(
                        PositionalPredicate(
                            q=is_not_null.q, a=is_not_null.a, position=operand_pos
                        )
                    )

                # Get the parent operation position
                parent_op = defining_op.parent_
                defining_op_pos = operand_pos.get_defining_op()

                # Parent operation should not be null
                is_not_null = Predicate.get_is_not_null()
                predicates.append(
                    PositionalPredicate(
                        q=is_not_null.q, a=is_not_null.a, position=defining_op_pos
                    )
                )

                match defining_op:
                    case pdl.ResultOp():
                        result_pos = defining_op_pos.get_result(
                            index if index is not None else 0
                        )
                    case pdl.ResultsOp():  # ResultsOp
                        result_pos = defining_op_pos.get_result_group(
                            index, is_variadic
                        )

                equal_to = Predicate.get_equal_to(operand_pos)
                predicates.append(
                    PositionalPredicate(q=equal_to.q, a=equal_to.a, position=result_pos)
                )

                # Recursively process the parent operation
                predicates.extend(
                    self.extract_tree_predicates(parent_op, defining_op_pos, inputs)
                )
            case _:
                pass

        return predicates

    def _extract_type_predicates(
        self,
        type_op: Operation,
        type_pos: TypePosition,
        inputs: dict[SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates for a type"""
        predicates: list[PositionalPredicate] = []

        match type_op:
            case pdl.TypeOp(constantType=const_type) if const_type:
                type_constraint = Predicate.get_type_constraint(const_type)
                predicates.append(
                    PositionalPredicate(
                        q=type_constraint.q, a=type_constraint.a, position=type_pos
                    )
                )
            case pdl.TypesOp(constantTypes=const_types) if const_types:
                type_constraint = Predicate.get_type_constraint(const_types)
                predicates.append(
                    PositionalPredicate(
                        q=type_constraint.q, a=type_constraint.a, position=type_pos
                    )
                )
            case _:
                pass

        return predicates

    def extract_non_tree_predicates(
        self,
        pattern: pdl.PatternOp,
        inputs: dict[SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates that cannot be determined via tree walking"""
        predicates: list[PositionalPredicate] = []

        for op in pattern.body.ops:
            match op:
                case pdl.AttributeOp():
                    if op.output not in inputs:
                        if op.value:
                            # Create literal position for constant attribute
                            attr_pos = AttributeLiteralPosition(
                                value=op.value, parent=None
                            )
                            inputs[op.output] = attr_pos

                case pdl.ApplyNativeConstraintOp():
                    # Collect all argument positions
                    arg_positions = tuple(inputs.get(arg) for arg in op.args)
                    for pos in arg_positions:
                        assert pos is not None
                    arg_positions = cast(tuple[Position, ...], arg_positions)

                    # Find the furthest position (deepest)
                    furthest_pos = max(
                        arg_positions, key=lambda p: p.get_operation_depth() if p else 0
                    )

                    # Create the constraint predicate
                    result_types = tuple(r.type for r in op.res)
                    is_negated = bool(op.is_negated.value.data)
                    constraint_pred = Predicate.get_constraint(
                        op.constraint_name.data, arg_positions, result_types, is_negated
                    )

                    # Register positions for constraint results
                    for i, result in enumerate(op.results):
                        assert isinstance(constraint_pred.q, ConstraintQuestion)
                        constraint_pos = ConstraintPosition.get_constraint(
                            constraint_pred.q, i
                        )
                        existing = inputs.get(result)
                        if existing:
                            # Add equality constraint if result already has a position
                            deeper, shallower = (
                                (constraint_pos, existing)
                                if furthest_pos.get_operation_depth()
                                > existing.get_operation_depth()
                                else (existing, constraint_pos)
                            )
                            eq_pred = Predicate.get_equal_to(shallower)
                            predicates.append(
                                PositionalPredicate(
                                    q=eq_pred.q, a=eq_pred.a, position=deeper
                                )
                            )
                        else:
                            inputs[result] = constraint_pos

                    predicates.append(
                        PositionalPredicate(
                            q=constraint_pred.q,
                            a=constraint_pred.a,
                            position=furthest_pos,
                        )
                    )

                case pdl.ResultOp():
                    # Ensure result exists
                    if op.val not in inputs:
                        assert isinstance(op.parent_.owner, pdl.OperationOp)
                        parent_pos = inputs.get(op.parent_.owner.op)
                        if parent_pos and isinstance(parent_pos, OperationPosition):
                            result_pos = parent_pos.get_result(op.index.value.data)
                            is_not_null = Predicate.get_is_not_null()
                            predicates.append(
                                PositionalPredicate(
                                    q=is_not_null.q,
                                    a=is_not_null.a,
                                    position=result_pos,
                                )
                            )

                case pdl.ResultsOp():
                    # Handle result groups
                    if op.val not in inputs:
                        assert isinstance(op.parent_.owner, pdl.OperationOp)
                        parent_pos = inputs.get(op.parent_.owner.op)
                        if parent_pos and isinstance(parent_pos, OperationPosition):
                            is_variadic = isinstance(op.val.type, pdl.RangeType)
                            index = op.index.value.data if op.index else None
                            result_pos = parent_pos.get_result_group(index, is_variadic)
                            if index is not None:
                                is_not_null = Predicate.get_is_not_null()
                                predicates.append(
                                    PositionalPredicate(
                                        q=is_not_null.q,
                                        a=is_not_null.a,
                                        position=result_pos,
                                    )
                                )

                case pdl.TypeOp():
                    # Handle constant types
                    if op.result not in inputs and op.constantType:
                        type_pos = TypeLiteralPosition.get_type_literal(
                            value=op.constantType
                        )
                        inputs[op.result] = type_pos

                case pdl.TypesOp():
                    # Handle constant type arrays
                    if op.result not in inputs and op.constantTypes:
                        type_pos = TypeLiteralPosition.get_type_literal(
                            value=op.constantTypes
                        )
                        inputs[op.result] = type_pos

                case _:
                    pass

        return predicates


# =============================================================================
# Predicate Ordering and Tree Construction
# =============================================================================


@dataclass
class OrderedPredicate:
    """Predicate with ordering information"""

    position: Position
    question: Question
    primary_score: int = 0  # Frequency across patterns
    secondary_score: int = 0  # Squared sum within patterns
    tie_breaker: int = 0  # Insertion order
    pattern_answers: dict[pdl.PatternOp, Answer] = field(default_factory=lambda: {})

    def __lt__(self, other: "OrderedPredicate") -> bool:
        """Comparison for priority ordering"""
        return (
            self.primary_score,
            self.secondary_score,
            -self.position.get_operation_depth(),  # Prefer lower depth
            -get_position_cost(self.position),  # Position dependency
            -get_question_cost(self.question),  # Predicate dependency
            -self.tie_breaker,  # Deterministic order
        ) > (
            other.primary_score,
            other.secondary_score,
            -other.position.get_operation_depth(),
            -get_position_cost(other.position),
            -get_question_cost(other.question),
            -other.tie_breaker,
        )

    def __hash__(self):
        """The hash is based on the immutable identity of the predicate."""
        return hash((self.position, self.question))


def _depends_on(pred_a: OrderedPredicate, pred_b: OrderedPredicate) -> bool:
    """Returns true if predicate 'b' depends on a result of predicate 'a'."""
    constraint_q_a = pred_a.question
    if not isinstance(constraint_q_a, ConstraintQuestion):
        return False

    def position_depends_on_a(pos: Position) -> bool:
        if isinstance(pos, ConstraintPosition):
            return pos.constraint == constraint_q_a
        return False

    if isinstance(pred_b.question, ConstraintQuestion):
        # Does any argument of b use a?
        return any(position_depends_on_a(arg) for arg in pred_b.question.arg_positions)
    if isinstance(pred_b.question, EqualToQuestion):
        return position_depends_on_a(pred_b.position) or position_depends_on_a(
            pred_b.question.other_position
        )
    return position_depends_on_a(pred_b.position)


def _stable_topological_sort(
    predicates: list[OrderedPredicate],
) -> list[OrderedPredicate]:
    """Sorts predicates topologically while maintaining stability for independent items."""
    # Build dependency graph
    dependencies: dict[OrderedPredicate, set[OrderedPredicate]] = {
        p: set() for p in predicates
    }
    for i, pred_b in enumerate(predicates):
        for j in range(i + 1, len(predicates)):
            pred_a = predicates[j]
            if _depends_on(pred_a, pred_b):
                dependencies[pred_b].add(pred_a)  # b depends on a

    sorted_list: list[OrderedPredicate] = []
    pred_list: list[OrderedPredicate] = predicates[:]

    while pred_list:
        # Find all items with no dependencies within the current list
        to_sort = [
            p for p in pred_list if all(dep not in pred_list for dep in dependencies[p])
        ]
        # It is not possible to have cycles in the dependency graph
        # because predicates can only depend on predicates containing
        # a ConstraintQuestion. It is however impossible to construct
        # a cycle with those because they are frozen at construction
        # and thus cannot refer to a ConstraintPosition that was
        # defined later.
        assert to_sort, "Encountered a cycle!"

        # Append them to the sorted list
        sorted_list.extend(to_sort)

        # Remove them from the list to be sorted
        pred_list = [p for p in pred_list if p not in to_sort]

    return sorted_list


@dataclass
class GroupedPredicates:
    position: OperationPosition
    predicates: list[OrderedPredicate] = field(default_factory=lambda: [])
    primary_score: int = 0
    secondary_score: int = 0

    def sort_key(self):
        return (
            -self.position.get_operation_depth(),
            self.primary_score,
            self.secondary_score,
        )

    def __lt__(self, other: "GroupedPredicates") -> bool:
        return (
            -self.position.get_operation_depth(),
            self.primary_score,
            self.secondary_score,
        ) > (
            -other.position.get_operation_depth(),
            other.primary_score,
            other.secondary_score,
        )


@dataclass
class PredicateSplit:
    splits: list[list["OrderedPredicate | PredicateSplit"]] = field(
        default_factory=lambda: []
    )


@dataclass
class OperationPositionTree:
    """Node in the tree representing an OperationPosition."""

    operation: OperationPosition
    covered_patterns: set[int] = field(default_factory=lambda: set())
    children: list["OperationPositionTree"] = field(default_factory=lambda: [])

    def pretty_print(self, indent: int = 0) -> None:
        """Pretty print the tree."""
        prefix = "  " * indent
        patterns = (
            f" -> {sorted(self.covered_patterns)}" if self.covered_patterns else ""
        )
        print(f"{prefix}{repr(self.operation)}{patterns}")
        for child in self.children:
            child.pretty_print(indent + 1)


def build_operation_position_tree(
    pattern_predicates: list[list[PositionalPredicate]],
) -> tuple[OperationPositionTree, list[list[int]]]:
    """
    Build a tree representing all operation positions from multiple patterns.

    Args:
        pattern_predicates: List of predicate lists, one per pattern

    Returns:
        Root of the operation position tree
    """
    # Extract operation positions for each pattern
    pattern_operations: list[set[OperationPosition]] = []

    for predicates in pattern_predicates:
        operations: set[OperationPosition] = set()
        for pred in predicates:
            op = pred.position.get_base_operation()
            # Add this operation and all ancestors
            while op:
                operations.add(op)
                if op.parent:
                    parent_op = op.parent.get_base_operation()
                    if parent_op:
                        op = parent_op
                    else:
                        break
                else:
                    break
        pattern_operations.append(operations)

    # Find root operation
    all_ops = set[OperationPosition]().union(*pattern_operations)
    roots = [op for op in all_ops if op.is_root()]
    if len(roots) != 1:
        raise ValueError(f"Did not find exactly one root operation, found: {roots}")

    root = OperationPositionTree(operation=roots[0])

    pattern_paths: list[list[int]] = [[] for _ in pattern_operations]

    # Build tree recursively
    def build_subtree(
        node: OperationPositionTree,
        prefix: set[OperationPosition],
        remaining_indices: list[int],
        current_paths: dict[int, list[int]],
    ):
        if not remaining_indices:
            return

        # Split patterns into covered and remaining
        covered: list[int] = []
        still_needed: list[int] = []
        for i in remaining_indices:
            uncovered = pattern_operations[i] - prefix
            if not uncovered:
                covered.append(i)
            else:
                still_needed.append(i)

        node.covered_patterns.update(covered)

        if not still_needed:
            return

        # Group patterns by next operation
        next_ops: dict[OperationPosition, list[int]] = defaultdict(list)
        for i in still_needed:
            candidates = pattern_operations[i] - prefix
            if candidates:
                # Pick operation with highest score (appears in most patterns, shallow depth)
                best_op = max(
                    candidates,
                    key=lambda op: (
                        sum(1 for j in still_needed if op in pattern_operations[j]),
                        -op.get_operation_depth(),
                    ),
                )
                next_ops[best_op].append(i)

        # Create children
        child_index = 0
        for op, indices in next_ops.items():
            child = OperationPositionTree(operation=op)
            node.children.append(child)

            child_paths: dict[int, list[int]] = {}
            for idx in indices:
                child_paths[idx] = current_paths.get(idx, []) + [child_index]
                pattern_paths[idx] = child_paths[idx]
            build_subtree(child, prefix | {op}, indices, child_paths)
            child_index += 1

    build_subtree(root, {roots[0]}, list(range(len(pattern_operations))), {})
    return root, pattern_paths


class PredicateTreeBuilder:
    """Builds optimized predicate matching trees"""

    analyzer: PatternAnalyzer
    _pattern_roots: dict[pdl.PatternOp, SSAValue]
    pattern_value_positions: dict[pdl.PatternOp, dict[SSAValue, Position]]
    optimize_for_eqsat: bool

    def __init__(self, optimize_for_eqsat: bool = False):
        self.analyzer = PatternAnalyzer()
        self._pattern_roots: dict[pdl.PatternOp, SSAValue] = {}
        self.pattern_value_positions: dict[pdl.PatternOp, dict[SSAValue, Position]] = {}
        self.optimize_for_eqsat = optimize_for_eqsat

    def _build_operation_groups(self, predicates: Collection[OrderedPredicate]):
        pos_to_group: dict[OperationPosition, GroupedPredicates] = {}

        for pred in predicates:
            op_pos = pred.position.get_base_operation()
            group = pos_to_group.setdefault(op_pos, GroupedPredicates(position=op_pos))
            group.predicates.append(pred)
            group.primary_score += pred.primary_score
            group.secondary_score += pred.secondary_score

        return pos_to_group

    def _sort_grouped(self, ordered_predicates: Collection[OrderedPredicate]):
        pos_to_group = self._build_operation_groups(ordered_predicates)
        sorted_predicates: list[GroupedPredicates] = []
        seen_positions: set[OperationPosition] = set()
        for group in sorted(pos_to_group.values()):
            seen_positions.add(group.position)
            to_delete: list[int] = []
            for i, pred in enumerate(group.predicates):
                if isinstance(q := pred.question, EqualToQuestion):
                    if q.other_position.get_base_operation() in seen_positions:
                        continue
                    # If an EqualQuestion refers to a position that comes later, move it to the later group.
                    new_q = EqualToQuestion(other_position=pred.position)
                    pred.position = q.other_position
                    pred.question = new_q
                    to_delete.append(i)
                    pos_to_group[pred.position.get_base_operation()].predicates.append(
                        pred
                    )
            for i in reversed(to_delete):
                del group.predicates[i]
            group.predicates.sort()
            sorted_predicates.append(group)
        return sorted_predicates

    def build_predicate_tree(self, patterns: list[pdl.PatternOp]) -> MatcherNode:
        """Build optimized matcher tree from multiple patterns"""

        # Extract predicates for all patterns
        all_pattern_predicates: list[
            tuple[pdl.PatternOp, list[PositionalPredicate]]
        ] = []
        for pattern in patterns:
            predicates, root, inputs = self._extract_pattern_predicates(pattern)
            all_pattern_predicates.append((pattern, predicates))
            self._pattern_roots[pattern] = root
            self.pattern_value_positions[pattern] = inputs

        # Create ordered predicates with frequency analysis
        ordered_predicates = self._create_ordered_predicates(all_pattern_predicates)
        if self.optimize_for_eqsat:
            # TODO: take into account extra position dependency in ConstraintQuestion and EqualToQuestion
            op_pos_tree, pattern_paths = build_operation_position_tree(
                [predicates for (_, predicates) in all_pattern_predicates]
            )
            pos_to_pred_group = self._build_operation_groups(
                ordered_predicates.values()
            )

            sorted_predicates = sorted(
                pos_to_pred_group[op_pos_tree.operation].predicates
            )
            sorted_predicates = cast(
                list[OrderedPredicate | PredicateSplit], sorted_predicates
            )
            worklist: list[
                tuple[
                    list[OrderedPredicate | PredicateSplit], list[OperationPositionTree]
                ]
            ] = [(sorted_predicates, op_pos_tree.children)]
            while worklist:
                preds, children = worklist.pop()
                preds.append(
                    PredicateSplit(
                        [
                            cast(
                                list[OrderedPredicate | PredicateSplit],
                                pos_to_pred_group[child.operation].predicates,
                            )
                            for child in children
                        ]
                    )
                )

                for child in children:
                    worklist.append(
                        (
                            cast(
                                list[OrderedPredicate | PredicateSplit],
                                sorted(pos_to_pred_group[child.operation].predicates),
                            ),
                            child.children,
                        )
                    )

            root_node = None
            for (pattern, predicates), path in zip(
                all_pattern_predicates, pattern_paths
            ):
                pattern_predicate_set = {
                    (pred.position, pred.q): pred for pred in predicates
                }
                root_node = self._propagate_pattern(
                    root_node,
                    pattern,
                    pattern_predicate_set,
                    sorted_predicates,
                    0,
                    path,
                )
        else:
            # Sort predicates by priority
            sorted_predicates = sorted(ordered_predicates.values())
            sorted_predicates = _stable_topological_sort(sorted_predicates)
            sorted_predicates = cast(
                list[OrderedPredicate | PredicateSplit], sorted_predicates
            )

            # Build matcher tree by propagating patterns
            root_node = None
            for pattern, predicates in all_pattern_predicates:
                pattern_predicate_set = {
                    (pred.position, pred.q): pred for pred in predicates
                }
                root_node = self._propagate_pattern(
                    root_node, pattern, pattern_predicate_set, sorted_predicates, 0
                )

        # Add exit node and optimize
        if root_node is not None:
            root_node = self._optimize_tree(root_node)
            root_node = self._insert_exit_node(root_node)
            return root_node
        else:
            # Return a default exit node if no patterns were processed
            return ExitNode()

    def _extract_pattern_predicates(
        self, pattern: pdl.PatternOp
    ) -> tuple[list[PositionalPredicate], SSAValue, dict[SSAValue, Position]]:
        """Extract all predicates for a single pattern"""
        predicates: list[PositionalPredicate] = []
        inputs: dict[SSAValue, Position] = {}

        roots = self.analyzer.detect_roots(pattern)

        rewriter = pattern.body.block.last_op
        assert isinstance(rewriter, pdl.RewriteOp)
        explicit_root = rewriter.root

        graph, parent_maps = self.analyzer.build_cost_graph(roots)

        best_root = explicit_root
        best_edges = None

        if best_root is None:
            best_cost = sys.maxsize
            for root_candidate in roots:
                solver = OptimalBranching(graph, root_candidate)
                cost = solver.solve()
                if cost < best_cost:
                    best_cost = cost
                    best_root = root_candidate
                    best_edges = solver.pre_order_traversal(roots)
        else:
            solver = OptimalBranching(graph, best_root)
            solver.solve()
            best_edges = solver.pre_order_traversal(roots)

        assert best_root is not None
        assert best_edges is not None

        # Downward traversal from the best root
        root_pos = OperationPosition(depth=0)
        predicates.extend(
            self.analyzer.extract_tree_predicates(best_root, root_pos, inputs)
        )

        # Upward traversal for other connected roots
        for i, (target, source) in enumerate(best_edges):
            if target in inputs:
                continue

            assert source is not None
            connector = graph[target][source].connector
            assert connector in inputs, "Connector not yet traversed"
            pos = inputs[connector]

            path_to_target: list[OpIndex] = []
            curr = connector
            while curr != target:
                op_index = parent_maps[target][curr]
                path_to_target.append(op_index)
                curr = op_index.parent

            for op_index in reversed(path_to_target):
                if not isinstance(pos, ValuePosition):
                    raise ValueError(
                        f"Expected ValuePosition for upward traversal but got {type(pos)}"
                    )
                pos = self.analyzer.visit_upward(predicates, op_index, inputs, pos, i)

        predicates.extend(self.analyzer.extract_non_tree_predicates(pattern, inputs))
        return predicates, best_root, inputs

    def _create_ordered_predicates(
        self,
        all_pattern_predicates: list[tuple[pdl.PatternOp, list[PositionalPredicate]]],
    ) -> dict[tuple[Position, Question], OrderedPredicate]:
        """Create ordered predicates with frequency analysis"""
        predicate_map: dict[tuple[Position, Question], OrderedPredicate] = {}
        tie_breaker = 0

        # Collect unique predicates
        for pattern, predicates in all_pattern_predicates:
            for pred in predicates:
                key = (pred.position, pred.q)

                if key not in predicate_map:
                    ordered_pred = OrderedPredicate(
                        position=pred.position,
                        question=pred.q,
                        tie_breaker=tie_breaker,
                    )
                    predicate_map[key] = ordered_pred
                    tie_breaker += 1

                # Track pattern answers and increment frequency
                predicate_map[key].pattern_answers[pattern] = pred.a
                predicate_map[key].primary_score += 1

        # Calculate secondary scores
        for pattern, predicates in all_pattern_predicates:
            pattern_primary_sum = 0
            seen_keys: set[tuple[Position, Question]] = (
                set()
            )  # Track unique keys per pattern

            # First pass: collect unique predicates for this pattern
            for pred in predicates:
                key = (pred.position, pred.q)
                if key not in seen_keys:
                    seen_keys.add(key)
                    ordered_pred = predicate_map[key]
                    pattern_primary_sum += ordered_pred.primary_score**2

            # Second pass: add secondary score to each unique predicate
            for key in seen_keys:
                ordered_pred = predicate_map[key]
                ordered_pred.secondary_score += pattern_primary_sum

        return predicate_map

    def _propagate_pattern(
        self,
        node: MatcherNode | None,
        pattern: pdl.PatternOp,
        pattern_predicates: dict[tuple[Position, Question], PositionalPredicate],
        sorted_predicates: list[OrderedPredicate | PredicateSplit],
        predicate_index: int,
        path: list[int] = [],
    ) -> MatcherNode:
        """Propagate a pattern through the predicate tree"""

        # Base case: reached end of predicates
        if predicate_index >= len(sorted_predicates):
            root_val = self._pattern_roots.get(pattern)
            return SuccessNode(pattern=pattern, root=root_val, failure_node=node)
        current_predicate = sorted_predicates[predicate_index]

        if isinstance(current_predicate, PredicateSplit):
            if not path:
                root_val = self._pattern_roots.get(pattern)
                return SuccessNode(pattern=pattern, root=root_val, failure_node=node)
            choice = path[0]
            path = path[1:]
            if node is None:
                node = ChooseNode(
                    None,
                    None,
                    None,
                    [None for _ in range(len(current_predicate.splits))],
                )
            assert isinstance(node, ChooseNode)
            node.choices[choice] = self._propagate_pattern(
                node.choices[choice],
                pattern,
                pattern_predicates,
                current_predicate.splits[choice],
                0,
                path,
            )
            return node

        assert isinstance(current_predicate, OrderedPredicate)

        pred_key = (current_predicate.position, current_predicate.question)

        # Skip predicates not in this pattern
        if pred_key not in pattern_predicates:
            return self._propagate_pattern(
                node,
                pattern,
                pattern_predicates,
                sorted_predicates,
                predicate_index + 1,
                path,
            )

        # Create or match existing node
        if node is None:
            # Create new switch node
            node = SwitchNode(
                position=current_predicate.position, question=current_predicate.question
            )

        if self._nodes_match(node, current_predicate):
            # Continue down matching path
            pattern_answer = pattern_predicates[pred_key].a
            assert isinstance(node, SwitchNode)
            if pattern_answer not in node.children:
                node.children[pattern_answer] = None

            node.children[pattern_answer] = self._propagate_pattern(
                node.children[pattern_answer],
                pattern,
                pattern_predicates,
                sorted_predicates,
                predicate_index + 1,
                path,
            )

        else:
            # Divergence - continue down failure path
            node.failure_node = self._propagate_pattern(
                node.failure_node,
                pattern,
                pattern_predicates,
                sorted_predicates,
                predicate_index,
                path,
            )

        return node

    def _nodes_match(self, node: MatcherNode, predicate: OrderedPredicate) -> bool:
        """Check if node matches the given predicate"""
        return (
            node.position == predicate.position and node.question == predicate.question
        )

    def _insert_exit_node(self, root: MatcherNode) -> MatcherNode:
        """Insert exit node at end of failure paths"""
        curr = root
        while curr.failure_node:
            curr = curr.failure_node
        curr.failure_node = ExitNode()
        return root

    def _optimize_tree(self, root: MatcherNode) -> MatcherNode:
        """Optimize the tree by collapsing single-child switches to bools"""
        # Recursively optimize children
        if isinstance(root, SwitchNode):
            for answer in root.children:
                child_node = root.children[answer]
                if child_node is not None:
                    root.children[answer] = self._optimize_tree(child_node)
        elif isinstance(root, ChooseNode) and len(root.choices) == 1:
            assert root.choices[0] is not None
            return self._optimize_tree(root.choices[0])
        elif isinstance(root, BoolNode):
            if root.success_node is not None:
                root.success_node = self._optimize_tree(root.success_node)

        if root.failure_node is not None:
            root.failure_node = self._optimize_tree(root.failure_node)

        if isinstance(root, SwitchNode) and len(root.children) == 1:
            # Convert switch to bool node
            answer, child = next(iter(root.children.items()))
            bool_node = BoolNode(
                position=root.position,
                question=root.question,
                success_node=child,
                failure_node=root.failure_node,
                answer=answer,
            )
            return bool_node

        return root


# =============================================================================
# Code Generation
# =============================================================================


class MatcherGenerator:
    """Generates PDL interpreter matcher from matcher tree"""

    def __init__(
        self,
        matcher_func: pdl_interp.FuncOp,
        rewriter_module: ModuleOp,
        optimize_for_eqsat: bool = False,
    ) -> None:
        self.matcher_func = matcher_func
        self.rewriter_module = rewriter_module
        self.rewriter_builder = Builder(InsertPoint.at_end(rewriter_module.body.block))
        self.value_to_position: dict[pdl.PatternOp, dict[SSAValue, Position]] = {}
        self.values: ScopedDict[Position, SSAValue] = ScopedDict()
        self.failure_block_stack: list[Block] = []
        self.builder = Builder(InsertPoint.at_start(matcher_func.body.block))
        self.constraint_op_map: dict[
            ConstraintQuestion, pdl_interp.ApplyConstraintOp
        ] = {}
        self.rewriter_names: dict[str, int] = {}
        self.optimize_for_eqsat = optimize_for_eqsat

    def lower(self, patterns: list[pdl.PatternOp]) -> None:
        """Lower PDL patterns to PDL interpreter"""

        # Build the predicate tree
        tree_builder = PredicateTreeBuilder(self.optimize_for_eqsat)
        root = tree_builder.build_predicate_tree(patterns)
        self.value_to_position = tree_builder.pattern_value_positions

        # Get the entry block and add root operation argument
        entry_block = self.matcher_func.body.block

        # The first argument is the root operation
        root_pos = OperationPosition(depth=0)
        self.values[root_pos] = entry_block.args[0]

        # Generate the matcher
        self.generate_matcher(root, self.matcher_func.body, block=entry_block)

    def generate_matcher(
        self, node: MatcherNode, region: Region, block: Block | None = None
    ) -> Block:
        """Generate PDL interpreter operations for a matcher node"""

        # Create block if needed
        if block is None:
            block = Block()
            region.add_block(block)
        self.values = ScopedDict(self.values)
        assert self.values.parent is not None

        # Handle exit node - just add finalize
        if isinstance(node, ExitNode):
            finalize_op = pdl_interp.FinalizeOp()
            self.builder.insert_op(finalize_op, InsertPoint.at_end(block))
            self.values = self.values.parent  # Pop scope
            return block

        # Handle failure node
        failure_block = None
        if node.failure_node:
            # temp = self.values
            # self.values = self.values.parent
            failure_block = self.generate_matcher(node.failure_node, region)
            # self.values = temp
            self.failure_block_stack.append(failure_block)
        else:
            assert self.failure_block_stack, "Expected valid failure block"
            failure_block = self.failure_block_stack[-1]

        # Get value for position if exists
        current_block = block
        val = None
        if node.position:
            val = self.get_value_at(current_block, node.position)

        # Dispatch based on node type
        if isinstance(node, BoolNode):
            assert val is not None
            self.generate_bool_node(node, current_block, val)
        elif isinstance(node, SwitchNode):
            assert val is not None
            self.generate_switch_node(node, current_block, val)
        elif isinstance(node, SuccessNode):
            self.generate_success_node(node, current_block)

        # Pop failure block if we pushed one
        if node.failure_node:
            self.failure_block_stack.pop()

        self.values = self.values.parent  # Pop scope
        return block

    def get_value_at(self, block: Block, position: Position) -> SSAValue:
        """Get or create SSA value for a position"""

        # Check cache
        if position in self.values:
            return self.values[position]

        # Get parent value if needed
        parent_val = None
        if position.parent:
            parent_val = self.get_value_at(block, position.parent)

        # Create value based on position type
        self.builder.insertion_point = InsertPoint.at_end(block)
        value = None

        if isinstance(position, OperationPosition):
            if position.is_operand_defining_op():
                assert parent_val is not None
                # Get defining operation of operand
                defining_op = pdl_interp.GetDefiningOpOp(parent_val)
                defining_op.attributes["position"] = StringAttr(position.__repr__())
                for op in self.builder.insertion_point.block.ops:
                    if isinstance(op, pdl_interp.GetDefiningOpOp):
                        raise ValueError(
                            "Cannot have two GetDefiningOpOp in the same block"
                        )
                self.builder.insert(defining_op)
                value = defining_op.input_op
            else:
                # Passthrough
                value = parent_val

        elif isinstance(position, OperandPosition):
            assert parent_val is not None
            get_operand_op = pdl_interp.GetOperandOp(
                position.operand_number, parent_val
            )
            self.builder.insert(get_operand_op)
            value = get_operand_op.value

        elif isinstance(position, OperandGroupPosition):
            assert parent_val is not None
            # Get operands (possibly variadic)
            result_type = (
                pdl.RangeType(pdl.ValueType())
                if position.is_variadic
                else pdl.ValueType()
            )
            raise NotImplementedError("pdl_interp.get_operands is not yet implemented")
            get_operands_op = pdl_interp.GetOperandsOp(
                position.group_number, parent_val, result_type
            )
            self.builder.insert(get_operands_op)
            value = get_operands_op.value

        elif isinstance(position, ResultPosition):
            assert parent_val is not None
            get_result_op = pdl_interp.GetResultOp(position.result_number, parent_val)
            self.builder.insert(get_result_op)
            value = get_result_op.value

        elif isinstance(position, ResultGroupPosition):
            assert parent_val is not None
            # Get results (possibly variadic)
            result_type = (
                pdl.RangeType(pdl.ValueType())
                if position.is_variadic
                else pdl.ValueType()
            )
            get_results_op = pdl_interp.GetResultsOp(
                position.group_number, parent_val, result_type
            )
            self.builder.insert(get_results_op)
            value = get_results_op.value

        elif isinstance(position, AttributePosition):
            assert parent_val is not None
            get_attr_op = pdl_interp.GetAttributeOp(position.attribute_name, parent_val)
            self.builder.insert(get_attr_op)
            value = get_attr_op.value

        elif isinstance(position, AttributeLiteralPosition):
            # Create a constant attribute
            create_attr_op = pdl_interp.CreateAttributeOp(position.value)
            self.builder.insert(create_attr_op)
            value = create_attr_op.attribute

        elif isinstance(position, TypePosition):
            assert parent_val is not None
            # Get type of value or attribute
            if parent_val.type == pdl.AttributeType():
                # TODO: fix?
                # Would use GetAttributeTypeOp if it existed
                get_type_op = pdl_interp.GetValueTypeOp(parent_val)
            else:
                get_type_op = pdl_interp.GetValueTypeOp(parent_val)
            self.builder.insert(get_type_op)
            value = get_type_op.result

        elif isinstance(position, TypeLiteralPosition):
            # Create a constant type or types
            raw_type_attr = position.value
            if isinstance(raw_type_attr, TypeAttribute):
                create_type_op = pdl_interp.CreateTypeOp(raw_type_attr)
                self.builder.insert(create_type_op)
                value = create_type_op.result
            else:
                # Assume it's an ArrayAttr of types
                assert isinstance(raw_type_attr, ArrayAttr)
                type_attr = cast(ArrayAttr[TypeAttribute], raw_type_attr)
                create_types_op = pdl_interp.CreateTypesOp(type_attr)
                self.builder.insert(create_types_op)
                value = create_types_op.result

        elif isinstance(position, ConstraintPosition):
            # The constraint op has already been created, find it in the map
            constraint_op = self.constraint_op_map.get(position.constraint)
            assert constraint_op is not None
            value = constraint_op.results[position.result_index]

        elif isinstance(position, UsersPosition):
            raise NotImplementedError("UsersPosition not implemented in lowering")
        elif isinstance(position, ForEachPosition):
            raise NotImplementedError("ForEachPosition not implemented in lowering")
        else:
            raise NotImplementedError(f"Unhandled position type {type(position)}")

        # Cache and return
        if value:
            self.values[position] = value
        assert value is not None
        return value

    def generate_bool_node(self, node: BoolNode, block: Block, val: SSAValue) -> None:
        """Generate operations for a boolean predicate node"""

        question = node.question
        answer = node.answer
        region = block.parent
        assert region is not None, "Block must be in a region"

        # Handle getValue queries first for constraint questions
        args: list[SSAValue] = []
        if isinstance(question, EqualToQuestion):
            args = [self.get_value_at(block, question.other_position)]
        elif isinstance(question, ConstraintQuestion):
            for position in question.arg_positions:
                args.append(self.get_value_at(block, position))

        # Create success block
        success_block = Block()
        region.add_block(success_block)
        failure_block = self.failure_block_stack[-1]

        # Generate predicate check operation based on question type
        if isinstance(question, IsNotNullQuestion):
            check_op = pdl_interp.IsNotNullOp(val, success_block, failure_block)

        elif isinstance(question, OperationNameQuestion):
            assert isinstance(answer, StringAnswer)
            check_op = pdl_interp.CheckOperationNameOp(
                answer.value, val, success_block, failure_block
            )

        elif isinstance(question, OperandCountQuestion | OperandCountAtLeastQuestion):
            assert isinstance(answer, UnsignedAnswer)
            compare_at_least = isinstance(question, OperandCountAtLeastQuestion)
            check_op = pdl_interp.CheckOperandCountOp(
                val, answer.value, success_block, failure_block, compare_at_least
            )

        elif isinstance(question, ResultCountQuestion | ResultCountAtLeastQuestion):
            assert isinstance(answer, UnsignedAnswer)
            compare_at_least = isinstance(question, ResultCountAtLeastQuestion)
            check_op = pdl_interp.CheckResultCountOp(
                val, answer.value, success_block, failure_block, compare_at_least
            )

        elif isinstance(question, EqualToQuestion):
            # Get the other value to compare with
            other_val = self.get_value_at(block, question.other_position)
            assert isinstance(answer, TrueAnswer)
            check_op = pdl_interp.AreEqualOp(
                val, other_val, success_block, failure_block
            )

        elif isinstance(question, AttributeConstraintQuestion):
            assert isinstance(answer, AttributeAnswer)
            check_op = pdl_interp.CheckAttributeOp(
                answer.value, val, success_block, failure_block
            )

        elif isinstance(question, TypeConstraintQuestion):
            assert isinstance(answer, TypeAnswer)
            if isinstance(val.type, pdl.RangeType):
                # Check multiple types
                raise NotImplementedError(
                    "pdl_interp.check_types is not yet implemented"
                )
                check_op = pdl_interp.CheckTypesOp(
                    val, answer.value, success_block, failure_block
                )
            else:
                # Check single type
                assert isinstance(answer.value, TypeAttribute)
                check_op = pdl_interp.CheckTypeOp(
                    answer.value, val, success_block, failure_block
                )

        elif isinstance(question, ConstraintQuestion):
            # TODO: question.result_types is not part of the dialect definition yet
            check_op = pdl_interp.ApplyConstraintOp(
                question.name,
                args,
                success_block,
                failure_block,
                is_negated=question.is_negated,
            )
            # Store the constraint op for later result access
            self.constraint_op_map[question] = check_op

        else:
            raise NotImplementedError(f"Unhandled question type {type(question)}")

        self.builder.insert_op(check_op, InsertPoint.at_end(block))

        # Generate matcher for success node
        if node.success_node:
            self.generate_matcher(node.success_node, region, success_block)

    def generate_switch_node(
        self, node: SwitchNode, block: Block, val: SSAValue
    ) -> None:
        """Generate operations for a switch node"""

        question = node.question
        region = block.parent
        assert region is not None, "Block must be in a region"
        default_dest = self.failure_block_stack[-1]

        # Handle at-least questions specially
        if isinstance(
            question, OperandCountAtLeastQuestion | ResultCountAtLeastQuestion
        ):
            # Sort children in reverse numerical order
            sorted_children = sorted(
                node.children.items(),
                key=lambda x: cast(UnsignedAnswer, x[0]).value,
                reverse=True,
            )

            current_failure_target = default_dest
            for answer, child_node in sorted_children:
                if child_node:
                    success_block = self.generate_matcher(child_node, region)
                    current_check_block = Block()
                    region.add_block(current_check_block)

                    self.builder.insertion_point = InsertPoint.at_end(
                        current_check_block
                    )
                    assert isinstance(answer, UnsignedAnswer)
                    if isinstance(question, OperandCountAtLeastQuestion):
                        check_op = pdl_interp.CheckOperandCountOp(
                            val,
                            answer.value,
                            success_block,
                            current_failure_target,
                            True,
                        )
                    else:
                        check_op = pdl_interp.CheckResultCountOp(
                            val,
                            answer.value,
                            success_block,
                            current_failure_target,
                            True,
                        )
                    self.builder.insert(check_op)
                    current_failure_target = current_check_block

            # Move ops from the first check block into the main block
            if block.parent:
                for op in list(current_failure_target.ops):
                    op.detach()
                    block.add_op(op)
                current_failure_target.erase()

            return

        # Generate child blocks and collect case values
        case_blocks: list[Block] = []
        case_values: list[Answer] = []

        for answer, child_node in node.children.items():
            if child_node:
                child_block = self.generate_matcher(child_node, region)
                case_blocks.append(child_block)
                case_values.append(answer)

        # Position builder at end of current block
        self.builder.insertion_point = InsertPoint.at_end(block)

        # Create switch operation based on question type
        if isinstance(question, OperationNameQuestion):
            # Extract string values from StringAnswer objects
            switch_values = [cast(StringAnswer, ans).value for ans in case_values]
            switch_attr = ArrayAttr([StringAttr(v) for v in switch_values])
            switch_op = pdl_interp.SwitchOperationNameOp(
                switch_attr, val, default_dest, case_blocks
            )

        elif isinstance(question, OperandCountQuestion):
            # Extract integer values from UnsignedAnswer objects
            switch_values = [cast(UnsignedAnswer, ans).value for ans in case_values]
            switch_attr = ArrayAttr([IntegerAttr(v, 32) for v in switch_values])
            raise NotImplementedError(
                "pdl_interp.switch_operand_count is not yet implemented"
            )
            switch_op = pdl_interp.SwitchOperandCountOp(
                switch_attr, val, default_dest, case_blocks
            )

        elif isinstance(question, ResultCountQuestion):
            # Extract integer values from UnsignedAnswer objects
            switch_values = [cast(UnsignedAnswer, ans).value for ans in case_values]
            switch_attr = ArrayAttr([IntegerAttr(v, 32) for v in switch_values])
            raise NotImplementedError(
                "pdl_interp.switch_result_count is not yet implemented"
            )
            switch_op = pdl_interp.SwitchResultCountOp(
                switch_attr, val, default_dest, case_blocks
            )

        elif isinstance(question, TypeConstraintQuestion):
            # Extract type attributes from TypeAnswer objects
            switch_values = [cast(TypeAnswer, ans).value for ans in case_values]
            raise NotImplementedError("pdl_interp.switch_types is not yet implemented")
            if isinstance(val.type, pdl.RangeType):
                switch_attr = ArrayAttr(switch_values)

                switch_op = pdl_interp.SwitchTypesOp(
                    switch_attr, val, default_dest, case_blocks
                )
            else:
                switch_attr = ArrayAttr(switch_values)
                switch_op = pdl_interp.SwitchTypeOp(
                    switch_attr, val, default_dest, case_blocks
                )

        elif isinstance(question, AttributeConstraintQuestion):
            # Extract attribute values from AttributeAnswer objects
            switch_values = [cast(AttributeAnswer, ans).value for ans in case_values]
            switch_attr = ArrayAttr(switch_values)
            switch_op = pdl_interp.SwitchAttributeOp(
                val, switch_attr, default_dest, case_blocks
            )
        else:
            raise NotImplementedError(f"Unhandled question type {type(question)}")

        self.builder.insert(switch_op)

    def generate_success_node(self, node: SuccessNode, block: Block) -> None:
        """Generate operations for a successful match"""
        self.builder.insertion_point = InsertPoint.at_end(block)

        pattern = node.pattern
        root = node.root

        # Generate a rewriter for the pattern
        used_match_positions: list[Position] = []
        rewriter_func_ref = self.generate_rewriter(pattern, used_match_positions)

        # Process values used in the rewrite that are defined in the match
        mapped_match_values = [
            self.get_value_at(block, pos) for pos in used_match_positions
        ]

        # Collect generated op names from DAG rewriter
        rewriter_op = pattern.body.block.last_op
        assert isinstance(rewriter_op, pdl.RewriteOp)
        generated_op_names: list[str] = []
        if not rewriter_op.name:
            assert rewriter_op.body is not None
            for op in rewriter_op.body.walk():
                if isinstance(op, pdl.OperationOp) and op.opName:
                    generated_op_names.append(op.opName.data)

        # Get root kind if present
        root_kind: StringAttr | None = None
        if root:
            defining_op = root.owner
            if isinstance(defining_op, pdl.OperationOp) and defining_op.opName:
                root_kind = StringAttr(defining_op.opName.data)

        # Create the RecordMatchOp
        record_op = pdl_interp.RecordMatchOp(
            rewriter_func_ref,
            root_kind,
            (
                ArrayAttr([StringAttr(s) for s in generated_op_names])
                if generated_op_names
                else None
            ),
            pattern.benefit,
            mapped_match_values,
            [],
            self.failure_block_stack[-1],
        )
        self.builder.insert(record_op)

    def generate_rewriter(
        self, pattern: pdl.PatternOp, used_match_positions: list[Position]
    ) -> SymbolRefAttr:
        """
        Generate a rewriter function for the given pattern, and return a
        reference to that function.
        """
        rewriter_op = pattern.body.block.last_op
        assert isinstance(rewriter_op, pdl.RewriteOp)

        rewriter_name = "pdl_generated_rewriter"
        if pattern.sym_name:
            rewriter_name = pattern.sym_name.data
        if rewriter_name in self.rewriter_names:
            self.rewriter_names[rewriter_name] += 1
            rewriter_name = f"{rewriter_name}_{self.rewriter_names[rewriter_name]}"
        else:
            self.rewriter_names[rewriter_name] = 1

        # Create the rewriter function
        rewriter_func = pdl_interp.FuncOp(rewriter_name, ([], []))

        self.rewriter_module.body.block.add_op(rewriter_func)
        entry_block = rewriter_func.body.block
        self.rewriter_builder.insertion_point = InsertPoint.at_end(entry_block)

        rewrite_values: dict[SSAValue, SSAValue] = {}
        pattern_value_positions = self.value_to_position[pattern]

        def map_rewrite_value(old_value: SSAValue) -> SSAValue:
            if new_value := rewrite_values.get(old_value):
                return new_value

            # Prefer materializing constants directly when possible.
            old_op = old_value.owner
            new_val_op: Operation | None = None
            if isinstance(old_op, pdl.AttributeOp) and old_op.value:
                new_val_op = pdl_interp.CreateAttributeOp(old_op.value)
            elif isinstance(old_op, pdl.TypeOp) and old_op.constantType:
                new_val_op = pdl_interp.CreateTypeOp(old_op.constantType)
            elif isinstance(old_op, pdl.TypesOp) and old_op.constantTypes:
                new_val_op = pdl_interp.CreateTypesOp(old_op.constantTypes)

            if new_val_op:
                self.rewriter_builder.insert(new_val_op)
                new_value = new_val_op.results[0]
                rewrite_values[old_value] = new_value
                return new_value

            # Otherwise, it's an input from the matcher.
            input_pos = pattern_value_positions.get(old_value)
            assert input_pos is not None, "Expected value to be a pattern input"
            if input_pos not in used_match_positions:
                used_match_positions.append(input_pos)

            arg = entry_block.insert_arg(old_value.type, len(entry_block.args))
            rewrite_values[old_value] = arg
            return arg

        # If this is a custom rewriter, dispatch to the registered method.
        if rewriter_op.name_:
            args: list[SSAValue] = []
            if rewriter_op.root:
                args.append(map_rewrite_value(rewriter_op.root))
            args.extend(map_rewrite_value(arg) for arg in rewriter_op.external_args)
            raise NotImplementedError("pdl_interp.apply_rewrite is not yet implemented")
            apply_op = pdl_interp.ApplyRewriteOp(args, name=rewriter_op.name)
            self.rewriter_builder.insert(apply_op)
        else:
            # Otherwise, this is a DAG rewriter defined using PDL operations.
            assert rewriter_op.body is not None
            for op in rewriter_op.body.ops:
                if isinstance(op, pdl.ApplyNativeRewriteOp):
                    self._generate_rewriter_for_apply_native_rewrite(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.AttributeOp):
                    self._generate_rewriter_for_attribute(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.EraseOp):
                    self._generate_rewriter_for_erase(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.OperationOp):
                    self._generate_rewriter_for_operation(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.RangeOp):
                    self._generate_rewriter_for_range(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.ReplaceOp):
                    self._generate_rewriter_for_replace(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.ResultOp):
                    self._generate_rewriter_for_result(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.ResultsOp):
                    self._generate_rewriter_for_results(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.TypeOp):
                    self._generate_rewriter_for_type(
                        op, rewrite_values, map_rewrite_value
                    )
                elif isinstance(op, pdl.TypesOp):
                    self._generate_rewriter_for_types(
                        op, rewrite_values, map_rewrite_value
                    )

        # Update the signature of the rewrite function.
        rewriter_func.function_type = FunctionType.from_lists(entry_block.arg_types, ())

        self.rewriter_builder.insert(pdl_interp.FinalizeOp())
        return SymbolRefAttr(
            "rewriters",
            [
                StringAttr(rewriter_name),
            ],
        )

    def _generate_rewriter_for_apply_native_rewrite(
        self,
        op: pdl.ApplyNativeRewriteOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ):
        arguments = [map_rewrite_value(arg) for arg in op.args]
        result_types = [res.type for res in op.res]
        raise NotImplementedError("pdl_interp.apply_rewrite is not yet implemented")
        interp_op = pdl_interp.ApplyRewriteOp(
            arguments, name=op.constraint_name, result_types=result_types
        )
        self.rewriter_builder.insert(interp_op)
        for old_res, new_res in zip(op.results, interp_op.results):
            rewrite_values[old_res] = new_res

    def _generate_rewriter_for_attribute(
        self,
        op: pdl.AttributeOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ):
        if op.value:
            new_attr_op = pdl_interp.CreateAttributeOp(op.value)
            self.rewriter_builder.insert(new_attr_op)
            rewrite_values[op.output] = new_attr_op.attribute

    def _generate_rewriter_for_erase(
        self,
        op: pdl.EraseOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ) -> None:
        raise NotImplementedError("pdl_interp.erase is not yet implemented")
        self.rewriter_builder.insert(pdl_interp.EraseOp(map_rewrite_value(op.op_value)))

    def _generate_rewriter_for_operation(
        self,
        op: pdl.OperationOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ):
        operands = [map_rewrite_value(operand) for operand in op.operand_values]
        attributes = [map_rewrite_value(attr) for attr in op.attribute_values]

        types: list[SSAValue] = []
        has_inferred_result_types = self._generate_operation_result_type_rewriter(
            op, map_rewrite_value, types, rewrite_values
        )

        if op.opName is None:
            raise ValueError("Cannot create operation without a name.")

        create_op = pdl_interp.CreateOperationOp(
            op.opName,
            UnitAttr() if has_inferred_result_types else None,
            op.attributeValueNames,
            operands,
            attributes,
            types,
        )
        self.rewriter_builder.insert(create_op)
        created_op_val = create_op.result_op
        rewrite_values[op.op] = created_op_val

        # Generate accesses for any results that have their types constrained.
        result_types = op.type_values
        if len(result_types) == 1 and isinstance(result_types[0].type, pdl.RangeType):
            if result_types[0] not in rewrite_values:
                get_results = pdl_interp.GetResultsOp(
                    None, created_op_val, pdl.RangeType(pdl.ValueType())
                )
                self.rewriter_builder.insert(get_results)
                get_type = pdl_interp.GetValueTypeOp(get_results.value)
                self.rewriter_builder.insert(get_type)
                rewrite_values[result_types[0]] = get_type.result
            return

        seen_variable_length = False
        for i, type_value in enumerate(result_types):
            if type_value in rewrite_values:
                continue
            is_variadic = isinstance(type_value.type, pdl.RangeType)
            seen_variable_length = seen_variable_length or is_variadic

            result_val: SSAValue
            if seen_variable_length:
                get_results = pdl_interp.GetResultsOp(
                    i, created_op_val, pdl.RangeType(pdl.ValueType())
                )
                self.rewriter_builder.insert(get_results)
                result_val = get_results.value
            else:
                get_result = pdl_interp.GetResultOp(i, created_op_val)
                self.rewriter_builder.insert(get_result)
                result_val = get_result.value

            get_type = pdl_interp.GetValueTypeOp(result_val)
            self.rewriter_builder.insert(get_type)
            rewrite_values[type_value] = get_type.result

    def _generate_rewriter_for_range(
        self,
        op: pdl.RangeOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ) -> None:
        args = [map_rewrite_value(arg) for arg in op.arguments]
        raise NotImplementedError("pdl_interp.create_range is not yet implemented")
        create_range_op = pdl_interp.CreateRangeOp(args, op.result.type)
        self.rewriter_builder.insert(create_range_op)
        rewrite_values[op.result] = create_range_op.range

    def _generate_rewriter_for_replace(
        self,
        op: pdl.ReplaceOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ):
        repl_operands: list[SSAValue] = []
        if op.repl_operation:
            op_op_def = op.op_value.owner
            has_results = not (
                isinstance(op_op_def, pdl.OperationOp) and not op_op_def.type_values
            )
            if has_results:
                get_results = pdl_interp.GetResultsOp(
                    None,
                    map_rewrite_value(op.repl_operation),
                    pdl.RangeType(pdl.ValueType()),
                )
                self.rewriter_builder.insert(get_results)
                repl_operands.append(get_results.value)
        else:
            repl_operands.extend(map_rewrite_value(val) for val in op.repl_values)

        mapped_op_value = map_rewrite_value(op.op_value)
        if not repl_operands:
            raise NotImplementedError("pdl_interp.erase is not yet implemented")
            self.rewriter_builder.insert(pdl_interp.EraseOp(mapped_op_value))
        else:
            self.rewriter_builder.insert(
                pdl_interp.ReplaceOp(mapped_op_value, repl_operands)
            )

    def _generate_rewriter_for_result(
        self,
        op: pdl.ResultOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ):
        get_result_op = pdl_interp.GetResultOp(op.index, map_rewrite_value(op.parent_))
        self.rewriter_builder.insert(get_result_op)
        rewrite_values[op.val] = get_result_op.value

    def _generate_rewriter_for_results(
        self,
        op: pdl.ResultsOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ):
        get_results_op = pdl_interp.GetResultsOp(
            op.index, map_rewrite_value(op.parent_), op.val.type
        )
        self.rewriter_builder.insert(get_results_op)
        rewrite_values[op.val] = get_results_op.value

    def _generate_rewriter_for_type(
        self,
        op: pdl.TypeOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ):
        if op.constantType:
            create_type_op = pdl_interp.CreateTypeOp(op.constantType)
            self.rewriter_builder.insert(create_type_op)
            rewrite_values[op.result] = create_type_op.result

    def _generate_rewriter_for_types(
        self,
        op: pdl.TypesOp,
        rewrite_values: dict[SSAValue, SSAValue],
        map_rewrite_value: Callable[[SSAValue], SSAValue],
    ):
        if op.constantTypes:
            create_types_op = pdl_interp.CreateTypesOp(op.constantTypes)
            self.rewriter_builder.insert(create_types_op)
            rewrite_values[op.result] = create_types_op.result

    def _generate_operation_result_type_rewriter(
        self,
        op: pdl.OperationOp,
        map_rewrite_value: Callable[[SSAValue], SSAValue],
        types_list: list[SSAValue],
        rewrite_values: dict[SSAValue, SSAValue],
    ) -> bool:
        """Returns `has_inferred_result_types`"""
        rewriter_block = op.parent
        assert rewriter_block is not None
        result_type_values = op.type_values

        # Strategy 1: Resolve all types individually
        if result_type_values:
            temp_types: list[SSAValue] = []
            can_resolve_all = True
            for result_type in result_type_values:
                if val := rewrite_values.get(result_type):
                    temp_types.append(val)
                elif result_type.owner.parent is not rewriter_block:
                    temp_types.append(map_rewrite_value(result_type))
                else:
                    can_resolve_all = False
                    break
            if can_resolve_all:
                types_list.extend(temp_types)
                return False

        # Strategy 2: Check for `inferredResultTypes` attribute hint
        if "inferredResultTypes" in op.attributes:
            return True

        # Strategy 3: Infer from a replaced operation
        for use in op.op.uses:
            user_op = use.operation
            if not isinstance(user_op, pdl.ReplaceOp) or use.index == 0:
                continue

            replaced_op_val = user_op.op_value
            replaced_op_def = replaced_op_val.owner
            if replaced_op_def.parent is rewriter_block:
                # MLIR has `Operation::isBeforeInBlock` to execute this check more efficiently:
                is_before = False
                p = rewriter_block.first_op
                assert p is not None
                while p is not replaced_op_def:
                    if p is op:
                        is_before = True
                        break
                    p = p.next_op
                    assert p is not None
                if is_before:
                    continue

            mapped_replaced_op = map_rewrite_value(replaced_op_val)
            get_results = pdl_interp.GetResultsOp(
                None, mapped_replaced_op, pdl.RangeType(pdl.ValueType())
            )
            self.rewriter_builder.insert(get_results)
            get_type = pdl_interp.GetValueTypeOp(get_results.value)
            self.rewriter_builder.insert(get_type)
            types_list.append(get_type.result)
            return False

        # Strategy 4: If no explicit types, assume no results
        if not result_type_values:
            return False

        raise ValueError(
            f"Unable to infer result types for pdl.operation '{op.opName}'"
        )


def lower_pdl_to_pdl_interp(
    module: ModuleOp,
    matcher_func: pdl_interp.FuncOp,
    rewriter_module: ModuleOp,
    optimize_for_eqsat: bool = False,
) -> None:
    """Main entry point to lower PDL patterns to PDL interpreter"""

    # Collect all patterns
    patterns = [op for op in module.body.ops if isinstance(op, pdl.PatternOp)]

    # Create generator and lower
    generator = MatcherGenerator(matcher_func, rewriter_module, optimize_for_eqsat)
    generator.lower(patterns)


# =============================================================================
# Main Transformation Pipeline
# =============================================================================


@dataclass(frozen=True)
class ConvertPDLToPDLInterpPass(ModulePass):
    """
    Pass to convert PDL operations to PDL interpreter operations.
    This is a somewhat faithful port of the implementation in MLIR, but it may not generate the same exact results.
    """

    name = "convert-pdl-to-pdl-interp"

    optimize_for_eqsat: bool = False

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        patterns = [
            pattern for pattern in op.body.ops if isinstance(pattern, pdl.PatternOp)
        ]
        if not patterns:
            return

        matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
        rewriter_module = ModuleOp([], sym_name=StringAttr("rewriters"))

        generator = MatcherGenerator(
            matcher_func, rewriter_module, self.optimize_for_eqsat
        )
        generator.lower(patterns)

        # Replace all pattern ops with the matcher func and rewriter module
        rewriter = Rewriter()
        for pattern in patterns:
            rewriter.erase_op(pattern)
        op.body.block.add_op(matcher_func)
        if rewriter_module.body.block.ops:
            op.body.block.add_op(rewriter_module)
