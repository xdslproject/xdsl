"""
PDL to PDL_interp Transformation
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from xdsl.dialects import pdl
from xdsl.ir import (
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
    Answer,
    AttributeLiteralPosition,
    AttributePosition,
    ConstraintPosition,
    ConstraintQuestion,
    OperandGroupPosition,
    OperandPosition,
    OperationPosition,
    Position,
    PositionalPredicate,
    Predicate,
    Question,
    TypeLiteralPosition,
    TypePosition,
    get_position_cost,
    get_question_cost,
)

# =============================================================================
# Pattern Analysis
# =============================================================================


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


class PredicateTreeBuilder:
    """Builds optimized predicate matching trees"""

    analyzer: PatternAnalyzer
    _pattern_roots: dict[pdl.PatternOp, SSAValue]
    pattern_value_positions: dict[pdl.PatternOp, dict[SSAValue, Position]]

    def __init__(self):
        self.analyzer = PatternAnalyzer()
        self._pattern_roots = {}
        self.pattern_value_positions = {}

    def _extract_pattern_predicates(
        self, pattern: pdl.PatternOp
    ) -> tuple[list[PositionalPredicate], SSAValue, dict[SSAValue, Position]]:
        """Extract all predicates for a single pattern"""
        predicates: list[PositionalPredicate] = []
        inputs: dict[SSAValue, Position] = {}

        roots = self.analyzer.detect_roots(pattern)
        if len(roots) != 1:
            raise ValueError("Multi-root patterns are not yet supported.")

        rewriter = pattern.body.block.last_op
        assert isinstance(rewriter, pdl.RewriteOp)
        best_root = rewriter.root if rewriter.root is not None else roots[0]

        # Downward traversal from the best root
        root_pos = OperationPosition(depth=0)
        predicates.extend(
            self.analyzer.extract_tree_predicates(best_root, root_pos, inputs)
        )

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
