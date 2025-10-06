"""
PDL to PDL_interp Transformation
"""

from collections.abc import Sequence
from typing import cast

from xdsl.dialects import pdl
from xdsl.ir import (
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
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
    TypeLiteralPosition,
    TypePosition,
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
        if isinstance(position, AttributePosition):
            assert isinstance(value, OpResult)
            predicates.extend(
                self._extract_attribute_predicates(value.owner, position, inputs)
            )
        elif isinstance(position, OperationPosition):
            assert isinstance(value, OpResult)
            predicates.extend(
                self._extract_operation_predicates(
                    value.owner, position, inputs, ignore_operand
                )
            )
        elif isinstance(position, TypePosition):
            assert isinstance(value, OpResult)
            predicates.extend(
                self._extract_type_predicates(value.owner, position, inputs)
            )
        elif isinstance(position, OperandPosition | OperandGroupPosition):
            assert isinstance(value, SSAValue)
            predicates.extend(
                self._extract_operand_tree_predicates(value, position, inputs)
            )
        else:
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

        if not op_pos.is_root():
            is_not_null = Predicate.get_is_not_null()
            predicates.append(
                PositionalPredicate(q=is_not_null.q, a=is_not_null.a, position=op_pos)
            )

        if not isinstance(op_op, pdl.OperationOp):
            return predicates

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
        for attr_name, attr in zip(op_op.attributeValueNames, op_op.attribute_values):
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

        if isinstance(defining_op, pdl.OperandOp | pdl.OperandsOp):
            if isinstance(defining_op, pdl.OperandOp):
                is_not_null = Predicate.get_is_not_null()
                predicates.append(
                    PositionalPredicate(
                        q=is_not_null.q, a=is_not_null.a, position=operand_pos
                    )
                )
            elif (
                isinstance(operand_pos, OperandGroupPosition)
                and operand_pos.group_number is not None
            ):
                is_not_null = Predicate.get_is_not_null()
                predicates.append(
                    PositionalPredicate(
                        q=is_not_null.q, a=is_not_null.a, position=operand_pos
                    )
                )

            if defining_op.value_type:
                type_pos = operand_pos.get_type()
                predicates.extend(
                    self.extract_tree_predicates(
                        defining_op.value_type, type_pos, inputs
                    )
                )

        elif isinstance(defining_op, pdl.ResultOp | pdl.ResultsOp):
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

            if isinstance(defining_op, pdl.ResultOp):
                result_pos = defining_op_pos.get_result(
                    index if index is not None else 0
                )
            else:  # ResultsOp
                result_pos = defining_op_pos.get_result_group(index, is_variadic)

            equal_to = Predicate.get_equal_to(operand_pos)
            predicates.append(
                PositionalPredicate(q=equal_to.q, a=equal_to.a, position=result_pos)
            )

            # Recursively process the parent operation
            predicates.extend(
                self.extract_tree_predicates(parent_op, defining_op_pos, inputs)
            )

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
            if isinstance(op, pdl.AttributeOp):
                if op.output not in inputs:
                    if op.value:
                        # Create literal position for constant attribute
                        attr_pos = AttributeLiteralPosition(value=op.value, parent=None)
                        inputs[op.output] = attr_pos

            elif isinstance(op, pdl.ApplyNativeConstraintOp):
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
                # TODO: is_negated is not part of the dialect definition yet
                is_negated = False
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
                            if constraint_pos.get_operation_depth()
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
                        q=constraint_pred.q, a=constraint_pred.a, position=furthest_pos
                    )
                )

            elif isinstance(op, pdl.ResultOp):
                # Ensure result exists
                if op.val not in inputs:
                    assert isinstance(op.parent_.owner, pdl.OperationOp)
                    parent_pos = inputs.get(op.parent_.owner.op)
                    if parent_pos and isinstance(parent_pos, OperationPosition):
                        result_pos = parent_pos.get_result(op.index.value.data)
                        is_not_null = Predicate.get_is_not_null()
                        predicates.append(
                            PositionalPredicate(
                                q=is_not_null.q, a=is_not_null.a, position=result_pos
                            )
                        )

            elif isinstance(op, pdl.ResultsOp):
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

            elif isinstance(op, pdl.TypeOp):
                # Handle constant types
                if op.result not in inputs and op.constantType:
                    type_pos = TypeLiteralPosition.get_type_literal(
                        value=op.constantType
                    )
                    inputs[op.result] = type_pos

            elif isinstance(op, pdl.TypesOp):
                # Handle constant type arrays
                if op.result not in inputs and op.constantTypes:
                    type_pos = TypeLiteralPosition.get_type_literal(
                        value=op.constantTypes
                    )
                    inputs[op.result] = type_pos

        return predicates
