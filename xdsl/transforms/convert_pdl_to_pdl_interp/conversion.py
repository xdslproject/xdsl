from xdsl.dialects import pdl
from xdsl.ir import (
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.transforms.convert_pdl_to_pdl_interp.predicate import (
    AttributePosition,
    Position,
    PositionalPredicate,
    Predicate,
    TypePosition,
)


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

    def _extract_attribute_predicates(
        self,
        attr_value: Operation | SSAValue,
        attr_pos: AttributePosition,
        inputs: dict[SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates for an attribute"""
        predicates: list[PositionalPredicate] = []

        is_not_null = Predicate.get_is_not_null()
        predicates.append(
            PositionalPredicate(q=is_not_null.q, a=is_not_null.a, position=attr_pos)
        )

        # Get the actual attribute operation
        if isinstance(attr_value, SSAValue):
            attr_op = attr_value.owner
        else:
            attr_op = attr_value

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
