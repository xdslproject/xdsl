"""
PDL to PDL_interp Transformation
"""

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional, cast

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import pdl, pdl_interp
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, ModuleOp, StringAttr
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue, TypeAttribute
from xdsl.passes import ModulePass
from xdsl.rewriter import InsertPoint, Rewriter


class Kind(IntEnum):
    OperationPosition = 0
    OperandPosition = 1
    OperandGroupPos = 2
    AttributePos = 3
    ConstraintResultPos = 4
    ResultPos = 5
    ResultGroupPos = 6
    TypePos = 7
    AttributeLiteralPos = 8
    TypeLiteralPos = 9
    UsersPos = 10
    ForEachPos = 11  # Not implemented yet

    #   Questions, ordered by dependency and decreasing priority
    IsNotNullQuestion = 12
    OperationNameQuestion = 13
    TypeQuestion = 14
    AttributeQuestion = 15
    OperandCountAtLeastQuestion = 16
    OperandCountQuestion = 17
    ResultCountAtLeastQuestion = 18
    ResultCountQuestion = 19
    EqualToQuestion = 20
    ConstraintQuestion = 21

    #   Answers
    AttributeAnswer = 22
    FalseAnswer = 23
    OperationNameAnswer = 24
    TrueAnswer = 25
    TypeAnswer = 26
    UnsignedAnswer = 27


# =============================================================================
# Core Data Structures - Positions
# =============================================================================


@dataclass(frozen=True)
class Position(ABC):
    """Base class for all position types"""

    parent: Optional["Position"] = None

    def get_operation_depth(self) -> int:
        """Returns depth of first ancestor operation position"""
        if isinstance(self, OperationPosition):
            return self.depth
        return self.parent.get_operation_depth() if self.parent else 0

    @property
    def kind(self) -> Kind:
        raise NotImplementedError()


@dataclass(frozen=True, kw_only=True)
class OperationPosition(Position):
    """Represents an operation in the IR"""

    depth: int

    def is_root(self) -> bool:
        return self.depth == 0

    def is_operand_defining_op(self) -> bool:
        return isinstance(self.parent, (OperandPosition | OperandGroupPosition))

    def __repr__(self):
        if self.is_root():
            return "root"
        else:
            return self.parent.__repr__() + ".defining_op"

    @property
    def kind(self) -> Kind:
        return Kind.OperationPosition


@dataclass(frozen=True, kw_only=True)
class OperandPosition(Position):
    """Represents an operand of an operation"""

    operand_number: int

    def __repr__(self):
        return f"{self.parent.__repr__()}.operand[{self.operand_number}]"

    @property
    def kind(self) -> Kind:
        return Kind.OperandPosition


@dataclass(frozen=True, kw_only=True)
class OperandGroupPosition(Position):
    """Represents a group of operands"""

    group_number: int | None
    is_variadic: bool

    @property
    def kind(self) -> Kind:
        return Kind.OperandGroupPos


@dataclass(frozen=True, kw_only=True)
class ResultPosition(Position):
    """Represents a result of an operation"""

    result_number: int

    def __repr__(self):
        return f"{self.parent.__repr__()}.result[{self.result_number}]"

    @property
    def kind(self) -> Kind:
        return Kind.ResultPos


@dataclass(frozen=True, kw_only=True)
class AttributePosition(Position):
    """Represents an attribute of an operation"""

    attribute_name: str

    def __repr__(self):
        return f"{self.parent.__repr__()}.attribute[{self.attribute_name}]"

    @property
    def kind(self) -> Kind:
        return Kind.AttributePos


@dataclass(frozen=True)
class TypePosition(Position):
    """Represents the type of a value"""

    @property
    def kind(self) -> Kind:
        return Kind.TypePos


@dataclass(frozen=True, kw_only=True)
class UsersPosition(Position):
    """Represents users of a value"""

    use_representative: bool

    @property
    def kind(self) -> Kind:
        return Kind.UsersPos


@dataclass(frozen=True, kw_only=True)
class ResultGroupPosition(Position):
    """Represents a group of results"""

    group_number: int | None
    is_variadic: bool

    @property
    def kind(self) -> Kind:
        return Kind.ResultGroupPos


@dataclass(frozen=True, kw_only=True)
class AttributeLiteralPosition(Position):
    """Represents a literal attribute value"""

    value: Attribute

    @property
    def kind(self) -> Kind:
        return Kind.AttributeLiteralPos


@dataclass(frozen=True, kw_only=True)
class TypeLiteralPosition(Position):
    """Represents a literal type value"""

    value: Attribute  # Can be a single type or array of types

    @property
    def kind(self) -> Kind:
        return Kind.TypeLiteralPos


@dataclass(frozen=True, kw_only=True)
class ConstraintPosition(Position):
    """Represents a result from a constraint"""

    constraint: "ConstraintQuestion"
    result_index: int

    @property
    def kind(self) -> Kind:
        return Kind.ConstraintResultPos


# =============================================================================
# Predicate System - Questions and Answers
# =============================================================================


@dataclass(frozen=True)
class Predicate(ABC):
    """Base predicate class"""

    @property
    def kind(self) -> Kind:
        raise NotImplementedError()


@dataclass(frozen=True)
class Question(Predicate):
    """Represents a question/check to perform"""

    pass


@dataclass(frozen=True)
class Answer(Predicate):
    """Represents an expected answer to a question"""

    pass


# Question Types
@dataclass(frozen=True)
class IsNotNullQuestion(Question):
    @property
    def kind(self) -> Kind:
        return Kind.IsNotNullQuestion


@dataclass(frozen=True)
class OperationNameQuestion(Question):
    @property
    def kind(self) -> Kind:
        return Kind.OperationNameQuestion


@dataclass(frozen=True)
class OperandCountQuestion(Question):
    @property
    def kind(self) -> Kind:
        return Kind.OperandCountQuestion


@dataclass(frozen=True)
class ResultCountQuestion(Question):
    @property
    def kind(self) -> Kind:
        return Kind.ResultCountQuestion


@dataclass(frozen=True)
class EqualToQuestion(Question):
    other_position: Position

    @property
    def kind(self) -> Kind:
        return Kind.EqualToQuestion


@dataclass(frozen=True)
class OperandCountAtLeastQuestion(Question):
    @property
    def kind(self) -> Kind:
        return Kind.OperandCountAtLeastQuestion


@dataclass(frozen=True)
class ResultCountAtLeastQuestion(Question):
    @property
    def kind(self) -> Kind:
        return Kind.ResultCountAtLeastQuestion


@dataclass(frozen=True)
class AttributeConstraintQuestion(Question):
    @property
    def kind(self) -> Kind:
        return Kind.AttributeQuestion


@dataclass(frozen=True)
class TypeConstraintQuestion(Question):
    @property
    def kind(self) -> Kind:
        return Kind.TypeQuestion


@dataclass(frozen=True)
class ConstraintQuestion(Question):
    """Represents a native constraint check"""

    name: str
    arg_positions: list[Position]
    result_types: list[pdl.AnyPDLType]
    is_negated: bool

    @property
    def kind(self) -> Kind:
        return Kind.ConstraintQuestion


# Answer Types
@dataclass(frozen=True)
class TrueAnswer(Answer):
    @property
    def kind(self) -> Kind:
        return Kind.TrueAnswer


@dataclass(frozen=True)
class UnsignedAnswer(Answer):
    value: int = 0

    @property
    def kind(self) -> Kind:
        return Kind.UnsignedAnswer


@dataclass(frozen=True)
class StringAnswer(Answer):
    value: str = ""

    @property
    def kind(self) -> Kind:
        return Kind.OperationNameAnswer


@dataclass(frozen=True)
class AttributeAnswer(Answer):
    value: Attribute

    @property
    def kind(self) -> Kind:
        return Kind.AttributeAnswer


@dataclass(frozen=True)
class TypeAnswer(Answer):
    value: TypeAttribute | ArrayAttr[TypeAttribute]

    @property
    def kind(self) -> Kind:
        return Kind.TypeAnswer


# =============================================================================
# Positional Predicates
# =============================================================================


@dataclass
class PositionalPredicate:
    """A predicate applied to a specific position"""

    position: Position
    question: Question
    answer: Answer


# =============================================================================
# Predicate Builder
# =============================================================================


class PredicateBuilder:
    """Utility for constructing predicates and positions"""

    def __init__(self):
        self._position_cache: dict[tuple[Any, ...], Position] = {}

    def get_root(self) -> OperationPosition:
        """Get the root operation position"""
        key = ("root",)
        if key not in self._position_cache:
            self._position_cache[key] = OperationPosition(depth=0, parent=None)
        return cast(OperationPosition, self._position_cache[key])

    def get_operand_defining_op(self, operand_pos: Position) -> OperationPosition:
        """Get the operation that defines an operand"""
        key = ("defining_op", operand_pos)
        if key not in self._position_cache:
            depth = operand_pos.get_operation_depth() + 1
            self._position_cache[key] = OperationPosition(
                depth=depth, parent=operand_pos
            )
        return cast(OperationPosition, self._position_cache[key])

    def get_operand(
        self, op_pos: OperationPosition, operand_num: int
    ) -> OperandPosition:
        """Get an operand position"""
        key = ("operand", op_pos, operand_num)
        if key not in self._position_cache:
            self._position_cache[key] = OperandPosition(
                operand_number=operand_num, parent=op_pos
            )
        return cast(OperandPosition, self._position_cache[key])

    def get_result(self, op_pos: OperationPosition, result_num: int) -> ResultPosition:
        """Get a result position"""
        key = ("result", op_pos, result_num)
        if key not in self._position_cache:
            self._position_cache[key] = ResultPosition(
                result_number=result_num, parent=op_pos
            )
        return cast(ResultPosition, self._position_cache[key])

    def get_attribute(
        self, op_pos: OperationPosition, attr_name: str
    ) -> AttributePosition:
        """Get an attribute position"""
        key = ("attribute", op_pos, attr_name)
        if key not in self._position_cache:
            self._position_cache[key] = AttributePosition(
                attribute_name=attr_name, parent=op_pos
            )
        return cast(AttributePosition, self._position_cache[key])

    def get_type(self, pos: Position) -> TypePosition:
        """Get a type position"""
        key = ("type", pos)
        if key not in self._position_cache:
            self._position_cache[key] = TypePosition(parent=pos)
        return cast(TypePosition, self._position_cache[key])

    # Predicate builders
    def get_is_not_null(self) -> tuple[Question, Answer]:
        return (IsNotNullQuestion(), TrueAnswer())

    def get_operation_name(self, name: str) -> tuple[Question, Answer]:
        return (OperationNameQuestion(), StringAnswer(value=name))

    def get_operand_count(self, count: int) -> tuple[Question, Answer]:
        return (OperandCountQuestion(), UnsignedAnswer(value=count))

    def get_result_count(self, count: int) -> tuple[Question, Answer]:
        return (ResultCountQuestion(), UnsignedAnswer(value=count))

    def get_equal_to(self, other_position: Position) -> tuple[Question, Answer]:
        return (EqualToQuestion(other_position=other_position), TrueAnswer())

    def get_operand_count_at_least(self, count: int) -> tuple[Question, Answer]:
        """Get predicate for minimum operand count (variadic case)"""
        return (OperandCountAtLeastQuestion(), UnsignedAnswer(value=count))

    def get_result_count_at_least(self, count: int) -> tuple[Question, Answer]:
        """Get predicate for minimum result count (variadic case)"""
        return (ResultCountAtLeastQuestion(), UnsignedAnswer(value=count))

    def get_attribute_constraint(
        self, attr_value: Attribute
    ) -> tuple[Question, Answer]:
        """Get predicate for attribute value constraint"""
        return (AttributeConstraintQuestion(), AttributeAnswer(value=attr_value))

    def get_type_constraint(
        self, type_value: TypeAttribute | ArrayAttr[TypeAttribute]
    ) -> tuple[Question, Answer]:
        """Get predicate for type value constraint"""
        return (TypeConstraintQuestion(), TypeAnswer(value=type_value))

    def get_operand_group(
        self, op_pos: OperationPosition, group_num: int, is_variadic: bool
    ) -> OperandGroupPosition:
        """Get an operand group position"""
        key = ("operand_group", op_pos, group_num, is_variadic)
        if key not in self._position_cache:
            self._position_cache[key] = OperandGroupPosition(
                group_number=group_num, is_variadic=is_variadic, parent=op_pos
            )
        return cast(OperandGroupPosition, self._position_cache[key])

    def get_result_group(
        self, op_pos: OperationPosition, group_num: int | None, is_variadic: bool
    ) -> ResultGroupPosition:
        """Get a result group position"""
        key = ("result_group", op_pos, group_num, is_variadic)
        if key not in self._position_cache:
            self._position_cache[key] = ResultGroupPosition(
                group_number=group_num, is_variadic=is_variadic, parent=op_pos
            )
        return cast(ResultGroupPosition, self._position_cache[key])

    def get_all_operands(self, op_pos: OperationPosition) -> OperandGroupPosition:
        """Get position representing all operands of an operation"""
        key = ("operand_group", op_pos, None, True)
        if key not in self._position_cache:
            self._position_cache[key] = OperandGroupPosition(
                group_number=None, is_variadic=True, parent=op_pos
            )
        return cast(OperandGroupPosition, self._position_cache[key])

    def get_all_results(self, op_pos: OperationPosition) -> ResultGroupPosition:
        """Get position representing all results of an operation"""
        key = ("result_group", op_pos, None, True)
        if key not in self._position_cache:
            self._position_cache[key] = ResultGroupPosition(
                group_number=None, is_variadic=True, parent=op_pos
            )
        return cast(ResultGroupPosition, self._position_cache[key])

    def get_attribute_literal(self, value: Attribute) -> AttributeLiteralPosition:
        """Get position for a literal attribute value"""
        key = ("attribute_literal", value)
        if key not in self._position_cache:
            self._position_cache[key] = AttributeLiteralPosition(
                value=value, parent=None
            )
        return cast(AttributeLiteralPosition, self._position_cache[key])

    def get_type_literal(self, value: Attribute) -> TypeLiteralPosition:
        """Get position for a literal type value"""
        key = ("type_literal", value)
        if key not in self._position_cache:
            self._position_cache[key] = TypeLiteralPosition(value=value, parent=None)
        return cast(TypeLiteralPosition, self._position_cache[key])

    def get_constraint(
        self,
        name: str,
        arg_positions: list[Position],
        result_types: list[pdl.AnyPDLType],
        is_negated: bool = False,
    ) -> tuple[Question, Answer]:
        """Get predicate for a native constraint"""
        question = ConstraintQuestion(
            name=name,
            arg_positions=arg_positions,
            result_types=result_types,
            is_negated=is_negated,
        )
        return (question, TrueAnswer())

    def get_constraint_position(
        self, constraint_question: ConstraintQuestion, result_index: int
    ) -> ConstraintPosition:
        """Get position for a constraint result"""
        key = ("constraint_pos", constraint_question, result_index)
        if key not in self._position_cache:
            self._position_cache[key] = ConstraintPosition(
                constraint=constraint_question, result_index=result_index, parent=None
            )
        return cast(ConstraintPosition, self._position_cache[key])


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

    success_node: MatcherNode | None = None


@dataclass
class SwitchNode(MatcherNode):
    """Multi-way switch node"""

    children: dict[Answer, MatcherNode | None] = field(default_factory=lambda: {})


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
    connector: Any  # Value that connects the roots


class OptimalBranching:
    """Edmonds' optimal branching algorithm for minimum spanning arborescence"""

    def __init__(self, graph: dict[Any, dict[Any, RootOrderingEntry]], root: Any):
        self.graph = graph
        self.root = root
        self.parents: dict[Any, Any] = {}

    def solve(self) -> int:
        """Solve for optimal branching, returns total cost"""
        self.parents.clear()
        self.parents[self.root] = None
        total_cost = 0

        # Find minimum incoming edge for each node
        for target in self.graph:
            if target in self.parents:
                continue

            # Follow chain of minimum parents
            node = target
            parent_depths: dict[Any, int] = {}

            while node not in self.parents:
                if node not in self.graph:
                    break

                # Find best parent
                best_parent = None
                best_cost = None

                for source, entry in self.graph[node].items():
                    if best_parent is None or entry.cost < cast(
                        tuple[int, int], best_cost
                    ):
                        best_parent = source
                        best_cost = entry.cost

                if best_parent is None:
                    break
                assert best_cost is not None

                self.parents[node] = best_parent
                parent_depths[node] = best_cost[0]
                total_cost += best_cost[0]
                node = best_parent

            # Check for cycles and contract if needed
            if node in parent_depths:
                cycle = self._get_cycle(node)
                total_cost += self._contract_cycle(cycle, parent_depths)

        return total_cost

    def _get_cycle(self, start: Any) -> list[Any]:
        """Get cycle starting from the given node"""
        cycle: list[Any] = []
        node = start
        while True:
            cycle.append(node)
            node = self.parents[node]
            if node == start:
                break
        return cycle

    def _contract_cycle(self, cycle: list[Any], parent_depths: dict[Any, int]) -> int:
        """Contract a cycle in the graph"""
        # Simplified cycle contraction
        cycle_cost = sum(parent_depths[node] for node in cycle)

        # Update parents to break cycle
        for i, node in enumerate(cycle[:-1]):
            self.parents[node] = cycle[i + 1]

        return cycle_cost


# =============================================================================
# Pattern Analysis
# =============================================================================


class PatternAnalyzer:
    """Analyzes PDL patterns and extracts predicates"""

    def __init__(self, builder: PredicateBuilder):
        self.builder = builder

    def detect_roots(self, pattern: pdl.PatternOp) -> list[pdl.OperationOp]:
        """Detect root operations in a pattern"""
        used: set[pdl.OperationOp] = set()

        for operation_op in pattern.body.ops:
            if not isinstance(operation_op, pdl.OperationOp):
                continue
            for operand in operation_op.operand_values:
                result_op = operand.owner
                if isinstance(result_op, pdl.ResultOp | pdl.ResultsOp):
                    assert isinstance(
                        used_op := result_op.parent_.owner, pdl.OperationOp
                    )
                    used.add(used_op)

        rewriter = pattern.body.block.last_op
        assert isinstance(rewriter, pdl.RewriteOp)
        if rewriter.root is not None:
            assert isinstance(root := rewriter.root.owner, pdl.OperationOp)
            if root in used:
                used.remove(root)

        roots = [
            op
            for op in pattern.body.ops
            if isinstance(op, pdl.OperationOp) and op not in used
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

                q, a = self.builder.get_equal_to(shallower_pos)
                predicates.append(PositionalPredicate(deeper_pos, q, a))
            return predicates

        inputs[value] = position

        # Dispatch based on position type (not value type!)
        if isinstance(position, AttributePosition):
            predicates.extend(
                self._extract_attribute_predicates(value, position, inputs)
            )
        elif isinstance(position, OperationPosition):
            predicates.extend(
                self._extract_operation_predicates(
                    value, position, inputs, ignore_operand
                )
            )
        elif isinstance(position, TypePosition):
            predicates.extend(self._extract_type_predicates(value, position, inputs))
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
        attr_value: Operation | SSAValue,
        attr_pos: AttributePosition,
        inputs: dict[SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates for an attribute"""
        predicates: list[PositionalPredicate] = []

        q, a = self.builder.get_is_not_null()
        predicates.append(PositionalPredicate(attr_pos, q, a))

        # Get the actual attribute operation
        if isinstance(attr_value, SSAValue):
            attr_op = attr_value.owner
        else:
            attr_op = attr_value

        if isinstance(attr_op, pdl.AttributeOp):
            if attr_op.value_type:
                type_pos = self.builder.get_type(attr_pos)
                predicates.extend(
                    self.extract_tree_predicates(attr_op.value_type, type_pos, inputs)
                )

            elif attr_op.value:
                q, a = self.builder.get_attribute_constraint(attr_op.value)
                predicates.append(PositionalPredicate(attr_pos, q, a))

        return predicates

    def _extract_operation_predicates(
        self,
        op_value: Operation | SSAValue,
        op_pos: OperationPosition,
        inputs: dict[SSAValue, Position],
        ignore_operand: int | None = None,
    ) -> list[PositionalPredicate]:
        """Extract predicates for an operation"""
        predicates: list[PositionalPredicate] = []

        if not op_pos.is_root():
            q, a = self.builder.get_is_not_null()
            predicates.append(PositionalPredicate(op_pos, q, a))

        # Get the actual operation
        if isinstance(op_value, SSAValue):
            assert isinstance(op_value.owner, Operation)
            op_value = op_value.owner

        if not isinstance(op_value, pdl.OperationOp):
            return predicates

        # Operation name check
        if op_value.opName:
            op_name = op_value.opName.data
            q, a = self.builder.get_operation_name(op_name)
            predicates.append(PositionalPredicate(op_pos, q, a))

        operands = op_value.operand_values
        min_operands = self._get_num_non_range_values(operands)
        if min_operands != len(operands):
            # Has variadic operands - check minimum
            if min_operands > 0:
                q, a = self.builder.get_operand_count_at_least(min_operands)
                predicates.append(PositionalPredicate(op_pos, q, a))
        else:
            # All non-variadic - check exact count
            q, a = self.builder.get_operand_count(min_operands)
            predicates.append(PositionalPredicate(op_pos, q, a))

        types = op_value.type_values
        min_results = self._get_num_non_range_values(types)
        if min_results == len(types):
            # All non-variadic - check exact count
            q, a = self.builder.get_result_count(len(types))
            predicates.append(PositionalPredicate(op_pos, q, a))
        elif min_results > 0:
            # Has variadic results - check minimum
            q, a = self.builder.get_result_count_at_least(min_results)
            predicates.append(PositionalPredicate(op_pos, q, a))

        # Process attributes
        for attr_name, attr in zip(
            op_value.attributeValueNames, op_value.attribute_values
        ):
            attr_pos = self.builder.get_attribute(op_pos, attr_name.data)
            predicates.extend(self.extract_tree_predicates(attr, attr_pos, inputs))

        if len(operands) == 1 and isinstance(operands[0].type, pdl.RangeType):
            # Special case: single variadic operand represents all operands
            if op_pos.is_root() or op_pos.is_operand_defining_op():
                all_operands_pos = self.builder.get_all_operands(op_pos)
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
                    operand_pos = self.builder.get_operand_group(op_pos, i, is_variadic)
                else:
                    operand_pos = self.builder.get_operand(op_pos, i)

                predicates.extend(
                    self.extract_tree_predicates(operand, operand_pos, inputs)
                )

        if len(types) == 1 and isinstance(types[0].type, pdl.RangeType):
            # Single variadic result represents all results
            all_results_pos = self.builder.get_all_results(op_pos)
            type_pos = self.builder.get_type(all_results_pos)
            predicates.extend(self.extract_tree_predicates(types[0], type_pos, inputs))
        else:
            # Process individual results
            found_variable_length = False
            for i, type_value in enumerate(types):
                is_variadic = isinstance(type_value.type, pdl.RangeType)
                found_variable_length = found_variable_length or is_variadic

                # Switch to group-based positioning after first variadic
                if found_variable_length:
                    result_pos = self.builder.get_result_group(op_pos, i, is_variadic)
                else:
                    result_pos = self.builder.get_result(op_pos, i)

                # Add not-null check for each result
                q, a = self.builder.get_is_not_null()
                predicates.append(PositionalPredicate(result_pos, q, a))

                # Process the result type
                type_pos = self.builder.get_type(result_pos)
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
                q, a = self.builder.get_is_not_null()
                predicates.append(PositionalPredicate(operand_pos, q, a))
            elif (
                isinstance(operand_pos, OperandGroupPosition)
                and operand_pos.group_number is not None
            ):
                q, a = self.builder.get_is_not_null()
                predicates.append(PositionalPredicate(operand_pos, q, a))

            if defining_op.value_type:
                type_pos = self.builder.get_type(operand_pos)
                predicates.extend(
                    self.extract_tree_predicates(
                        defining_op.value_type, type_pos, inputs
                    )
                )

        elif isinstance(defining_op, pdl.ResultOp | pdl.ResultsOp):
            index = defining_op.index
            if index is not None:
                q, a = self.builder.get_is_not_null()
                predicates.append(PositionalPredicate(operand_pos, q, a))

            # Get the parent operation position
            parent_op = defining_op.parent_
            defining_op_pos = self.builder.get_operand_defining_op(operand_pos)

            # Parent operation should not be null
            q, a = self.builder.get_is_not_null()
            predicates.append(PositionalPredicate(defining_op_pos, q, a))

            if isinstance(defining_op, pdl.ResultOp):
                result_pos = self.builder.get_result(
                    defining_op_pos, index.value.data if index else 0
                )
            else:  # ResultsOp
                result_pos = self.builder.get_result_group(
                    defining_op_pos, index.value.data if index else None, is_variadic
                )

            q, a = self.builder.get_equal_to(operand_pos)
            predicates.append(PositionalPredicate(result_pos, q, a))

            # Recursively process the parent operation
            predicates.extend(
                self.extract_tree_predicates(parent_op, defining_op_pos, inputs)
            )

        return predicates

    def _extract_type_predicates(
        self,
        type_value: Operation | SSAValue,
        type_pos: TypePosition,
        inputs: dict[SSAValue, Position],
    ) -> list[PositionalPredicate]:
        """Extract predicates for a type"""
        predicates: list[PositionalPredicate] = []

        # Get the actual type operation
        if isinstance(type_value, SSAValue):
            type_op = type_value.owner
        else:
            type_op = type_value

        if isinstance(type_op, pdl.TypeOp) and type_op.constantType:
            q, a = self.builder.get_type_constraint(type_op.constantType)
            predicates.append(PositionalPredicate(type_pos, q, a))
        elif isinstance(type_op, pdl.TypesOp) and type_op.constantTypes:
            q, a = self.builder.get_type_constraint(type_op.constantTypes)
            predicates.append(PositionalPredicate(type_pos, q, a))

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
                if op not in inputs:
                    if op.value:
                        # Create literal position for constant attribute
                        attr_pos = self.builder.get_attribute_literal(op.value)
                        inputs[op.output] = attr_pos

            elif isinstance(op, pdl.ApplyNativeConstraintOp):
                # Collect all argument positions
                arg_positions: list[Position] = []
                for arg in op.args:
                    assert (pos := inputs.get(arg)) is not None
                    arg_positions.append(pos)

                # Find the furthest position (deepest)
                furthest_pos = max(
                    arg_positions, key=lambda p: p.get_operation_depth() if p else 0
                )

                # Create the constraint predicate
                result_types = [r.type for r in op.res]
                # TODO: is_negated is not part of the dialect definition yet
                is_negated = False
                q, a = self.builder.get_constraint(
                    op.name, arg_positions, result_types, is_negated
                )

                # Register positions for constraint results
                for i, result in enumerate(op.results):
                    assert isinstance(q, ConstraintQuestion)
                    constraint_pos = self.builder.get_constraint_position(q, i)
                    existing = inputs.get(result)
                    if existing:
                        # Add equality constraint if result already has a position
                        deeper, shallower = (
                            (constraint_pos, existing)
                            if constraint_pos.get_operation_depth()
                            > existing.get_operation_depth()
                            else (existing, constraint_pos)
                        )
                        eq_q, eq_a = self.builder.get_equal_to(shallower)
                        predicates.append(PositionalPredicate(deeper, eq_q, eq_a))
                    else:
                        inputs[result] = constraint_pos

                predicates.append(PositionalPredicate(furthest_pos, q, a))

            elif isinstance(op, pdl.ResultOp):
                # Ensure result exists
                if op.val not in inputs:
                    assert isinstance(op.parent_.owner, pdl.OperationOp)
                    parent_pos = inputs.get(op.parent_.owner.op)
                    if parent_pos and isinstance(parent_pos, OperationPosition):
                        result_pos = self.builder.get_result(
                            parent_pos, op.index.value.data
                        )
                        q, a = self.builder.get_is_not_null()
                        predicates.append(PositionalPredicate(result_pos, q, a))

            elif isinstance(op, pdl.ResultsOp):
                # Handle result groups
                if op.val not in inputs:
                    assert isinstance(op.parent_.owner, pdl.OperationOp)
                    parent_pos = inputs.get(op.parent_.owner.op)
                    if parent_pos and isinstance(parent_pos, OperationPosition):
                        is_variadic = isinstance(op.val.type, pdl.RangeType)
                        index = op.index.value.data if op.index else None
                        result_pos = self.builder.get_result_group(
                            parent_pos, index, is_variadic
                        )
                        if index is not None:
                            q, a = self.builder.get_is_not_null()
                            predicates.append(PositionalPredicate(result_pos, q, a))

            elif isinstance(op, pdl.TypeOp):
                # Handle constant types
                if op not in inputs and op.constantType:
                    type_pos = self.builder.get_type_literal(op.constantType)
                    inputs[op.result] = type_pos

            elif isinstance(op, pdl.TypesOp):
                # Handle constant type arrays
                if op not in inputs and op.constantTypes:
                    type_pos = self.builder.get_type_literal(op.constantTypes)
                    inputs[op.result] = type_pos

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
    pattern_answers: dict[Any, Answer] = field(default_factory=lambda: {})

    def __lt__(self, other: "OrderedPredicate") -> bool:
        """Comparison for priority ordering"""
        return (
            self.primary_score,
            self.secondary_score,
            -self.position.get_operation_depth(),  # Prefer lower depth
            -self.position.kind.value,  # Position dependency
            -self.question.kind.value,  # Predicate dependency
            -self.tie_breaker,  # Deterministic order
        ) > (
            other.primary_score,
            other.secondary_score,
            -other.position.get_operation_depth(),
            -other.position.kind.value,
            -other.question.kind.value,
            -other.tie_breaker,
        )


class PredicateTreeBuilder:
    """Builds optimized predicate matching trees"""

    def __init__(self):
        self.analyzer = PatternAnalyzer(PredicateBuilder())

    def build_predicate_tree(self, patterns: list[Any]) -> MatcherNode:
        """Build optimized matcher tree from multiple patterns"""

        # Extract predicates for all patterns
        all_pattern_predicates: list[tuple[Any, list[PositionalPredicate]]] = []
        for pattern in patterns:
            predicates = self._extract_pattern_predicates(pattern)
            all_pattern_predicates.append((pattern, predicates))

        # Create ordered predicates with frequency analysis
        ordered_predicates = self._create_ordered_predicates(all_pattern_predicates)

        # Sort predicates by priority
        sorted_predicates = sorted(ordered_predicates.values())

        # Build matcher tree by propagating patterns
        root_node = None
        for pattern, predicates in all_pattern_predicates:
            pattern_predicate_set = {
                (pred.position, pred.question): pred for pred in predicates
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
    ) -> list[PositionalPredicate]:
        """Extract all predicates for a single pattern"""
        roots = self.analyzer.detect_roots(pattern)

        # For simplicity, use first root (in real implementation,
        # would use optimal root selection)
        if len(roots) > 1:
            raise NotImplementedError("Multiple roots not yet supported")
        root = roots[0] if roots else None
        if not root:
            return []

        inputs: dict[SSAValue, Position] = {}
        root_pos = self.analyzer.builder.get_root()

        predicates = self.analyzer.extract_tree_predicates(root.op, root_pos, inputs)

        predicates.extend(self.analyzer.extract_non_tree_predicates(pattern, inputs))

        return predicates

    def _create_ordered_predicates(
        self, all_pattern_predicates: list[tuple[Any, list[PositionalPredicate]]]
    ) -> dict[tuple[Position, Question], OrderedPredicate]:
        """Create ordered predicates with frequency analysis"""
        predicate_map: dict[tuple[Position, Question], OrderedPredicate] = {}
        tie_breaker = 0

        # Collect unique predicates
        for pattern, predicates in all_pattern_predicates:
            for pred in predicates:
                key = (pred.position, pred.question)

                if key not in predicate_map:
                    ordered_pred = OrderedPredicate(
                        position=pred.position,
                        question=pred.question,
                        tie_breaker=tie_breaker,
                    )
                    predicate_map[key] = ordered_pred
                    tie_breaker += 1

                # Track pattern answers and increment frequency
                predicate_map[key].pattern_answers[pattern] = pred.answer
                predicate_map[key].primary_score += 1

        # Calculate secondary scores
        for pattern, predicates in all_pattern_predicates:
            pattern_primary_sum = 0
            seen_keys: set[tuple[Position, Question]] = (
                set()
            )  # Track unique keys per pattern

            # First pass: collect unique predicates for this pattern
            for pred in predicates:
                key = (pred.position, pred.question)
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
        pattern: Any,
        pattern_predicates: dict[tuple[Position, Question], PositionalPredicate],
        sorted_predicates: list[OrderedPredicate],
        predicate_index: int,
    ) -> MatcherNode:
        """Propagate a pattern through the predicate tree"""

        # Base case: reached end of predicates
        if predicate_index >= len(sorted_predicates):
            return SuccessNode(pattern=pattern, root=None, failure_node=node)

        current_predicate = sorted_predicates[predicate_index]
        pred_key = (current_predicate.position, current_predicate.question)

        # Skip predicates not in this pattern
        if pred_key not in pattern_predicates:
            return self._propagate_pattern(
                node,
                pattern,
                pattern_predicates,
                sorted_predicates,
                predicate_index + 1,
            )

        # Create or match existing node
        if node is None:
            # Create new switch node
            node = SwitchNode(
                position=current_predicate.position, question=current_predicate.question
            )

        if self._nodes_match(node, current_predicate):
            # Continue down matching path
            pattern_answer = pattern_predicates[pred_key].answer

            if isinstance(node, SwitchNode):
                if pattern_answer not in node.children:
                    node.children[pattern_answer] = None

                node.children[pattern_answer] = self._propagate_pattern(
                    node.children[pattern_answer],
                    pattern,
                    pattern_predicates,
                    sorted_predicates,
                    predicate_index + 1,
                )

        else:
            # Divergence - continue down failure path
            node.failure_node = self._propagate_pattern(
                node.failure_node,
                pattern,
                pattern_predicates,
                sorted_predicates,
                predicate_index,
            )

        return node

    def _nodes_match(self, node: MatcherNode, predicate: OrderedPredicate) -> bool:
        """Check if node matches the given predicate"""
        return (
            node.position == predicate.position and node.question == predicate.question
        )

    def _insert_exit_node(self, root: MatcherNode) -> MatcherNode:
        """Insert exit node at end of failure paths"""
        root.failure_node = ExitNode()
        return root

    def _optimize_tree(self, root: MatcherNode) -> MatcherNode:
        """Optimize the tree by collapsing single-child switches to bools"""
        # Recursively optimize children
        if isinstance(root, SwitchNode):
            for answer in root.children:
                child_node = root.children[answer]
                if child_node is not None:
                    root.children[answer] = self._optimize_tree(child_node)
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

    def __init__(self, matcher_func: pdl_interp.FuncOp):
        self.matcher_func = matcher_func
        self.values: dict[Position, SSAValue] = {}
        self.failure_block_stack: list[Block] = []
        self.builder = Builder(InsertPoint.at_start(matcher_func.body.block))
        self.constraint_op_map: dict[
            ConstraintQuestion, pdl_interp.ApplyConstraintOp
        ] = {}

    def lower(self, patterns: list[pdl.PatternOp]) -> None:
        """Lower PDL patterns to PDL interpreter"""

        # Build the predicate tree
        tree_builder = PredicateTreeBuilder()
        root = tree_builder.build_predicate_tree(patterns)

        # Get the entry block and add root operation argument
        entry_block = self.matcher_func.body.block

        # The first argument is the root operation
        builder = PredicateBuilder()
        root_pos = builder.get_root()
        self.values[root_pos] = entry_block.args[0]

        # Generate the matcher
        first_matcher_block = self.generate_matcher(root, self.matcher_func.body)

        # Merge first matcher block into entry if different
        if first_matcher_block != entry_block:
            entry_block.add_ops(first_matcher_block.ops)
            first_matcher_block.erase()

    def generate_matcher(
        self, node: MatcherNode, region: Region, block: Block | None = None
    ) -> Block:
        """Generate PDL interpreter operations for a matcher node"""

        # Create block if needed
        if block is None:
            block = Block()
            region.add_block(block)

        # Handle exit node - just add finalize
        if isinstance(node, ExitNode):
            finalize_op = pdl_interp.FinalizeOp()
            self.builder.insert_op(finalize_op, InsertPoint.at_end(block))
            return block

        # Handle failure node
        failure_block = None
        if node.failure_node:
            failure_block = self.generate_matcher(node.failure_node, region)
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
            # Handle getting users of a value
            # This would require GetUsersOp which may not be implemented
            # For now, just use parent value
            raise NotImplementedError("UsersPosition not implemented in lowering")

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
            check_op = pdl_interp.ApplyConstraintOp(
                question.name,
                args,
                success_block,
                failure_block,
                question.result_types,
                question.is_negated,
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

            # Build cascading checks
            self.failure_block_stack.append(default_dest)

            for answer, child_node in sorted_children:
                if child_node:
                    child_block = self.generate_matcher(child_node, region)
                    predicate_block = Block()
                    region.add_block(predicate_block)

                    self.builder.insertion_point = InsertPoint.at_end(predicate_block)
                    assert isinstance(answer, UnsignedAnswer)

                    if isinstance(question, OperandCountAtLeastQuestion):
                        check_op = pdl_interp.CheckOperandCountOp(
                            val, answer.value, child_block, default_dest, True
                        )
                    else:  # ResultCountAtLeastQuestion
                        check_op = pdl_interp.CheckResultCountOp(
                            val, answer.value, child_block, default_dest, True
                        )

                    self.builder.insert(check_op)
                    self.failure_block_stack[-1] = predicate_block

            # Move operations from first predicate block to current block
            first_predicate_block = self.failure_block_stack.pop()
            self.builder.insertion_point = InsertPoint.at_end(block)
            for op in first_predicate_block.ops:
                block.add_op(op)
            first_predicate_block.erase()
            return

        # Generate child blocks and collect case values
        case_blocks: list[Block] = []
        case_values: list[Any] = []

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
            raise NotImplementedError(
                "pdl_interp.switch_attributes is not yet implemented"
            )
            switch_op = pdl_interp.SwitchAttributeOp(
                switch_attr, val, default_dest, case_blocks
            )

        else:
            raise NotImplementedError(f"Unhandled question type {type(question)}")

        self.builder.insert(switch_op)

    def generate_success_node(self, node: SuccessNode, block: Block) -> None:
        """Generate operations for a successful match"""

        self.builder.insertion_point = InsertPoint.at_end(block)

        # In the full implementation, we would:
        # 1. Generate the rewriter function
        # 2. Collect used match values
        # 3. Create RecordMatchOp with all necessary info

        # For now, create a simplified record match
        # Get the pattern and root
        pattern = node.pattern
        _root = node.root

        # Create a dummy rewriter reference
        rewriter_ref = StringAttr("dummy_rewriter")

        # Collect matched values (simplified)
        matched_values: list[SSAValue] = []
        for pos, val in self.values.items():
            if isinstance(pos, OperationPosition) and pos.is_root():
                matched_values.append(val)
                break

        # Create the record match operation
        if False and matched_values:
            # Get root operation name if available
            root_kind = None
            if hasattr(pattern, "root_op_name"):
                root_kind = pattern.root_op_name

            benefit = IntegerAttr.from_int_and_width(1, 16)  # Default benefit

            record_op = pdl_interp.RecordMatchOp(
                rewriter_ref,
                root_kind if root_kind else "",
                None,  # generated_ops
                benefit,
                [],  # inputs
                matched_values,  # matched_ops
                self.failure_block_stack[-1] if self.failure_block_stack else None,
            )
            self.builder.insert(record_op)
        else:
            # If no match values, just finalize
            finalize_op = pdl_interp.FinalizeOp()
            self.builder.insert(finalize_op)


def lower_pdl_to_pdl_interp(module: ModuleOp, matcher_func: pdl_interp.FuncOp) -> None:
    """Main entry point to lower PDL patterns to PDL interpreter"""

    # Collect all patterns
    patterns = [op for op in module.body.ops if isinstance(op, pdl.PatternOp)]

    # Create generator and lower
    generator = MatcherGenerator(matcher_func)
    generator.lower(patterns)


# =============================================================================
# Main Transformation Pipeline
# =============================================================================


class ConvertPDLToPDLInterpPass(ModulePass):
    """
    Pass to convert PDL operations to PDL interpreter operations.
    This is a somewhat faithful port of the implementation in MLIR, but it may not generate the same exact results.

    Currently the pass only generates the matcher code (not the rewriter), and does not yet support multiple patterns.
    """

    name = "convert-pdl-to-pdl-interp"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        patterns = [
            pattern for pattern in op.body.ops if isinstance(pattern, pdl.PatternOp)
        ]

        if len(patterns) != 1:
            raise NotImplementedError("Currently only single pattern is supported")

        y = PredicateTreeBuilder().build_predicate_tree(patterns)
        matcher_func = pdl_interp.FuncOp("matcher", ((pdl.OperationType(),), ()))
        mg = MatcherGenerator(matcher_func)
        mg.values[OperationPosition(None, depth=0)] = matcher_func.body.block.args[0]
        mg.generate_matcher(y, matcher_func.body)

        rewriter = Rewriter()
        rewriter.replace_op(patterns[0], matcher_func)
