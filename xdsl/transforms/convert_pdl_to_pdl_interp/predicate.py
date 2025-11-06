"""
This file implements some of the core data structures used in the pdl-to-pdl-interp conversion.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Optional

from xdsl.dialects import pdl
from xdsl.dialects.builtin import (
    ArrayAttr,
    TypeAttribute,
)
from xdsl.ir import (
    Attribute,
)


@dataclass(frozen=True)
class Position(ABC):
    """The position class encodes a location in a pattern.
    Each pattern has a root position. From there, other positions can be reached representing operands, results, and more.
    """

    parent: Optional["Position"] = None

    def get_operation_depth(self) -> int:
        """Returns depth of first ancestor operation position"""
        op = self.get_base_operation()
        return op.depth

    def get_base_operation(self) -> "OperationPosition":
        pos = self
        while not isinstance(pos, OperationPosition):
            assert pos.parent is not None
            pos = pos.parent
        return pos


@dataclass(frozen=True, kw_only=True)
class OperationPosition(Position):
    """Represents an operation in the IR"""

    depth: int

    def is_root(self) -> bool:
        return self.depth == 0

    def is_operand_defining_op(self) -> bool:
        return isinstance(self.parent, OperandPosition | OperandGroupPosition)

    def __repr__(self):
        if self.is_root():
            return "root"
        else:
            return self.parent.__repr__() + ".defining_op"

    def get_operand(self, operand_num: int) -> "OperandPosition":
        return OperandPosition(operand_number=operand_num, parent=self)

    def get_operand_group(
        self, group_num: int, is_variadic: bool
    ) -> "OperandGroupPosition":
        return OperandGroupPosition(
            group_number=group_num, is_variadic=is_variadic, parent=self
        )

    def get_all_operands(self) -> "OperandGroupPosition":
        return OperandGroupPosition(group_number=None, is_variadic=True, parent=self)

    def get_result(self, result_num: int) -> "ResultPosition":
        return ResultPosition(result_number=result_num, parent=self)

    def get_result_group(
        self, group_num: int | None, is_variadic: bool
    ) -> "ResultGroupPosition":
        return ResultGroupPosition(
            group_number=group_num, is_variadic=is_variadic, parent=self
        )

    def get_all_results(self) -> "ResultGroupPosition":
        return ResultGroupPosition(group_number=None, is_variadic=True, parent=self)

    def get_attribute(self, attr_name: str) -> "AttributePosition":
        return AttributePosition(attribute_name=attr_name, parent=self)

    def get_attribute_literal(self, value: Attribute) -> "AttributeLiteralPosition":
        return AttributeLiteralPosition(value=value, parent=None)


class ValuePosition(Position):
    def get_defining_op(self) -> OperationPosition:
        return OperationPosition(depth=self.get_operation_depth() + 1, parent=self)

    def get_type(self) -> "TypePosition":
        return TypePosition(parent=self)

    def get_type_literal(self, value: Attribute) -> "TypeLiteralPosition":
        return TypeLiteralPosition(value=value, parent=None)

    def get_users(self, use_representative: bool) -> "UsersPosition":
        return UsersPosition(parent=self, use_representative=use_representative)


@dataclass(frozen=True, kw_only=True)
class OperandPosition(ValuePosition):
    """Represents an operand of an operation"""

    operand_number: int

    def __repr__(self):
        return f"{self.parent.__repr__()}.operand[{self.operand_number}]"


@dataclass(frozen=True, kw_only=True)
class OperandGroupPosition(Position):
    """Represents a group of operands"""

    group_number: int | None
    is_variadic: bool

    def get_defining_op(self) -> "OperationPosition":
        return OperationPosition(depth=self.get_operation_depth() + 1, parent=self)

    def get_type(self) -> "TypePosition":
        return TypePosition(parent=self)


@dataclass(frozen=True, kw_only=True)
class ResultPosition(ValuePosition):
    """Represents a result of an operation"""

    result_number: int

    def __repr__(self):
        return f"{self.parent.__repr__()}.result[{self.result_number}]"


@dataclass(frozen=True, kw_only=True)
class AttributePosition(Position):
    """Represents an attribute of an operation"""

    attribute_name: str

    def __repr__(self):
        return f"{self.parent.__repr__()}.attribute[{self.attribute_name}]"

    def get_type(self) -> "TypePosition":
        return TypePosition(parent=self)


@dataclass(frozen=True)
class TypePosition(Position):
    """Represents the type of a value"""

    def __repr__(self):
        return f"{self.parent.__repr__()}.type"

    def get_type_literal(self, value: Attribute) -> "TypeLiteralPosition":
        return TypeLiteralPosition(value=value, parent=None)


@dataclass(frozen=True, kw_only=True)
class UsersPosition(Position):
    """Represents users of a value"""

    use_representative: bool

    def get_for_each(self, for_each_id: int) -> "ForEachPosition":
        return ForEachPosition(parent=self, id=for_each_id)


@dataclass(frozen=True, kw_only=True)
class ForEachPosition(Position):
    """Represents an iterative choice of an operation from a set of users."""

    id: int

    def get_passthrough_op(self) -> "OperationPosition":
        return OperationPosition(depth=self.get_operation_depth() + 1, parent=self)


@dataclass(frozen=True, kw_only=True)
class ResultGroupPosition(Position):
    """Represents a group of results"""

    group_number: int | None
    is_variadic: bool

    def get_type(self) -> "TypePosition":
        return TypePosition(parent=self)


@dataclass(frozen=True, kw_only=True)
class AttributeLiteralPosition(Position):
    """Represents a literal attribute value"""

    value: Attribute


@dataclass(frozen=True, kw_only=True)
class TypeLiteralPosition(Position):
    """Represents a literal type value"""

    value: Attribute  # Can be a single type or array of types

    @staticmethod
    def get_type_literal(value: Attribute) -> "TypeLiteralPosition":
        return TypeLiteralPosition(value=value, parent=None)


@dataclass(frozen=True, kw_only=True)
class ConstraintPosition(Position):
    """Represents a result from a constraint"""

    constraint: "ConstraintQuestion"
    result_index: int

    @staticmethod
    def get_constraint(
        constraint_question: "ConstraintQuestion", result_index: int
    ) -> "ConstraintPosition":
        return ConstraintPosition(
            constraint=constraint_question, result_index=result_index, parent=None
        )


POSITION_COSTS = {
    OperationPosition: 1,
    OperandPosition: 2,
    OperandGroupPosition: 3,
    AttributePosition: 4,
    ConstraintPosition: 5,
    ResultPosition: 6,
    ResultGroupPosition: 7,
    TypePosition: 8,
    AttributeLiteralPosition: 9,
    TypeLiteralPosition: 10,
    UsersPosition: 11,
    ForEachPosition: 12,
}
"""
Different position types are ranked by priority.
A lower cost means a higher priority.
This is used to decide which position to branch on first when evaluating predicates.
"""

# =============================================================================
# Predicate System - Questions and Answers
# =============================================================================


@dataclass(frozen=True)
class Question:
    """Represents a question/check to perform"""

    pass


@dataclass(frozen=True)
class Answer:
    """Represents an expected answer to a question"""

    pass


@dataclass()
class Predicate:
    """Base predicate class"""

    q: Question
    a: Answer

    @staticmethod
    def get_is_not_null() -> "Predicate":
        return Predicate(IsNotNullQuestion(), TrueAnswer())

    @staticmethod
    def get_operation_name(name: str) -> "Predicate":
        return Predicate(OperationNameQuestion(), StringAnswer(value=name))

    @staticmethod
    def get_operand_count(count: int) -> "Predicate":
        return Predicate(OperandCountQuestion(), UnsignedAnswer(value=count))

    @staticmethod
    def get_result_count(count: int) -> "Predicate":
        return Predicate(ResultCountQuestion(), UnsignedAnswer(value=count))

    @staticmethod
    def get_equal_to(other_position: Position) -> "Predicate":
        return Predicate(EqualToQuestion(other_position=other_position), TrueAnswer())

    @staticmethod
    def get_operand_count_at_least(count: int) -> "Predicate":
        """Get predicate for minimum operand count (variadic case)"""
        return Predicate(OperandCountAtLeastQuestion(), UnsignedAnswer(value=count))

    @staticmethod
    def get_result_count_at_least(count: int) -> "Predicate":
        """Get predicate for minimum result count (variadic case)"""
        return Predicate(ResultCountAtLeastQuestion(), UnsignedAnswer(value=count))

    @staticmethod
    def get_attribute_constraint(attr_value: Attribute) -> "Predicate":
        """Get predicate for attribute value constraint"""
        return Predicate(
            AttributeConstraintQuestion(), AttributeAnswer(value=attr_value)
        )

    @staticmethod
    def get_type_constraint(
        type_value: TypeAttribute | ArrayAttr[TypeAttribute],
    ) -> "Predicate":
        """Get predicate for type value constraint"""
        return Predicate(TypeConstraintQuestion(), TypeAnswer(value=type_value))

    @staticmethod
    def get_constraint(
        name: str,
        arg_positions: tuple[Position, ...],
        result_types: tuple[pdl.AnyPDLType, ...],
        is_negated: bool = False,
    ) -> "Predicate":
        """Get predicate for a native constraint"""
        question = ConstraintQuestion(
            name=name,
            arg_positions=tuple(arg_positions),
            result_types=tuple(result_types),
            is_negated=is_negated,
        )
        return Predicate(question, TrueAnswer())


# Question Types
@dataclass(frozen=True)
class IsNotNullQuestion(Question):
    pass


@dataclass(frozen=True)
class OperationNameQuestion(Question):
    pass


@dataclass(frozen=True)
class OperandCountQuestion(Question):
    pass


@dataclass(frozen=True)
class ResultCountQuestion(Question):
    pass


@dataclass(frozen=True)
class EqualToQuestion(Question):
    other_position: Position


@dataclass(frozen=True)
class OperandCountAtLeastQuestion(Question):
    pass


@dataclass(frozen=True)
class ResultCountAtLeastQuestion(Question):
    pass


@dataclass(frozen=True)
class AttributeConstraintQuestion(Question):
    pass


@dataclass(frozen=True)
class TypeConstraintQuestion(Question):
    pass


@dataclass(frozen=True)
class ConstraintQuestion(Question):
    """Represents a native constraint check"""

    name: str
    arg_positions: tuple[Position, ...]
    result_types: tuple[pdl.AnyPDLType, ...]
    is_negated: bool


QUESTION_COSTS = {
    IsNotNullQuestion: 1,
    OperationNameQuestion: 2,
    OperandCountAtLeastQuestion: 3,
    OperandCountQuestion: 4,
    ResultCountAtLeastQuestion: 5,
    ResultCountQuestion: 6,
    EqualToQuestion: 7,
    AttributeConstraintQuestion: 8,
    TypeConstraintQuestion: 9,
    ConstraintQuestion: 10,
}
"""
Different question types are ranked by priority.
A lower cost means a higher priority.
This is used to decide which question to branch on first when evaluating predicates.
"""


def get_position_cost(position: Position) -> int:
    """Get cost for a position type, with fallback for unknown types."""
    assert (t := type(position)) in POSITION_COSTS
    return POSITION_COSTS[t]


def get_question_cost(question: Question) -> int:
    """Get cost for a question type, with fallback for unknown types."""
    assert (t := type(question)) in QUESTION_COSTS
    return QUESTION_COSTS[t]


# Answer Types
@dataclass(frozen=True)
class TrueAnswer(Answer):
    pass


@dataclass(frozen=True)
class FalseAnswer(Answer):
    pass


@dataclass(frozen=True)
class UnsignedAnswer(Answer):
    value: int


@dataclass(frozen=True)
class StringAnswer(Answer):
    value: str


@dataclass(frozen=True)
class AttributeAnswer(Answer):
    value: Attribute


@dataclass(frozen=True)
class TypeAnswer(Answer):
    value: TypeAttribute | ArrayAttr[TypeAttribute]


# =============================================================================
# Positional Predicates
# =============================================================================


@dataclass
class PositionalPredicate(Predicate):
    """A predicate applied to a specific position"""

    position: Position
