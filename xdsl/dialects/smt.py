from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import ClassVar, TypeAlias

from typing_extensions import Self

from xdsl.dialects.builtin import BoolAttr
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    AtLeast,
    IRDLOperation,
    RangeOf,
    VarConstraint,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import ConstantLike, Pure


@irdl_attr_definition
class BoolType(ParametrizedAttribute, TypeAttribute):
    """A boolean."""

    name = "smt.bool"


NonFuncSMTType: TypeAlias = BoolType


@irdl_op_definition
class ConstantBoolOp(IRDLOperation):
    """
    This operation represents a constant boolean value. The semantics are
    equivalent to the ‘true’ and ‘false’ keywords in the Core theory of the
    SMT-LIB Standard 2.7.
    """

    name = "smt.constant"

    value_attr = prop_def(BoolAttr, prop_name="value")
    result = result_def(BoolType())

    traits = traits_def(Pure(), ConstantLike())

    assembly_format = "$value attr-dict"

    def __init__(self, value: bool):
        value_attr = BoolAttr.from_bool(value)
        super().__init__(properties={"value": value_attr}, result_types=[BoolType()])

    @property
    def value(self) -> bool:
        return bool(self.value_attr)


class VariadicBoolOp(IRDLOperation):
    """
    A variadic operation on boolean. It has a variadic number of operands, but
    requires at least two.
    """

    inputs = var_operand_def(RangeOf(base(BoolType), length=AtLeast(2)))
    result = result_def(BoolType())

    traits = traits_def(Pure())

    assembly_format = "$inputs attr-dict"

    def __init__(self, *operands: SSAValue):
        super().__init__(operands=[operands], result_types=[BoolType()])


@irdl_op_definition
class AndOp(VariadicBoolOp):
    """
    This operation performs a boolean conjunction. The semantics are equivalent
    to the ‘and’ operator in the Core theory of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    """

    name = "smt.and"


@irdl_op_definition
class OrOp(VariadicBoolOp):
    """
    This operation performs a boolean disjunction. The semantics are equivalent
    to the ‘or’ operator in the Core theory of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    """

    name = "smt.or"


@irdl_op_definition
class XOrOp(VariadicBoolOp):
    """
    This operation performs a boolean exclusive or. The semantics are equivalent
    to the ‘xor’ operator in the Core theory of the SMT-LIB Standard 2.7.

    It supports a variadic number of operands, but requires at least two.
    """

    name = "smt.xor"


def _parse_same_operand_type_variadic_to_bool_op(
    parser: Parser,
) -> tuple[Sequence[SSAValue], dict[str, Attribute]]:
    """
    Parse a variadic operation with boolean result, with format
    `%op1, %op2, ..., %opN attr-dict : T` where `T` is the type of all
    operands.
    """
    operand_pos = parser.pos
    operands = parser.parse_comma_separated_list(
        parser.Delimiter.NONE, parser.parse_unresolved_operand, "operand list"
    )
    attr_dict = parser.parse_optional_attr_dict()
    parser.parse_punctuation(":")
    operand_types = parser.parse_type()
    operands = parser.resolve_operands(
        operands, (operand_types,) * len(operands), operand_pos
    )
    return operands, attr_dict


def _print_same_operand_type_variadic_to_bool_op(
    printer: Printer, operands: Sequence[SSAValue], attr_dict: dict[str, Attribute]
):
    """
    Print a variadic operation with boolean result, with format
    `%op1, %op2, ..., %opN attr-dict : T` where `T` is the type of all
    operands.
    """
    printer.print(" ")
    printer.print_list(operands, printer.print_ssa_value)
    if attr_dict:
        printer.print_string(" ")
        printer.print_attr_dict(attr_dict)
    printer.print(" : ", operands[0].type)


class VariadicPredicateOp(IRDLOperation, ABC):
    """
    A predicate with a variadic number (but at least 2) operands.
    """

    T: ClassVar = VarConstraint("T", base(NonFuncSMTType))

    inputs = var_operand_def(RangeOf(T, length=AtLeast(2)))
    result = result_def(BoolType())

    traits = traits_def(Pure())

    @classmethod
    def parse(cls: type[Self], parser: Parser) -> Self:
        operands, attr_dict = _parse_same_operand_type_variadic_to_bool_op(parser)
        op = cls(*operands)
        op.attributes = attr_dict
        return op

    def print(self, printer: Printer):
        _print_same_operand_type_variadic_to_bool_op(
            printer, self.inputs, self.attributes
        )

    def __init__(self, *operands: SSAValue):
        super().__init__(operands=[operands], result_types=[BoolType()])


@irdl_op_definition
class DistinctOp(VariadicPredicateOp):
    """
    This operation compares the operands and returns true iff all operands are not
    identical to any of the other operands. The semantics are equivalent to the
    `distinct` operator defined in the SMT-LIB Standard 2.7 in the Core theory.

    Any SMT sort/type is allowed for the operands and it supports a variadic
    number of operands, but requires at least two. This is because the `distinct`
    operator is annotated with `:pairwise` which means that `distinct a b c d` is
    equivalent to

    ```
    and (distinct a b) (distinct a c) (distinct a d)
        (distinct b c) (distinct b d) (distinct c d)
    ```
    """

    name = "smt.distinct"


@irdl_op_definition
class EqOp(VariadicPredicateOp):
    """
    This operation compares the operands and returns true iff all operands are
    identical. The semantics are equivalent to the `=` operator defined in the
    SMT-LIB Standard 2.7 in the Core theory.

    Any SMT sort/type is allowed for the operands and it supports a variadic number of
    operands, but requires at least two. This is because the `=` operator is annotated
    with `:chainable` which means that `= a b c d` is equivalent to
    `and (= a b) (= b c) (= c d)` where and is annotated `:left-assoc`, i.e., it can
    be further rewritten to `and (and (= a b) (= b c)) (= c d)`.
    """

    name = "smt.eq"


SMT = Dialect(
    "smt",
    [
        ConstantBoolOp,
        AndOp,
        OrOp,
        XOrOp,
        DistinctOp,
        EqOp,
    ],
    [BoolType],
)
