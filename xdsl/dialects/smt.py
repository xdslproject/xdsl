from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from xdsl.dialects.builtin import BoolAttr, IntegerAttr
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    AtLeast,
    IRDLOperation,
    RangeOf,
    base,
    irdl_attr_definition,
    irdl_op_definition,
    prop_def,
    result_def,
    traits_def,
    var_operand_def,
)
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
        value_attr = IntegerAttr(-1 if value else 0, 1)
        super().__init__(properties={"value": value_attr}, result_types=[BoolType()])

    @property
    def value(self) -> bool:
        return self.value_attr.value.data != 0


class VariadicBoolOp(IRDLOperation):
    """
    A variadic operation on boolean. It has a variadic number of operands, but
    requires at least two.
    """

    inputs = var_operand_def(RangeOf(base(BoolType), length=AtLeast(2)))
    result = result_def(BoolType())

    traits = traits_def(Pure())

    assembly_format = "$inputs attr-dict"

    def __init__(self, operands: Sequence[SSAValue]):
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


SMT = Dialect("smt", [ConstantBoolOp, AndOp, OrOp, XOrOp], [BoolType])
