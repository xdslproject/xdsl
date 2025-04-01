"""
Implementation of the LTL dialect by CIRCT.

See external [documentation](https://circt.llvm.org/docs/Dialects/LTL/).
"""

from __future__ import annotations

from typing import ClassVar

from xdsl.dialects.builtin import IntegerType
from xdsl.ir import Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    AnyOf,
    IRDLOperation,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    result_def,
    var_operand_def,
)


@irdl_attr_definition
class Property(ParametrizedAttribute, TypeAttribute):
    """
    The `ltl.property` type represents a verifiable property built from linear
    temporal logic sequences and quantifiers, for example,
    "if you see sequence A, eventually you will see sequence B".
    """

    name = "ltl.property"


@irdl_attr_definition
class Sequence(ParametrizedAttribute, TypeAttribute):
    """
    The ltl.sequence type represents a sequence of linear temporal logic, for example,
    “A is true two cycles after B is true”.
    """

    name = "ltl.sequence"


@irdl_op_definition
class AndOp(IRDLOperation):
    """
    A conjunction of booleans, sequences, or properties. If any of the `inputs` is of type
    !ltl.property or !ltl.sequence or 1-bit signless int, the result of the op is consistent with the input type.
    """

    name = "ltl.and"

    T: ClassVar = VarConstraint("T", AnyOf([Sequence, Property, IntegerType(1)]))

    input = var_operand_def(T)

    result = result_def(T)

    def __init__(
        self,
        operand: SSAValue,
    ):
        super().__init__(operands=[operand])


LTL = Dialect(
    "ltl",
    [
        AndOp,
    ],
    [
        Property,
        Sequence,
    ],
)
