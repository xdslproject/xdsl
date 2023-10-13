from __future__ import annotations

from typing import Annotated

from xdsl.dialects.builtin import IntegerType, Signedness
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    AnyOf,
    ConstraintVar,
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    result_def,
    var_operand_def,
)

"""
Implementation of the LTL dialect by CIRCT. Documentation: https://circt.llvm.org/docs/Dialects/LTL/
"""


@irdl_attr_definition
class Property(ParametrizedAttribute, TypeAttribute):
    """
    The `ltl.property` type represents a verifiable property built from linear
    temporal logic sequences and quantifiers, for example, *"if you see sequence
    A, eventually you will see sequence B"*.
    Note that this type explicitly identifies a *property*. However, a boolean
    value (`i1`) or a sequence (`ltl.sequence`) is also a valid property.
    Operations that accept a property as an operand will use the `AnyProperty`
    constraint, which also accepts `ltl.sequence` and `i1`.
    """

    name = "ltl.property"


@irdl_attr_definition
class Sequence(ParametrizedAttribute, TypeAttribute):
    """
    The ltl.sequence type represents a sequence of linear temporal logic, for example,
    “A is true two cycles after B is true”. Note that this type explicitly identifies
    a sequence. However, a boolean value (i1) is also a valid sequence.
    Operations that accept a sequence as an operand will use the AnySequence constraint,
    which also accepts i1.
    """

    name = "ltl.sequence"


@irdl_op_definition
class AndOp(IRDLOperation):
    """
    A conjunction of booleans, sequences, or properties. If any of the $inputs is of type
    !ltl.property, the result of the op is an !ltl.property. Otherwise it is an !ltl.sequence.
    """

    name = "ltl.and"

    T = Annotated[
        Attribute,
        AnyOf([Sequence, Property, IntegerType(1, signedness=Signedness.SIGNLESS)]),
        ConstraintVar("T"),
    ]

    input = var_operand_def(T)

    result = result_def(T)

    def __init__(
        self,
        operand: SSAValue,
    ):
        super().__init__(operands=[operand])


LTL = Dialect(
    [
        AndOp,
    ],
    [Property, Sequence],
)
