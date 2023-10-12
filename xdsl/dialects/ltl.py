from __future__ import annotations
from xdsl.dialects.builtin import ContainerOf
from xdsl.ir import Attribute, SSAValue

from xdsl.ir.core import Attribute, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
)

from typing import Annotated
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    AnyOf,
    ConstraintVar

)

from xdsl.dialects.arith import signlessIntegerLike
from xdsl.irdl.irdl import result_def



"""
Implementation of the LTL dialect by CIRCT. Documentation: https://circt.llvm.org/docs/Dialects/LTL/
"""


@irdl_attr_definition
class property(ParametrizedAttribute, TypeAttribute):
    """
    The `ltl.property` type represents a verifiable property built from linear
    temporal logic sequences and quantifiers, for example, *"if you see sequence
    A, eventually you will see sequence B"*.
    Note that this type explicitly identifies a *property*. However, a boolean
    value (`i1`) or a sequence (`ltl.sequence`) is also a valid property.
    Operations that accept a property as an operand will use the `AnyProperty`
    constraint, which also accepts `ltl.sequence` and `i1`.
    """

    name = "ltl.propertytype"


@irdl_attr_definition
class sequence(ParametrizedAttribute, TypeAttribute):
    """
    The ltl.sequence type represents a sequence of linear temporal logic, for example,
    “A is true two cycles after B is true”. Note that this type explicitly identifies
    a sequence. However, a boolean value (i1) is also a valid sequence.
    Operations that accept a sequence as an operand will use the AnySequence constraint,
    which also accepts i1.
    """

    name = "ltl.sequencetype"


ltltype = ContainerOf(AnyOf([sequence, property, signlessIntegerLike]))

@irdl_op_definition
class AndOp(IRDLOperation):
    """
    A conjunction of booleans, sequences, or properties. If any of the $inputs is of type
    !ltl.property, the result of the op is an !ltl.property. Otherwise it is an !ltl.sequence.
    """

    name = "ltl.and"

    T = Annotated[Attribute, ConstraintVar("T"), sequence | property | signlessIntegerLike]

    input = operand_def(T)

    result = result_def(T)

    def __init__(
            self, 
            operand: SSAValue, 
    ):
        super().__init__(operands = [operand])

    def verify_(self):
        if isinstance(self.input, property) ^ isinstance(self.result, property):
            raise ValueError("AndOp: property type mismatch")
        if isinstance(self.input, sequence) ^ isinstance(self.result, sequence):
            raise ValueError("AndOp: sequence type mismatch")
