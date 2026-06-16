"""
mod_arith is a dialect implementing modular arithmetic, originally
implemented as part of the HEIR project (https://github.com/google/heir/tree/main).
"""

from abc import ABC
from typing import ClassVar

from xdsl.dialects.arith import signlessIntegerLike
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)
from xdsl.traits import Pure


class BinaryOp(IRDLOperation, ABC):
    """
    Simple binary operation
    """

    T: ClassVar = VarConstraint("T", signlessIntegerLike)

    lhs = operand_def(T)
    rhs = operand_def(T)
    output = result_def(T)
    modulus = prop_def(IntegerAttr)

    irdl_options = (ParsePropInAttrDict(),)

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($output)"
    traits = traits_def(Pure())

    def __init__(
        self,
        lhs: SSAValue | Operation,
        rhs: SSAValue | Operation,
        result_type: Attribute | None,
        modulus: Attribute,
    ):
        if result_type is None:
            result_type = SSAValue.get(lhs).type

        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type],
            properties={"modulus": modulus},
        )


@irdl_op_definition
class AddOp(BinaryOp):
    name = "mod_arith.add"


ModArith = Dialect(
    "mod_arith",
    [
        AddOp,
    ],
)
