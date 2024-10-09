"""
mod_arith is a dialect implementing modular arithmetic, originally
implemented as part of the HEIR project (https://github.com/google/heir/tree/main).
"""

from abc import ABC
from typing import Annotated, Generic, TypeVar

from xdsl.dialects.arith import signlessIntegerLike
from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
)
from xdsl.traits import Pure


class ModArithOp(IRDLOperation, ABC):
    pass


class BinaryOp(ModArithOp, ABC):
    """
    Simple binary operation
    """

    T = Annotated[Attribute, ConstraintVar("T"), signlessIntegerLike]
    modulus = prop_def(AnyIntegerAttr)
    lhs = operand_def(T)
    rhs = operand_def(T)
    output = result_def(T)

    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$lhs `,` $rhs attr-dict `:` type($output)"
    traits = frozenset((Pure(),))

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
