from abc import ABC

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)

from .register import IntRegisterType


class ARMOperation(IRDLOperation, ABC):
    """
    Base class for operations that can be a part of ARM assembly printing.
    """

    comment = opt_attr_def(StringAttr)
    """
    An optional comment that will be printed along with the instruction.
    """


@irdl_op_definition
class DSMovOp(ARMOperation):
    """
    Copies the value of s into d.

    https://developer.arm.com/documentation/dui0473/m/arm-and-thumb-instructions/mov
    """

    name = "arm.ds.mov"

    d = result_def(IntRegisterType)
    s = operand_def(IntRegisterType)
    assembly_format = "$s attr-dict `:` `(` type($s) `)` `->` type($d)"

    def __init__(
        self,
        d: IntRegisterType,
        s: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=(s,),
            attributes={
                "comment": comment,
            },
            result_types=(d,),
        )


@irdl_op_definition
class GetRegisterOp(ARMOperation):
    """
    This instruction allows us to create an SSAValue for a given register name.
    """

    name = "arm.get_register"

    result = result_def(IntRegisterType)
    assembly_format = "attr-dict `:` type($result)"

    def __init__(self, register_type: IntRegisterType):
        super().__init__(result_types=[register_type])
