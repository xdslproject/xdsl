from abc import ABC

from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def

from .register import IntRegisterType


class ARMOperation(IRDLOperation, ABC):
    """
    Base class for operations that can be a part of ARM assembly printing.
    """

    ...


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
