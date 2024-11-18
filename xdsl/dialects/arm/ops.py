from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from xdsl.irdl import IRDLOperation, irdl_op_definition, result_def

from .register import ARMRegisterType

R1InvT = TypeVar("R1InvT", bound=ARMRegisterType)


class ARMOp(IRDLOperation, ABC):
    """
    Base class for operations that can be a part of ARM assembly printing.
    """

    @abstractmethod
    def assembly_line(self) -> str | None:
        raise NotImplementedError()


class GetAnyRegisterOperation(Generic[R1InvT], ARMOp, ABC):
    """
    This instruction allows us to create an SSAValue for a given register name.
    """

    result = result_def(R1InvT)

    def __init__(self, register_type: R1InvT):
        super().__init__(result_types=[register_type])

    def assembly_line(self) -> str | None:
        return None


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[ARMRegisterType]):
    name = "arm.get_register"
    assembly_format = "attr-dict `:` type($result)"
