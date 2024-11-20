from abc import ABC, abstractmethod

from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Operation, SSAValue
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)

from .assembly import AssemblyInstructionArg, assembly_arg_str, assembly_line
from .register import IntRegisterType


class ARMOperation(IRDLOperation, ABC):
    """
    Base class for operations that can be a part of ARM assembly printing.
    """

    @abstractmethod
    def assembly_line(self) -> str | None:
        raise NotImplementedError()


class ARMInstruction(ARMOperation, ABC):
    """
    Base class for operations that can be a part of x86 assembly printing. Must
    represent an instruction in the x86 instruction set.
    The name of the operation will be used as the x86 assembly instruction name.
    """

    comment = opt_attr_def(StringAttr)
    """
    An optional comment that will be printed along with the instruction.
    """

    @abstractmethod
    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        """
        The arguments to the instruction, in the order they should be printed in the
        assembly.
        """
        raise NotImplementedError()

    def assembly_instruction_name(self) -> str:
        """
        By default, the name of the instruction is the same as the name of the operation.
        """

        return self.name.split(".")[-1]

    def assembly_line(self) -> str | None:
        # default assembly code generator
        instruction_name = self.assembly_instruction_name()
        arg_str = ", ".join(
            assembly_arg_str(arg)
            for arg in self.assembly_line_args()
            if arg is not None
        )
        return assembly_line(instruction_name, arg_str, self.comment)


@irdl_op_definition
class DSMovOp(ARMInstruction):
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
        s: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        d: IntRegisterType,
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

    def assembly_line_args(self):
        return (self.d, self.s)


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

    def assembly_line(self):
        return None
