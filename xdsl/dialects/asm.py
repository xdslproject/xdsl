"""
The `asm` dialect provides utilities for embedding assembly-level abstractions in higher-level functions.
"""

from collections.abc import Sequence
from typing import ClassVar

from xdsl import ir, irdl
from xdsl.backend.register_type import RegisterAllocatedMemoryEffect, RegisterType
from xdsl.interfaces import HasFolderInterface
from xdsl.traits import HasParent, IsolatedFromAbove, IsTerminator, Pure
from xdsl.utils.exceptions import VerifyException


@irdl.irdl_op_definition
class FromRegOp(irdl.IRDLOperation, HasFolderInterface):
    """
    Get the value stored in a given register.
    """

    name = "asm.from_reg"

    register = irdl.operand_def(RegisterType)
    value = irdl.result_def()

    traits = irdl.traits_def(Pure())

    assembly_format = "$register attr-dict `:` type($register) `->` type($value)"

    def __init__(self, register: ir.SSAValue | ir.Operation, result_type: ir.Attribute):
        super().__init__(operands=(register,), result_types=(result_type,))

    @classmethod
    def get(cls, register: ir.SSAValue | ir.Operation, result_type: ir.Attribute):
        """
        Create the operation, and set the result's name hint to `register`'s name hint.
        """
        register = ir.SSAValue.get(register)
        op = cls(register, result_type)
        op.value.name_hint = register.name_hint
        return op

    def verify_(self) -> None:
        if (
            isinstance(to_reg_op := self.register.owner, ToRegOp)
            and to_reg_op.value.type != self.value.type
        ):
            raise VerifyException(
                f"Expected original value type {to_reg_op.value.type} to be "
                f"equal to own value type {self.value.type}."
            )

    def fold(self) -> Sequence[ir.SSAValue | ir.Attribute] | None:
        if isinstance(to_reg_op := self.register.owner, ToRegOp):
            return (to_reg_op.value,)


@irdl.irdl_op_definition
class ToRegOp(irdl.IRDLOperation, HasFolderInterface):
    """
    Get the register holding a given value.
    """

    name = "asm.to_reg"

    value = irdl.operand_def()
    register = irdl.result_def(RegisterType)

    traits = irdl.traits_def(Pure())

    assembly_format = "$value attr-dict `:` type($value) `->` type($register)"

    def __init__(self, value: ir.SSAValue | ir.Operation, result_type: RegisterType):
        super().__init__(operands=(value,), result_types=(result_type,))

    @classmethod
    def get(cls, value: ir.SSAValue | ir.Operation, result_type: RegisterType):
        """
        Create the operation, and set the result's name hint to `value`'s name hint.
        """
        value = ir.SSAValue.get(value)
        op = cls(value, result_type)
        op.register.name_hint = value.name_hint
        return op

    def verify_(self) -> None:
        if (
            isinstance(from_reg_op := self.value.owner, FromRegOp)
            and from_reg_op.register.type != self.register.type
        ):
            raise VerifyException(
                f"Expected original register type {from_reg_op.register.type} to be "
                f"equal to own register type {self.register.type}."
            )

    def fold(self) -> Sequence[ir.SSAValue | ir.Attribute] | None:
        if isinstance(from_reg_op := self.value.owner, FromRegOp):
            return (from_reg_op.register,)


@irdl.irdl_op_definition
class RegionOp(irdl.IRDLOperation):
    """
    Contains target-specific code.
    Operands take values, stored in the body's entry block arguments' registers.
    Yields resulting registers, and returns the values they contain.
    """

    name = "asm.region"

    OPERAND_COUNT: ClassVar = irdl.IntVarConstraint("OPERAND_COUNT", irdl.AnyInt())

    operands_ = irdl.var_operand_def(
        irdl.RangeOf(irdl.AnyAttr()).of_length(OPERAND_COUNT)
    )
    results_ = irdl.var_result_def()
    body = irdl.region_def(
        entry_args=irdl.RangeOf(RegisterType).of_length(OPERAND_COUNT)
    )

    traits = irdl.traits_def(IsolatedFromAbove())

    assembly_format = "( `(` operands^ `)` )? attr-dict-with-keyword $body `:` functional-type(operands, results)"

    def __init__(
        self,
        operands: Sequence[ir.SSAValue],
        body: ir.Region,
        result_types: Sequence[ir.Attribute],
    ):
        super().__init__(
            operands=(operands,),
            result_types=(result_types,),
            regions=(body,),
        )


@irdl.irdl_op_definition
class YieldOp(irdl.IRDLOperation):
    """
    Terminates an `asm.region` operation.
    The yielded registers contain the return values of the parent `asm.region`
    operation.
    """

    name = "asm.yield"

    arguments = irdl.var_operand_def(RegisterType)

    assembly_format = "attr-dict ($arguments^ `:` type($arguments))?"

    traits = irdl.traits_def(
        IsTerminator(),
        RegisterAllocatedMemoryEffect(),
        HasParent(RegionOp),
    )

    def __init__(self, *arguments: ir.SSAValue[RegisterType]):
        super().__init__(operands=(arguments,))


ASM = ir.Dialect(
    "asm",
    [
        FromRegOp,
        RegionOp,
        ToRegOp,
        YieldOp,
    ],
    [],
    [],
)
