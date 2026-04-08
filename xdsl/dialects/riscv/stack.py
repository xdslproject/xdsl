from __future__ import annotations

from typing import ClassVar, Generic

from typing_extensions import TypeVar

from xdsl.dialects.builtin import Attribute, ContainerType
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    IRDLAttrConstraint,
    IRDLOperation,
    ParamAttrConstraint,
    VarConstraint,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)

_StackValueType = TypeVar(
    "_StackValueType", bound=Attribute, covariant=True, default=Attribute
)


@irdl_attr_definition
class StackSlotType(
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[_StackValueType],
    Generic[_StackValueType],
):
    name = "riscv_stack.ptr"
    value_type: _StackValueType

    def __init__(self, value_type: _StackValueType):
        super().__init__(value_type)

    def get_element_type(self) -> _StackValueType:
        return self.value_type

    @staticmethod
    def constr(
        element_type: IRDLAttrConstraint[_StackValueType] = AnyAttr(),
    ) -> AttrConstraint[StackSlotType[_StackValueType]]:
        return ParamAttrConstraint[StackSlotType[_StackValueType]](
            StackSlotType, (element_type,)
        )


@irdl_op_definition
class AllocaOp(IRDLOperation):
    """Allocates space on the stack."""

    name = "riscv_stack.alloca"

    ref = result_def(StackSlotType)

    def __init__(self, value_type: Attribute):
        super().__init__(
            result_types=(StackSlotType(value_type),),
        )


@irdl_op_definition
class StoreOp(IRDLOperation):
    """Stores a value into a stack pointer."""

    name = "riscv_stack.store"

    _T: ClassVar = VarConstraint("_T", AnyAttr())
    ptr = operand_def(StackSlotType.constr(_T))
    rs = operand_def(_T)

    def __init__(
        self,
        ptr: SSAValue[StackSlotType[_StackValueType]] | AllocaOp,
        value: SSAValue[_StackValueType] | Operation,
    ):
        ptr_val = ptr.ref if isinstance(ptr, AllocaOp) else ptr
        super().__init__(
            operands=(ptr_val, value),
        )


@irdl_op_definition
class LoadOp(IRDLOperation):
    """Loads a value from a stack-allocated slot."""

    name = "riscv_stack.load"

    _T: ClassVar = VarConstraint("_T", AnyAttr())
    ptr = operand_def(StackSlotType.constr(_T))
    rd = result_def(_T)

    def __init__(self, ptr: SSAValue[StackSlotType] | AllocaOp):
        ptr_val = ptr.ref if isinstance(ptr, AllocaOp) else ptr
        super().__init__(operands=(ptr,), result_types=[ptr_val.type.value_type])


# Define the Dialect
RISCVStack = Dialect(
    "riscv_stack",
    [
        AllocaOp,
        StoreOp,
        LoadOp,
    ],
    [
        StackSlotType,
    ],
)
