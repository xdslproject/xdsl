from xdsl.dialects.builtin import (
    AnyFloat,
    ContainerType,
    FixedBitwidthType,
    IntegerType,
)
from xdsl.dialects.riscv.abstract_ops import RISCVCustomFormatOperation
from xdsl.dialects.riscv.registers import Registers, RISCVRegisterType
from xdsl.ir import Dialect, Operation, ParametrizedAttribute, SSAValue, TypeAttribute
from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    result_def,
)


@irdl_attr_definition
class StackSlotType(
    ParametrizedAttribute,
    TypeAttribute,
    ContainerType[FixedBitwidthType],
):
    name = "riscv_stack.ptr"
    value_type: FixedBitwidthType

    def __init__(self, value_type: FixedBitwidthType):
        super().__init__(value_type)

    def get_element_type(self) -> FixedBitwidthType:
        return self.value_type


@irdl_op_definition
class AllocaOp(RISCVCustomFormatOperation):
    """Allocates space on the stack."""

    name = "riscv_stack.alloca"

    ref = result_def(StackSlotType)

    def __init__(self, value_type: FixedBitwidthType):
        super().__init__(
            result_types=(StackSlotType(value_type),),
        )


@irdl_op_definition
class StoreOp(RISCVCustomFormatOperation):
    """Stores a value into a stack pointer."""

    name = "riscv_stack.store"

    ptr = operand_def(StackSlotType)
    rs = operand_def(RISCVRegisterType)

    def __init__(
        self,
        ptr: SSAValue[StackSlotType] | AllocaOp,
        value: SSAValue | Operation,
    ):
        ptr_val = ptr.ref if isinstance(ptr, AllocaOp) else ptr
        super().__init__(
            operands=(ptr_val, value),
        )


@irdl_op_definition
class LoadOp(RISCVCustomFormatOperation):
    """Loads a value from a stack-allocated slot."""

    name = "riscv_stack.load"

    ptr = operand_def(StackSlotType)
    rd = result_def(RISCVRegisterType)

    def __init__(
        self,
        ptr: SSAValue[StackSlotType] | AllocaOp,
        rd: RISCVRegisterType | None = None,
    ):
        ptr_val = ptr.ref if isinstance(ptr, AllocaOp) else ptr
        if rd is None:
            if isinstance(ptr_val.type.value_type, AnyFloat):
                rd = Registers.UNALLOCATED_FLOAT
            elif isinstance(ptr_val.type.value_type, IntegerType):
                rd = Registers.UNALLOCATED_INT
        super().__init__(operands=(ptr,), result_types=[rd])


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
