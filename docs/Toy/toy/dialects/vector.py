from __future__ import annotations

from typing import TypeAlias

from math import prod

from xdsl.ir import Operation, SSAValue, Dialect, OpResult
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_op_definition,
    Operand,
    operand_def,
    result_def,
)
from xdsl.dialects.builtin import (
    StringAttr,
    TensorType,
    VectorType,
    Float64Type,
    DenseIntOrFPElementsAttr,
    f64,
)
from xdsl.traits import Pure

from .toy import UnrankedTensorTypeF64, AnyTensorTypeF64

VectorTypeF64: TypeAlias = VectorType[Float64Type]


@irdl_op_definition
class VectorAddOp(IRDLOperation):
    name = "vector.add"

    res: OpResult = result_def(VectorTypeF64)
    lhs: Operand = operand_def(VectorTypeF64)
    rhs: Operand = operand_def(VectorTypeF64)

    traits = frozenset([Pure()])

    def __init__(self, lhs: Operation | SSAValue, rhs: Operation | SSAValue):
        if isinstance(lhs, Operation):
            lhs = lhs.results[0]
        super().__init__(operands=[lhs, rhs], result_types=[lhs.typ])


@irdl_op_definition
class VectorMulOp(IRDLOperation):
    name = "vector.mul"

    res: OpResult = result_def(VectorTypeF64)
    lhs: Operand = operand_def(VectorTypeF64)
    rhs: Operand = operand_def(VectorTypeF64)

    traits = frozenset([Pure()])

    def __init__(self, lhs: Operation | SSAValue, rhs: Operation | SSAValue):
        if isinstance(lhs, Operation):
            lhs = lhs.results[0]
        super().__init__(operands=[lhs, rhs], result_types=[lhs.typ])


@irdl_op_definition
class VectorConstantOp(IRDLOperation):
    """
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = riscv.vector_constant array<[1, 2, 3, 4, 5, 6]>: array<i32>
    ```
    """

    name = "vector.constant"
    data: DenseIntOrFPElementsAttr = attr_def(DenseIntOrFPElementsAttr)
    label: StringAttr = attr_def(StringAttr)
    res: OpResult = result_def(VectorTypeF64)

    traits = frozenset([Pure()])

    def __init__(self, data: DenseIntOrFPElementsAttr, label: str | StringAttr):
        if isinstance(label, str):
            label = StringAttr(label)
        result_type = data.type
        super().__init__(
            result_types=[result_type], attributes={"data": data, "label": label}
        )

    def get_data(self) -> list[int]:
        return [int(el.value.data) for el in self.data.data.data]


# Tensor <-> Vector conversion


@irdl_op_definition
class TensorMakeOp(IRDLOperation):
    name = "vector.tensor.make"

    tensor: OpResult = result_def(UnrankedTensorTypeF64)
    shape: Operand = operand_def(VectorTypeF64)
    data: Operand = operand_def(VectorTypeF64)

    traits = frozenset([Pure()])

    def __init__(
        self,
        shape: Operation | SSAValue,
        data: Operation | SSAValue,
        result_type: AnyTensorTypeF64,
    ):
        super().__init__(operands=[shape, data], result_types=[result_type])


@irdl_op_definition
class TensorDataOp(IRDLOperation):
    name = "vector.tensor.data"

    tensor: Operand = operand_def(AnyTensorTypeF64)
    data: OpResult = result_def(VectorTypeF64)

    traits = frozenset([Pure()])

    def __init__(self, tensor: Operation | SSAValue):
        if isinstance(tensor, Operation):
            tensor = tensor.results[0]
        if isinstance(tensor.typ, TensorType):
            data_len = prod(tensor.typ.get_shape())
        else:
            # Use -1 for unknown length
            data_len = -1
        result_type = VectorTypeF64.from_element_type_and_shape(f64, [data_len])
        super().__init__(operands=[tensor], result_types=[result_type])


@irdl_op_definition
class TensorShapeOp(IRDLOperation):
    name = "vector.tensor.shape"

    tensor: Operand = operand_def(VectorTypeF64)
    data: OpResult = result_def(AnyTensorTypeF64)

    traits = frozenset([Pure()])

    def __init__(self, tensor: Operation | SSAValue):
        if isinstance(tensor, Operation):
            tensor = tensor.results[0]

        # If we know the tensor shape at compile time, we can set the
        # vector length in the type

        if isinstance(tensor.typ, TensorType):
            shape = tensor.typ.get_shape()
        else:
            # Use -1 for unknown length
            shape = [-1]

        result_type = VectorTypeF64.from_element_type_and_shape(f64, shape)

        super().__init__(operands=[tensor], result_types=[result_type])


Vector = Dialect(
    [
        VectorAddOp,
        VectorMulOp,
        TensorMakeOp,
        TensorShapeOp,
        TensorDataOp,
        VectorConstantOp,
    ],
    [],
)
