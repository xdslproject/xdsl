from typing import Any

from xdsl.dialects.builtin import (
    ContainerType,
    IndexType,
    MemRefType,
    ShapedType,
    TensorType,
    UnitAttr,
    UnrankedMemrefType,
    UnrankedTensorType,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyOf,
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    result_def,
    var_operand_def,
)


@irdl_op_definition
class AllocTensorOp(IRDLOperation):
    name = "bufferization.alloc_tensor"

    dynamic_sizes = var_operand_def(IndexType())
    copy = opt_operand_def(AnyOf((TensorType, UnrankedTensorType)))
    size_hint = opt_operand_def(IndexType())

    tensor = result_def(AnyOf((TensorType, UnrankedTensorType)))

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    def __init__(
        self,
        result_type: Attribute,
        dynamic_sizes: list[Operation | SSAValue] | None = None,
        copy: SSAValue | Operation | None = None,
        size_hint: SSAValue | Operation | None = None,
    ):
        super().__init__(
            operands=(dynamic_sizes, copy, size_hint),
            result_types=(result_type,),
        )


@irdl_op_definition
class ToTensorOp(IRDLOperation):
    name = "bufferization.to_tensor"

    memref = operand_def(AnyOf((MemRefType, UnrankedMemrefType)))
    tensor = result_def(AnyOf((TensorType, UnrankedTensorType)))
    writable = opt_prop_def(UnitAttr)
    restrict = opt_prop_def(UnitAttr)

    def __init__(
        self,
        memref: SSAValue | Operation,
        restrict: bool = False,
        writable: bool = False,
    ):
        memref_v = SSAValue.get(memref)
        memref_t = memref_v.type
        if not isinstance(memref_t, ContainerType):
            raise ValueError(f"Expected ContainerType, got {memref_t}")
        if not isinstance(memref_t, ShapedType):
            raise ValueError(f"Expected ShapedType, got {memref_t}")
        properties = dict[str, Attribute]()
        if restrict:
            properties["restrict"] = UnitAttr()
        if writable:
            properties["writable"] = UnitAttr()
        super().__init__(
            operands=(memref,),
            result_types=(
                TensorType[Any](memref_t.get_element_type(), memref_t.get_shape()),
            ),
            properties=properties,
        )


Bufferization = Dialect(
    "bufferization",
    [
        AllocTensorOp,
        ToTensorOp,
    ],
    [],
)
