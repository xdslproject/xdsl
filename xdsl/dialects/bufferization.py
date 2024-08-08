from typing import Any, Literal

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
    AttrConstraint,
    AttrSizedOperandSegments,
    ConstraintContext,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    result_def,
    var_operand_def,
)
from xdsl.utils.hints import isa


class TensorMemrefInferenceConstraint(VarConstraint[Attribute]):
    """
    Constraint to infer tensor shapes from memref shapes. Use `T` and `M` variable names for tensors and memrefs.
    """

    def __init__(self, name: Literal["T", "M"], constraint: AttrConstraint):
        super().__init__(name, constraint)

    def can_infer(self, constraint_names: set[str]) -> bool:
        return (
            self.name == "T"
            and "M" in constraint_names
            or self.name in constraint_names
        )

    def infer(self, constraint_context: ConstraintContext) -> Attribute:
        if self.name in constraint_context.variables:
            return constraint_context.variables[self.name]
        if self.name == "T" and "M" in constraint_context.variables:
            m_type = constraint_context.variables["M"]
            if isa(m_type, MemRefType[Attribute]):
                return TensorType(m_type.get_element_type(), m_type.get_shape())
            if isa(m_type, UnrankedMemrefType[Attribute]):
                return UnrankedTensorType(m_type.element_type)
        raise ValueError(f"Unexpected {self.name} - cannot infer attribute")


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

    memref = operand_def(
        TensorMemrefInferenceConstraint("M", AnyOf([MemRefType, UnrankedMemrefType]))
    )
    tensor = result_def(
        TensorMemrefInferenceConstraint("T", AnyOf([TensorType, UnrankedTensorType]))
    )
    writable = opt_prop_def(UnitAttr)
    restrict = opt_prop_def(UnitAttr)

    assembly_format = "$memref (`restrict` $restrict^)? (`writable` $writable^)? attr-dict `:` type($memref)"

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


@irdl_op_definition
class ToMemrefOp(IRDLOperation):
    name = "bufferization.to_memref"

    tensor = operand_def(
        TensorMemrefInferenceConstraint("T", AnyOf([TensorType, UnrankedTensorType]))
    )
    memref = result_def(
        TensorMemrefInferenceConstraint("M", AnyOf([MemRefType, UnrankedMemrefType]))
    )
    read_only = opt_prop_def(UnitAttr)

    assembly_format = "$tensor (`read_only` $read_only^)?  `:` attr-dict type($memref)"


Bufferization = Dialect(
    "bufferization",
    [
        AllocTensorOp,
        ToTensorOp,
        ToMemrefOp,
    ],
    [],
)
