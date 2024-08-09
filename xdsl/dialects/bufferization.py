from dataclasses import dataclass
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
    ConstraintContext,
    GenericAttrConstraint,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    result_def,
    var_operand_def,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class TensorMemrefInferenceConstraint(
    GenericAttrConstraint[TensorType[Attribute] | UnrankedTensorType[Attribute]]
):
    """
    Constrain a ranked or unranked tensor type to be of same element type and shape as a
    named ranked or unranked memref type.
    """

    memref_type_name: str

    def can_infer(self, constraint_names: set[str]) -> bool:
        return self.memref_type_name in constraint_names

    def infer(self, constraint_context: ConstraintContext) -> Attribute:
        assert self.memref_type_name in constraint_context.variables
        m_type = constraint_context.variables[self.memref_type_name]
        if isa(m_type, MemRefType[Attribute]):
            return TensorType(m_type.get_element_type(), m_type.get_shape())
        elif isa(m_type, UnrankedMemrefType[Attribute]):
            return UnrankedTensorType(m_type.get_element_type())
        else:
            raise ValueError(
                f"Unexpected {self.memref_type_name} - cannot infer attribute"
            )

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        assert self.memref_type_name in constraint_context.variables
        memref = constraint_context.variables[self.memref_type_name]
        if isinstance(memref, MemRefType):
            if not isinstance(attr, TensorType):
                raise VerifyException(
                    "Expected ranked tensor to match the ranked memref"
                )
            if not memref.get_shape() == attr.get_shape():
                raise VerifyException("Expected tensor shape to match memref shape")
            if not memref.get_element_type() == attr.get_element_type():
                raise VerifyException(
                    "Expected tensor element type to match memref element type"
                )
        elif isinstance(memref, UnrankedMemrefType):
            if not isinstance(attr, UnrankedTensorType):
                raise VerifyException(
                    "Expected unranked tensor to match the unranked memref"
                )
            if not memref.get_element_type() == attr.get_element_type():
                raise VerifyException(
                    "Expected tensor element type to match memref element type"
                )
        else:
            raise VerifyException(
                f"Expected {self.memref_type_name} to be ranked or unranked memref type"
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

    memref = operand_def(VarConstraint("M", AnyOf([MemRefType, UnrankedMemrefType])))
    tensor = result_def(TensorMemrefInferenceConstraint("M"))
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

    tensor = operand_def(TensorMemrefInferenceConstraint("M"))
    memref = result_def(VarConstraint("M", AnyOf([MemRefType, UnrankedMemrefType])))
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
