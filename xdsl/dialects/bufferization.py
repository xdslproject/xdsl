from typing import Any

from xdsl.dialects.builtin import (
    AnyMemRefTypeConstr,
    AnyTensorTypeConstr,
    AnyUnrankedMemrefTypeConstr,
    AnyUnrankedTensorTypeConstr,
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


class TensorMemrefInferenceConstraint(VarConstraint[Attribute]):
    """
    Constraint to infer tensor shapes from memref shapes, inferring ranked tensor from ranked memref
    (and unranked from unranked, respectively).

    Verification checks that attributes of the same variable name are either all ranked or all unranked,
    and checks for matching element type, shape (ranked only), as well as verifying sub constraints.
    """

    def infer(self, constraint_context: ConstraintContext) -> Attribute:
        if self.name in constraint_context.variables:
            m_type = constraint_context.get_variable(self.name)
            if isa(m_type, MemRefType[Attribute]):
                return TensorType(m_type.get_element_type(), m_type.get_shape())
            if isa(m_type, UnrankedMemrefType[Attribute]):
                return UnrankedTensorType(m_type.element_type)
        raise ValueError(f"Unexpected {self.name} - cannot infer attribute")

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if self.name in constraint_context.variables:
            seen = constraint_context.get_variable(self.name)
            if not (
                isinstance(attr, ContainerType)
                and isinstance(seen, ContainerType)
                and attr.get_element_type() == seen.get_element_type()
            ):
                raise VerifyException(
                    f"Unexpected {self.name} - cannot verify element type of attribute {attr}"
                )
            if (
                isinstance(attr, ShapedType) != isinstance(seen, ShapedType)
                or isinstance(attr, ShapedType)
                and isinstance(seen, ShapedType)
                and attr.get_shape() != seen.get_shape()
            ):
                raise VerifyException(
                    f"Unexpected {self.name} - cannot verify shape of attribute {attr}"
                )
        elif isinstance(attr, ContainerType):
            self.constraint.verify(attr, constraint_context)
            constraint_context.set_variable(self.name, attr)
        else:
            raise VerifyException(
                f"Unexpected {self.name} - attribute must be ContainerType"
            )


@irdl_op_definition
class AllocTensorOp(IRDLOperation):
    name = "bufferization.alloc_tensor"

    dynamic_sizes = var_operand_def(IndexType())
    copy = opt_operand_def(AnyOf((AnyTensorTypeConstr, AnyUnrankedTensorTypeConstr)))
    size_hint = opt_operand_def(IndexType())

    tensor = result_def(AnyOf((AnyTensorTypeConstr, AnyUnrankedTensorTypeConstr)))

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
        TensorMemrefInferenceConstraint(
            "T", AnyOf([AnyMemRefTypeConstr, AnyUnrankedMemrefTypeConstr])
        )
    )
    tensor = result_def(
        TensorMemrefInferenceConstraint(
            "T", AnyOf([AnyTensorTypeConstr, AnyUnrankedTensorTypeConstr])
        )
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
        TensorMemrefInferenceConstraint(
            "T", AnyOf([AnyTensorTypeConstr, AnyUnrankedTensorTypeConstr])
        )
    )
    memref = result_def(
        TensorMemrefInferenceConstraint(
            "T", AnyOf([AnyMemRefTypeConstr, AnyUnrankedMemrefTypeConstr])
        )
    )
    read_only = opt_prop_def(UnitAttr)

    assembly_format = "$tensor (`read_only` $read_only^)?  `:` attr-dict type($memref)"


@irdl_op_definition
class MaterializeInDestination(IRDLOperation):
    name = "bufferization.materialize_in_destination"

    source = operand_def(
        TensorMemrefInferenceConstraint(
            "T", AnyTensorTypeConstr | AnyUnrankedTensorTypeConstr
        )
    )
    dest = operand_def(
        TensorMemrefInferenceConstraint(
            "T", AnyTensorTypeConstr | AnyUnrankedTensorTypeConstr
        )
    )
    result = result_def(
        TensorMemrefInferenceConstraint(
            "T", AnyTensorTypeConstr | AnyUnrankedTensorTypeConstr
        )
    )
    restrict = opt_prop_def(UnitAttr)
    writable = opt_prop_def(UnitAttr)

    assembly_format = "$source `in` (`restrict` $restrict^)? (`writable` $writable^)? $dest attr-dict `:` `(` type($source) `,` type($dest) `)` `->` type($result)"


Bufferization = Dialect(
    "bufferization",
    [
        AllocTensorOp,
        ToTensorOp,
        ToMemrefOp,
        MaterializeInDestination,
    ],
    [],
)
