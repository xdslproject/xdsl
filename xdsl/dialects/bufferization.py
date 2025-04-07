from collections.abc import Sequence, Set
from dataclasses import dataclass
from typing import Any, ClassVar

from xdsl.dialects.builtin import (
    AnyTensorTypeConstr,
    AnyUnrankedMemRefTypeConstr,
    AnyUnrankedTensorTypeConstr,
    ContainerType,
    IndexType,
    MemRefType,
    ShapedType,
    TensorType,
    UnitAttr,
    UnrankedMemRefType,
    UnrankedTensorType,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    ConstraintContext,
    GenericAttrConstraint,
    IRDLOperation,
    VarConstraint,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    result_def,
    var_operand_def,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@dataclass(frozen=True)
class TensorFromMemRefConstraint(
    GenericAttrConstraint[TensorType[Attribute] | UnrankedTensorType[Attribute]]
):
    """
    Converts an input memref constraint to the corresponding tensor constraint, i.e. the constraints
    on element type and shape are the same as the input constraint, but the attribute is verified to be
    a tensor instead of a memref.
    """

    memref_constraint: GenericAttrConstraint[
        MemRefType[Attribute] | UnrankedMemRefType[Attribute]
    ]

    def can_infer(self, var_constraint_names: Set[str]) -> bool:
        return self.memref_constraint.can_infer(var_constraint_names)

    def infer(
        self, context: ConstraintContext
    ) -> TensorType[Attribute] | UnrankedTensorType[Attribute]:
        memref_type = self.memref_constraint.infer(context)
        if isinstance(memref_type, MemRefType):
            return TensorType(memref_type.element_type, memref_type.shape)
        return UnrankedTensorType(memref_type.element_type)

    def verify(self, attr: Attribute, constraint_context: ConstraintContext) -> None:
        if isa(attr, TensorType[Attribute]):
            memref_type = MemRefType(attr.element_type, attr.shape)
        elif isa(attr, UnrankedTensorType[Attribute]):
            memref_type = UnrankedMemRefType.from_type(attr.element_type)
        else:
            raise VerifyException(
                f"Expected tensor or unranked tensor type, got {attr}"
            )
        return self.memref_constraint.verify(memref_type, constraint_context)


@irdl_op_definition
class AllocTensorOp(IRDLOperation):
    """
    `bufferization.alloc_tensor` materializes an uninitialized tensor with a
    given shape (dynamic or static). It always bufferizes to a new buffer
    allocation of the given shape. The optional `copy` operand specifies the
    contents of the tensors. If no `copy` operand is specified, reading from the
    result of an `alloc_tensor` op yields an undefined value.

    If `copy` is specified, no dynamic sizes should be passed, since they are
    the same as the dynamic sizes of the `copy` operand.

    `alloc_tensor` is a helper op for bufferization. The operation is provided
    as an anchor that marks the beginning of a new tensor SSA use-def chain. It
    can be used to control in-place bufferization decisions during One-Shot
    Bufferize: The bufferized result of a `bufferization.alloc_tensor` does not
    alias with any other buffer, so it can be used to resolve read-after-write
    conflicts that would have been introduced by the in-place bufferization of
    another op.

    The optional `memory_space` attribute specifies the memory space when
    bufferizing this op. The memory space is inferred from `copy` if specified.
    If neither `copy` nor `memory_space` is specified, the default memory space
    is used during bufferization.

    The optional `size_hint` operand specifies the number of non-zero elements
    for sparse tensors. The value of `size_hint` should be not less than 1 and
    not larger than the linear size of the corresponding dense tensor type. If
    this requirement is not met, the behavior of the operator is undefined.

    Note: An `alloc_tensor` with a `copy` should also be expressed as an
    `alloc_tensor` without `copy`, followed by a `copy_tensor`.

    https://mlir.llvm.org/docs/Dialects/BufferizationOps/#bufferizationalloc_tensor-bufferizationalloctensorop
    """

    name = "bufferization.alloc_tensor"

    T: ClassVar = VarConstraint("T", AnyTensorTypeConstr | AnyUnrankedTensorTypeConstr)

    dynamic_sizes = var_operand_def(IndexType())
    copy = opt_operand_def(T)
    size_hint = opt_operand_def(IndexType())

    tensor = result_def(T)

    irdl_options = [AttrSizedOperandSegments(as_property=True)]

    assembly_format = "`(` $dynamic_sizes `)` ( `copy` `(` $copy^ `)`)? (`size_hint` `=` $size_hint^)? attr-dict `:` type($tensor)"  # noqa E501

    def __init__(
        self,
        result_type: Attribute,
        dynamic_sizes: Sequence[Operation | SSAValue] | None = None,
        copy: SSAValue | Operation | None = None,
        size_hint: SSAValue | Operation | None = None,
    ):
        super().__init__(
            operands=(dynamic_sizes, copy, size_hint),
            result_types=(result_type,),
        )


@irdl_op_definition
class CloneOp(IRDLOperation):
    name = "bufferization.clone"

    T: ClassVar = VarConstraint("T", MemRefType.constr() | AnyUnrankedMemRefTypeConstr)

    input = operand_def(T)
    output = result_def(T)

    assembly_format = "$input attr-dict `:` type($input) `to` type($output)"

    def __init__(self, input: SSAValue | Operation):
        result_type = SSAValue.get(input).type
        super().__init__(operands=(input,), result_types=(result_type,))


@irdl_op_definition
class ToTensorOp(IRDLOperation):
    name = "bufferization.to_tensor"

    T: ClassVar = VarConstraint("T", MemRefType.constr() | AnyUnrankedMemRefTypeConstr)

    memref = operand_def(T)
    tensor = result_def(TensorFromMemRefConstraint(T))

    writable = opt_prop_def(UnitAttr)
    restrict = opt_prop_def(UnitAttr)

    assembly_format = "$memref (`restrict` $restrict^)? (`writable` $writable^)? attr-dict `:` type($memref) `to` type($tensor)"

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
class ToMemRefOp(IRDLOperation):
    name = "bufferization.to_memref"

    T: ClassVar = VarConstraint("T", MemRefType.constr() | AnyUnrankedMemRefTypeConstr)
    tensor = operand_def(TensorFromMemRefConstraint(T))
    memref = result_def(T)

    read_only = opt_prop_def(UnitAttr)

    assembly_format = "$tensor (`read_only` $read_only^)?  `:` attr-dict type($tensor) `to` type($memref)"


@irdl_op_definition
class MaterializeInDestinationOp(IRDLOperation):
    name = "bufferization.materialize_in_destination"

    T: ClassVar = VarConstraint("T", MemRefType.constr() | AnyUnrankedMemRefTypeConstr)
    source = operand_def(TensorFromMemRefConstraint(T))
    dest = operand_def(T | TensorFromMemRefConstraint(T))
    result = opt_result_def(TensorFromMemRefConstraint(T))

    restrict = opt_prop_def(UnitAttr)
    writable = opt_prop_def(UnitAttr)

    assembly_format = "$source `in` (`restrict` $restrict^)? (`writable` $writable^)? $dest attr-dict `:` functional-type(operands, results)"  # noqa: E501


Bufferization = Dialect(
    "bufferization",
    [
        AllocTensorOp,
        CloneOp,
        ToTensorOp,
        ToMemRefOp,
        MaterializeInDestinationOp,
    ],
    [],
)
