from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import TypeVar, deprecated

from xdsl.dialects.builtin import (
    AnyTensorTypeConstr,
    AnyUnrankedMemRefTypeConstr,
    AnyUnrankedTensorTypeConstr,
    IndexType,
    MemRefType,
    TensorType,
    UnitAttr,
    UnrankedMemRefType,
    UnrankedTensorType,
)
from xdsl.ir import Attribute, Dialect, Operation, SSAValue
from xdsl.irdl import (
    AnyAttr,
    AttrConstraint,
    AttrSizedOperandSegments,
    InferenceContext,
    IntConstraint,
    IRDLOperation,
    VarConstraint,
    VerificationContext,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    opt_result_def,
    result_def,
    var_operand_def,
    ParamAttrConstraint,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


@deprecated("Specify the constraints manually via element_type and shape")
@dataclass(frozen=True)
class TensorFromMemRefConstraint(
    AttrConstraint[TensorType[Attribute] | UnrankedTensorType[Attribute]]
):
    """
    Converts an input memref constraint to the corresponding tensor constraint, i.e. the constraints
    on element type and shape are the same as the input constraint, but the attribute is verified to be
    a tensor instead of a memref.
    """

    memref_constraint: AttrConstraint[MemRefType | UnrankedMemRefType]

    @staticmethod
    def tensor_to_memref(
        tensor: TensorType | UnrankedTensorType,
    ) -> MemRefType | UnrankedMemRefType:
        if isinstance(tensor, TensorType):
            return MemRefType(tensor.element_type, tensor.shape)
        else:
            return UnrankedMemRefType.from_type(tensor.element_type)

    @staticmethod
    def memref_to_tensor(
        memref: MemRefType | UnrankedMemRefType,
    ) -> TensorType | UnrankedTensorType:
        if isinstance(memref, MemRefType):
            return TensorType(memref.element_type, memref.shape)
        else:
            return UnrankedTensorType(memref.element_type)

    def can_infer(self, var_constraint_names: AbstractSet[str]) -> bool:
        return self.memref_constraint.can_infer(var_constraint_names)

    def infer(
        self, context: InferenceContext
    ) -> TensorType[Attribute] | UnrankedTensorType[Attribute]:
        memref_type = self.memref_constraint.infer(context)
        return self.memref_to_tensor(memref_type)

    def verify(self, attr: Attribute, constraint_context: VerificationContext) -> None:
        if isa(attr, TensorType | UnrankedTensorType):
            memref_type = self.tensor_to_memref(attr)
        else:
            raise VerifyException(
                f"Expected tensor or unranked tensor type, got {attr}"
            )
        return self.memref_constraint.verify(memref_type, constraint_context)

    def get_bases(self) -> set[type[Attribute]] | None:
        return {TensorType, UnrankedTensorType}

    def mapping_type_vars(
        self, type_var_mapping: Mapping[TypeVar, AttrConstraint | IntConstraint]
    ):
        return TensorFromMemRefConstraint(  # pyright: ignore[reportDeprecated]
            self.memref_constraint.mapping_type_vars(type_var_mapping)
        )


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

    irdl_options = (AttrSizedOperandSegments(as_property=True),)

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

    ELEM_T: ClassVar = VarConstraint("ELEM_T", AnyAttr())
    SHAPE: ClassVar = VarConstraint("SHAPE", AnyAttr())

    memref = operand_def(
        MemRefType.constr(ELEM_T, shape=SHAPE)
        | ParamAttrConstraint.get(UnrankedMemRefType, ELEM_T, AnyAttr())
    )
    tensor = result_def(
        TensorType.constr(ELEM_T, shape=SHAPE)
        | ParamAttrConstraint.get(UnrankedTensorType, ELEM_T)
    )

    writable = opt_prop_def(UnitAttr)
    restrict = opt_prop_def(UnitAttr)

    assembly_format = "$memref (`restrict` $restrict^)? (`writable` $writable^)? attr-dict `:` type($memref) `to` type($tensor)"

    def __init__(
        self,
        memref: SSAValue | Operation,
        restrict: bool = False,
        writable: bool = False,
        *,
        tensor_type: TensorType | UnrankedTensorType | None = None,
    ):
        memref_v = SSAValue.get(memref, type=MemRefType | UnrankedMemRefType)
        properties = dict[str, Attribute]()
        if restrict:
            properties["restrict"] = UnitAttr()
        if writable:
            properties["writable"] = UnitAttr()
        if tensor_type is None:
            tensor_type = self.__memref_to_tensor(memref_v.type)
        super().__init__(
            operands=(memref,),
            result_types=(tensor_type,),
            properties=properties,
        )

    @staticmethod
    def __memref_to_tensor(
        memref: MemRefType | UnrankedMemRefType,
    ) -> TensorType | UnrankedTensorType:
        if isinstance(memref, MemRefType):
            return TensorType(memref.element_type, memref.shape)
        else:
            return UnrankedTensorType(memref.element_type)


@irdl_op_definition
class ToBufferOp(IRDLOperation):
    name = "bufferization.to_buffer"

    ELEM_T: ClassVar = VarConstraint("ELEM_T", AnyAttr())
    SHAPE: ClassVar = VarConstraint("SHAPE", AnyAttr())

    tensor = operand_def(
        TensorType.constr(ELEM_T, shape=SHAPE)
        | ParamAttrConstraint.get(UnrankedTensorType, ELEM_T)
    )
    memref = result_def(
        MemRefType.constr(ELEM_T, shape=SHAPE)
        | ParamAttrConstraint.get(UnrankedMemRefType, ELEM_T, AnyAttr())
    )

    read_only = opt_prop_def(UnitAttr)

    assembly_format = "$tensor (`read_only` $read_only^)?  `:` attr-dict type($tensor) `to` type($memref)"


@irdl_op_definition
class MaterializeInDestinationOp(IRDLOperation):
    name = "bufferization.materialize_in_destination"

    ELEM_T: ClassVar = VarConstraint("ELEM_T", AnyAttr())
    SHAPE: ClassVar = VarConstraint("SHAPE", AnyAttr())
    TENSOR_CONSTR: ClassVar = TensorType.constr(
        ELEM_T, shape=SHAPE
    ) | ParamAttrConstraint.get(UnrankedTensorType, ELEM_T)

    source = operand_def(TENSOR_CONSTR)
    dest = operand_def(
        MemRefType.constr(ELEM_T, shape=SHAPE)
        | ParamAttrConstraint.get(UnrankedMemRefType, ELEM_T, AnyAttr())
        | TENSOR_CONSTR
    )
    result = opt_result_def(TENSOR_CONSTR)

    restrict = opt_prop_def(UnitAttr)
    writable = opt_prop_def(UnitAttr)

    assembly_format = "$source `in` (`restrict` $restrict^)? (`writable` $writable^)? $dest attr-dict `:` functional-type(operands, results)"  # noqa: E501


Bufferization = Dialect(
    "bufferization",
    [
        AllocTensorOp,
        CloneOp,
        ToTensorOp,
        ToBufferOp,
        MaterializeInDestinationOp,
    ],
    [],
)
