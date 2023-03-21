from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Sequence, TypeVar, Any, cast

from xdsl.dialects.builtin import (AnyIntegerAttr, IntegerAttr,
                                   ParametrizedAttribute, ArrayAttr, f32, f64,
                                   IntegerType, IntAttr, AnyFloat)
from xdsl.dialects import builtin
from xdsl.ir import Operation, Dialect, MLIRType
from xdsl.irdl import (AnyAttr, irdl_attr_definition, irdl_op_definition,
                       ParameterDef, AttrConstraint, Attribute, Region,
                       VerifyException, Generic, AnyOf, Annotated, Operand,
                       OpAttr, OpResult, VarOperand, VarOpResult, OptOpAttr,
                       AttrSizedOperandSegments)


@dataclass
class IntOrUnknown(AttrConstraint):
    length: int = 0

    def verify(self, attr: Attribute) -> None:
        if not isinstance(attr, ArrayAttr):
            raise VerifyException(
                f"Expected {ArrayAttr} attribute, but got {attr.name}.")

        attr = cast(ArrayAttr[Any], attr)
        if len(attr.data) != self.length:
            raise VerifyException(
                f"Expected array of length {self.length}, got {len(attr.data)}."
            )


_FieldTypeElement = TypeVar("_FieldTypeElement", bound=Attribute)


@irdl_attr_definition
class FieldType(Generic[_FieldTypeElement], ParametrizedAttribute, MLIRType):
    name = "stencil.field"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_FieldTypeElement]

    @staticmethod
    def from_shape(
            shape: list[int] | list[IntAttr]) -> FieldType[_FieldTypeElement]:
        # TODO: why do we need all these casts here, can we tell pyright "trust me"
        if all(isinstance(elm, IntAttr) for elm in shape):
            shape = cast(list[IntAttr], shape)
            return FieldType([ArrayAttr(shape)])

        shape = cast(list[int], shape)
        return FieldType([ArrayAttr([IntAttr.from_int(d) for d in shape])])


@irdl_attr_definition
class TempType(Generic[_FieldTypeElement], ParametrizedAttribute, MLIRType):
    name = "stencil.temp"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_FieldTypeElement]

    @staticmethod
    def from_shape(
        shape: ArrayAttr[IntAttr] | list[IntAttr] | list[int]
    ) -> TempType[_FieldTypeElement]:
        assert len(shape) > 0

        if isinstance(shape, ArrayAttr):
            return TempType.new([shape])

        # cast to list
        shape = cast(list[IntAttr] | list[int], shape)

        if isinstance(shape[0], IntAttr):
            # the if above is a sufficient type guard, but pyright does not understand :/
            return TempType([ArrayAttr(shape)])  # type: ignore
        shape = cast(list[int], shape)
        return TempType([ArrayAttr([IntAttr.from_int(d) for d in shape])])

    def __repr__(self):
        repr: str = "stencil.Temp<["
        for size in self.shape.data:
            repr += f"{size.value.data} "
        repr += "]>"
        return repr


@irdl_attr_definition
class ResultType(ParametrizedAttribute, MLIRType):
    name = "stencil.result"
    elem: ParameterDef[AnyFloat]

    @staticmethod
    def from_type(float_t: AnyFloat):
        return ResultType([float_t])


@dataclass
class ArrayLength(AttrConstraint):
    length: int = 0

    def verify(self, attr: Attribute) -> None:
        if not isinstance(attr, ArrayAttr):
            raise VerifyException(
                f"Expected {ArrayAttr} attribute, but got {attr.name}.")
        attr = cast(ArrayAttr[Any], attr)
        if len(attr.data) != self.length:
            raise VerifyException(
                f"Expected array of length {self.length}, got {len(attr.data)}."
            )


# TODO: How can we inherit from MLIRType and ParametrizedAttribute?
@dataclass(frozen=True)
class ElementType(ParametrizedAttribute):
    name = "stencil.element"
    element = AnyOf([f32, f64])


@irdl_attr_definition
class IndexAttr(ParametrizedAttribute):
    # TODO: can you have an attr and an op with the same name?
    name = "stencil.index"

    array: ParameterDef[ArrayAttr[AnyIntegerAttr]]

    def verify(self) -> None:
        if len(self.array.data) < 1 or len(self.array.data) > 3:
            raise VerifyException(
                f"Expected 1 to 3 indexes for stencil.index, got {len(self.array.data)}."
            )

    @staticmethod
    def size_from_bounds(lb: IndexAttr, ub: IndexAttr) -> Sequence[int]:
        return [
            ub.value.data - lb.value.data
            for lb, ub in zip(lb.array.data, ub.array.data)
        ]

    #TODO : come to an agreement on, do we want to allow that kind of things on
    # Attributes? Author's opinion is a clear yes :P
    def __neg__(self) -> IndexAttr:
        integer_attrs: list[Attribute] = [
            IntegerAttr(-e.value.data, IntegerType(64))
            for e in self.array.data
        ]
        return IndexAttr([ArrayAttr(integer_attrs)])

    def __add__(self, o: IndexAttr) -> IndexAttr:
        integer_attrs: list[Attribute] = [
            IntegerAttr(se.value.data + oe.value.data, IntegerType(64))
            for se, oe in zip(self.array.data, o.array.data)
        ]
        return IndexAttr([ArrayAttr(integer_attrs)])

    def __sub__(self, o: IndexAttr) -> IndexAttr:
        return self + -o

    @staticmethod
    def min(a: IndexAttr, b: IndexAttr | None) -> IndexAttr:
        if b is None:
            return a
        integer_attrs: list[Attribute] = [
            IntegerAttr(min(ae.value.data, be.value.data), IntegerType(64))
            for ae, be in zip(a.array.data, b.array.data)
        ]
        return IndexAttr([ArrayAttr(integer_attrs)])

    @staticmethod
    def max(a: IndexAttr, b: IndexAttr | None) -> IndexAttr:
        if b is None:
            return a
        integer_attrs: list[Attribute] = [
            IntegerAttr(max(ae.value.data, be.value.data), IntegerType(64))
            for ae, be in zip(a.array.data, b.array.data)
        ]
        return IndexAttr([ArrayAttr(integer_attrs)])


@dataclass(frozen=True)
class LoopAttr(ParametrizedAttribute):
    name = "stencil.loop"
    shape = Annotated[ArrayAttr[IntAttr], ArrayLength(4)]


# Operations
@irdl_op_definition
class CastOp(Operation):
    """
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
      %0 = stencil.cast %in ([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64>
    """
    name: str = "stencil.cast"
    field: Annotated[Operand, FieldType]
    lb: OpAttr[IndexAttr]
    ub: OpAttr[IndexAttr]
    result: Annotated[OpResult, FieldType]


# Operations
@irdl_op_definition
class ExternalLoadOp(Operation):
    """
    This operation loads from an external field type, e.g. to bring data into the stencil

    Example:
      %0 = stencil.external_load %in : (!fir.array<128x128xf64>) -> !stencil.field<128x128xf64>
    """
    name: str = "stencil.external_load"
    field: Annotated[Operand, Attribute]
    result: Annotated[OpResult, FieldType]


@irdl_op_definition
class ExternalStoreOp(Operation):
    """
    This operation takes a stencil field and then stores this to an external type

    Example:
      stencil.store %temp to %field : !stencil.field<128x128xf64> to !fir.array<128x128xf64>
    """
    name: str = "stencil.external_store"
    temp: Annotated[Operand, FieldType]
    field: Annotated[Operand, Attribute]


@irdl_op_definition
class IndexOp(Operation):
    """
    This operation returns the index of the current loop iteration for the
    chosen direction (0, 1, or 2).
    The offset is specified relative to the current position.

    Example:
      %0 = stencil.index 0 [-1, 0, 0] : index
    """
    name: str = "stencil.index"
    dim: OpAttr[IntegerType]
    offset: OpAttr[IndexAttr]
    idx: Annotated[OpResult, builtin.IndexType]


@irdl_op_definition
class AccessOp(Operation):
    """
    This operation accesses a temporary element given a constant
    offset. The offset is specified relative to the current position.

    Example:
      %0 = stencil.access %temp [-1, 0, 0] : !stencil.temp<?x?x?xf64> -> f64
    """
    name: str = "stencil.access"
    temp: Annotated[Operand, TempType]
    offset: OpAttr[IndexAttr]
    res: Annotated[OpResult, Attribute]


@irdl_op_definition
class DynAccessOp(Operation):
    """
    This operation accesses a temporary element given a dynamic offset.
    The offset is specified in absolute coordinates. An additional
    range attribute specifies the maximal access extent relative to the
    iteration domain of the parent apply operation.

    Example:
      %0 = stencil.dyn_access %temp (%i, %j, %k) in [-1, -1, -1] : [1, 1, 1] : !stencil.temp<?x?x?xf64> -> f64
    """
    name: str = "stencil.dyn_access"
    temp: Annotated[Operand, TempType]
    offset: OpAttr[IndexAttr]
    lb: OpAttr[IndexAttr]
    ub: OpAttr[IndexAttr]
    res: Annotated[OpResult, ElementType]


@irdl_op_definition
class LoadOp(Operation):
    """
    This operation takes a field and returns a temporary values.

    Example:
      %0 = stencil.load %field : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
    """
    name: str = "stencil.load"
    field: Annotated[Operand, FieldType]
    lb: OptOpAttr[IndexAttr]
    ub: OptOpAttr[IndexAttr]
    res: Annotated[OpResult, TempType]


@irdl_op_definition
class BufferOp(Operation):
    """
    Prevents fusion of consecutive stencil.apply operations.

    Example:
      %0 = stencil.buffer %buffered : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    """
    name: str = "stencil.buffer"
    temp: Annotated[Operand, TempType]
    lb: OpAttr[IndexAttr]
    ub: OpAttr[IndexAttr]
    res: Annotated[OpResult, TempType]


@irdl_op_definition
class StoreOp(Operation):
    """
    This operation takes a temp and writes a field on a user defined range.

    Example:
      stencil.store %temp to %field ([0,0,0] : [64,64,60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
    """
    name: str = "stencil.store"
    temp: Annotated[Operand, TempType]
    field: Annotated[Operand, FieldType]
    lb: OpAttr[IndexAttr]
    ub: OpAttr[IndexAttr]


@irdl_op_definition
class ApplyOp(Operation):
    """
    This operation takes a stencil function plus parameters and applies
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply (%arg0=%0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
        ...
      }
    """
    name: str = "stencil.apply"
    args: Annotated[VarOperand, AnyAttr()]
    lb: OptOpAttr[IndexAttr]
    ub: OptOpAttr[IndexAttr]
    region: Region
    res: Annotated[VarOpResult, TempType]


@irdl_op_definition
class StoreResultOp(Operation):
    """
    The store_result operation either stores an operand value or nothing.

    Examples:
      stencil.store_result %0 : !stencil.result<f64>
      stencil.store_result : !stencil.result<f64>
    """
    name: str = "stencil.store_result"
    args: Annotated[VarOperand, Attribute]
    res: Annotated[OpResult, ResultType]


@irdl_op_definition
class ReturnOp(Operation):
    """
    The return operation terminates the stencil apply and writes
    the results of the stencil operator to the temporary values returned
    by the stencil apply operation. The types and the number of operands
    must match the results of the stencil apply operation.

    The optional unroll attribute enables the implementation of loop
    unrolling at the stencil dialect level.

    Examples:
      stencil.return %0 : !stencil.result<f64>
    """
    name: str = "stencil.return"
    arg: Annotated[Operand, ResultType | AnyFloat]


@irdl_op_definition
class CombineOp(Operation):
    """
    Combines the results computed on a lower with the results computed on
    an upper domain. The operation combines the domain at a given index/offset
    in a given dimension. Optional extra operands allow to combine values
    that are only written / defined on the lower or upper subdomain. The result
    values have the order upper/lower, lowerext, upperext.

    Example:
      %result = stencil.combine 2 at 11 lower = (%0 : !stencil.temp<?x?x?xf64>) upper = (%1 : !stencil.temp<?x?x?xf64>) lowerext = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
    """
    name: str = "stencil.combine"
    dim: Annotated[
        Operand,
        IntegerType]  # TODO: how to use the ArrayLength constraint here? 0 <= dim <= 2
    index: Annotated[Operand, IntegerType]

    lower: Annotated[VarOperand, TempType]
    upper: Annotated[VarOperand, TempType]
    lower_ext: Annotated[VarOperand, TempType]
    upper_ext: Annotated[VarOperand, TempType]

    lb: OptOpAttr[IndexAttr]
    ub: OptOpAttr[IndexAttr]

    region: Region
    res: VarOpResult

    irdl_options = [AttrSizedOperandSegments()]


Stencil = Dialect([
    CastOp,
    ExternalLoadOp,
    ExternalStoreOp,
    IndexOp,
    AccessOp,
    DynAccessOp,
    LoadOp,
    BufferOp,
    StoreOp,
    ApplyOp,
    StoreResultOp,
    ReturnOp,
    CombineOp,
], [
    FieldType,
    TempType,
    ResultType,
    ElementType,
    IndexAttr,
    LoopAttr,
])
