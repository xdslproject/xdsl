from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar, Any, cast, Iterable, Iterator, List

from xdsl.dialects import builtin
from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntegerAttr,
    ParametrizedAttribute,
    ArrayAttr,
    f32,
    f64,
    IntegerType,
    IntAttr,
    AnyFloat,
)
from xdsl.ir import Operation, Dialect, TypeAttribute
from xdsl.ir import SSAValue

from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    AttrConstraint,
    Attribute,
    Region,
    VerifyException,
    Generic,
    AnyOf,
    Annotated,
    Operand,
    OpAttr,
    OpResult,
    VarOperand,
    VarOpResult,
    OptOpAttr,
    AttrSizedOperandSegments,
    Block,
    IRDLOperation,
)
from xdsl.utils.hints import isa


_FieldTypeElement = TypeVar("_FieldTypeElement", bound=Attribute)


@irdl_attr_definition
class FieldType(Generic[_FieldTypeElement], ParametrizedAttribute, TypeAttribute):
    name = "stencil.field"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_FieldTypeElement]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

    def verify(self):
        if self.get_num_dims() <= 0:
            raise VerifyException(
                f"Number of field dimensions must be greater than zero, got {self.get_num_dims()}."
            )

    def __init__(
        self,
        shape: ArrayAttr[AnyIntegerAttr] | Sequence[AnyIntegerAttr] | Sequence[int],
        typ: _FieldTypeElement,
    ) -> None:
        if isinstance(shape, ArrayAttr):
            super().__init__([shape, typ])
            return

        # cast to list
        shape = cast(list[int], shape)
        super().__init__(
            [ArrayAttr([IntegerAttr[IntegerType](d, 64) for d in shape]), typ]
        )


@irdl_attr_definition
class TempType(Generic[_FieldTypeElement], ParametrizedAttribute, TypeAttribute):
    name = "stencil.temp"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_FieldTypeElement]

    def get_num_dims(self) -> int:
        return len(self.shape.data)

    def get_shape(self) -> List[int]:
        return [i.value.data for i in self.shape.data]

    def verify(self):
        if self.get_num_dims() <= 0:
            raise VerifyException(
                f"Number of field dimensions must be greater than zero, got {self.get_num_dims()}."
            )

    def __init__(
        self,
        shape: ArrayAttr[AnyIntegerAttr] | Sequence[AnyIntegerAttr] | Sequence[int],
        typ: _FieldTypeElement,
    ) -> None:
        if isinstance(shape, ArrayAttr):
            super().__init__([shape, typ])
            return

        # cast to list
        shape = cast(list[int], shape)
        super().__init__(
            [ArrayAttr([IntegerAttr[IntegerType](d, 64) for d in shape]), typ]
        )

    def __repr__(self):
        repr: str = "stencil.Temp<["
        for size in self.shape.data:
            repr += f"{size.value.data} "
        repr += "]>"
        return repr


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
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
                f"Expected {ArrayAttr} attribute, but got {attr.name}."
            )
        attr = cast(ArrayAttr[Any], attr)
        if len(attr.data) != self.length:
            raise VerifyException(
                f"Expected array of length {self.length}, got {len(attr.data)}."
            )


# TODO: How can we inherit from TypeAttribute and ParametrizedAttribute?
@dataclass(frozen=True)
class ElementType(ParametrizedAttribute):
    name = "stencil.element"
    element = AnyOf([f32, f64])


@irdl_attr_definition
class IndexAttr(ParametrizedAttribute, Iterable[int]):
    # TODO: can you have an attr and an op with the same name?
    name = "stencil.index"

    array: ParameterDef[ArrayAttr[IntegerAttr[IntegerType]]]

    def verify(self) -> None:
        if len(self.array.data) < 1 or len(self.array.data) > 3:
            raise VerifyException(
                f"Expected 1 to 3 indexes for stencil.index, got {len(self.array.data)}."
            )

    @staticmethod
    def get(*indices: int | IntegerAttr[IntegerType]):
        return IndexAttr(
            [
                ArrayAttr(
                    [
                        (
                            IntegerAttr[IntegerType](idx, 64)
                            if isinstance(idx, int)
                            else idx
                        )
                        for idx in indices
                    ]
                )
            ]
        )

    @staticmethod
    def size_from_bounds(lb: IndexAttr, ub: IndexAttr) -> list[int]:
        return [
            ub.value.data - lb.value.data
            for lb, ub in zip(lb.array.data, ub.array.data)
        ]

    # TODO : come to an agreement on, do we want to allow that kind of things
    # on Attributes? Author's opinion is a clear yes :P
    def __neg__(self) -> IndexAttr:
        integer_attrs: list[Attribute] = [
            IntegerAttr(-e.value.data, IntegerType(64)) for e in self.array.data
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

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(e.value.data for e in self.array.data)

    def __len__(self):
        return len(self.array)

    def __iter__(self) -> Iterator[int]:
        return (e.value.data for e in self.array.data)


@dataclass(frozen=True)
class LoopAttr(ParametrizedAttribute):
    name = "stencil.loop"
    shape = Annotated[ArrayAttr[IntAttr], ArrayLength(4)]


# Operations
@irdl_op_definition
class ExternalLoadOp(IRDLOperation):
    """
    This operation loads from an external field type, e.g. to bring data into the stencil

    Example:
      %0 = stencil.external_load %in : (!fir.array<128x128xf64>) -> !stencil.field<128x128xf64> # noqa
    """

    name = "stencil.external_load"
    field: Annotated[Operand, Attribute]
    result: Annotated[OpResult, FieldType | memref.MemRefType]

    @staticmethod
    def get(
        arg: SSAValue | Operation,
        res_type: FieldType[Attribute] | memref.MemRefType[Attribute],
    ):
        return ExternalLoadOp.build(operands=[arg], result_types=[res_type])


@irdl_op_definition
class ExternalStoreOp(IRDLOperation):
    """
    This operation takes a stencil field and then stores this to an external type

    Example:
      stencil.store %temp to %field : !stencil.field<128x128xf64> to !fir.array<128x128xf64> # noqa
    """

    name = "stencil.external_store"
    temp: Annotated[Operand, FieldType]
    field: Annotated[Operand, Attribute]


@irdl_op_definition
class IndexOp(IRDLOperation):
    """
    This operation returns the index of the current loop iteration for the
    chosen direction (0, 1, or 2).
    The offset is specified relative to the current position.

    Example:
      %0 = stencil.index 0 [-1, 0, 0] : index
    """

    name = "stencil.index"
    dim: OpAttr[AnyIntegerAttr]
    offset: OpAttr[IndexAttr]
    idx: Annotated[OpResult, builtin.IndexType]


@irdl_op_definition
class AccessOp(IRDLOperation):
    """
    This operation accesses a temporary element given a constant
    offset. The offset is specified relative to the current position.

    Example:
      %0 = stencil.access %temp [-1, 0, 0] : !stencil.temp<?x?x?xf64> -> f64
    """

    name = "stencil.access"
    temp: Annotated[Operand, TempType]
    offset: OpAttr[IndexAttr]
    res: Annotated[OpResult, Attribute]

    @staticmethod
    def get(temp: SSAValue | Operation, offset: Sequence[int]):
        temp_type = SSAValue.get(temp).typ
        assert isinstance(temp_type, TempType)
        temp_type = cast(TempType[Attribute], temp_type)

        return AccessOp.build(
            operands=[temp],
            attributes={
                "offset": IndexAttr(
                    [
                        ArrayAttr(
                            IntegerAttr[IntegerType](value, 64) for value in offset
                        ),
                    ]
                ),
            },
            result_types=[temp_type.element_type],
        )


@irdl_op_definition
class DynAccessOp(IRDLOperation):
    """
    This operation accesses a temporary element given a dynamic offset.
    The offset is specified in absolute coordinates. An additional
    range attribute specifies the maximal access extent relative to the
    iteration domain of the parent apply operation.

    Example:
      %0 = stencil.dyn_access %temp (%i, %j, %k) in [-1, -1, -1] : [1, 1, 1] : !stencil.temp<?x?x?xf64> -> f64
    """

    name = "stencil.dyn_access"
    temp: Annotated[Operand, TempType]
    offset: OpAttr[IndexAttr]
    lb: OpAttr[IndexAttr]
    ub: OpAttr[IndexAttr]
    res: Annotated[OpResult, ElementType]


@irdl_op_definition
class LoadOp(IRDLOperation):
    """
    This operation takes a field and returns a temporary values.

    Example:
      %0 = stencil.load %field : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
    """

    name = "stencil.load"
    field: Annotated[Operand, FieldType]
    lb: OptOpAttr[IndexAttr]
    ub: OptOpAttr[IndexAttr]
    res: Annotated[OpResult, TempType]

    @staticmethod
    def get(
        field: SSAValue | Operation,
        lb: IndexAttr | None = None,
        ub: IndexAttr | None = None,
    ):
        field_t = SSAValue.get(field).typ
        assert isinstance(field_t, FieldType)
        field_t = cast(FieldType[Attribute], field_t)

        return LoadOp.build(
            operands=[field],
            attributes={
                "lb": lb,
                "ub": ub,
            },
            result_types=[
                TempType[Attribute](
                    [-1] * len(field_t.shape.data), field_t.element_type
                )
            ],
        )

    def verify_(self) -> None:
        for use in self.field.uses:
            if isa(use.operation, StoreOp):
                raise VerifyException("Cannot Load and Store the same field!")


@irdl_op_definition
class BufferOp(IRDLOperation):
    """
    Prevents fusion of consecutive stencil.apply operations.

    Example:
      %0 = stencil.buffer %buffered : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    """

    name = "stencil.buffer"
    temp: Annotated[Operand, TempType]
    lb: OpAttr[IndexAttr]
    ub: OpAttr[IndexAttr]
    res: Annotated[OpResult, TempType]


@irdl_op_definition
class StoreOp(IRDLOperation):
    """
    This operation takes a temp and writes a field on a user defined range.

    Example:
      stencil.store %temp to %field ([0,0,0] : [64,64,60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>
    """

    name = "stencil.store"
    temp: Annotated[Operand, TempType]
    field: Annotated[Operand, FieldType]
    lb: OpAttr[IndexAttr]
    ub: OpAttr[IndexAttr]

    @staticmethod
    def get(
        temp: SSAValue | Operation,
        field: SSAValue | Operation,
        lb: IndexAttr,
        ub: IndexAttr,
    ):
        return StoreOp.build(operands=[temp, field], attributes={"lb": lb, "ub": ub})

    def verify_(self) -> None:
        for use in self.field.uses:
            if isa(use.operation, LoadOp):
                raise VerifyException("Cannot Load and Store the same field!")


@irdl_op_definition
class ApplyOp(IRDLOperation):
    """
    This operation takes a stencil function plus parameters and applies
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply (%arg0=%0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
        ...
      }
    """

    name = "stencil.apply"
    args: Annotated[VarOperand, Attribute]
    lb: OptOpAttr[IndexAttr]
    ub: OptOpAttr[IndexAttr]
    region: Region
    res: Annotated[VarOpResult, TempType]

    @staticmethod
    def get(
        args: Sequence[SSAValue] | Sequence[Operation],
        body: Block,
        result_types: Sequence[TempType[_FieldTypeElement]],
        lb: IndexAttr | None = None,
        ub: IndexAttr | None = None,
    ):
        assert len(result_types) > 0

        attributes = {}
        if lb is not None:
            attributes["lb"] = lb
        if ub is not None:
            attributes["ub"] = ub

        return ApplyOp.build(
            operands=[list(args)],
            attributes=attributes,
            regions=[Region(body)],
            result_types=[result_types],
        )


@irdl_op_definition
class StoreResultOp(IRDLOperation):
    """
    The store_result operation either stores an operand value or nothing.

    Examples:
      stencil.store_result %0 : !stencil.result<f64>
      stencil.store_result : !stencil.result<f64>
    """

    name = "stencil.store_result"
    args: Annotated[VarOperand, Attribute]
    res: Annotated[OpResult, ResultType]


@irdl_op_definition
class ReturnOp(IRDLOperation):
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

    name = "stencil.return"
    arg: Annotated[VarOperand, ResultType | AnyFloat]

    @staticmethod
    def get(res: Sequence[SSAValue | Operation]):
        return ReturnOp.build(operands=[list(res)])


@irdl_op_definition
class CombineOp(IRDLOperation):
    """
    Combines the results computed on a lower with the results computed on
    an upper domain. The operation combines the domain at a given index/offset
    in a given dimension. Optional extra operands allow to combine values
    that are only written / defined on the lower or upper subdomain. The result
    values have the order upper/lower, lowerext, upperext.

    Example:
      %result = stencil.combine 2 at 11 lower = (%0 : !stencil.temp<?x?x?xf64>) upper = (%1 : !stencil.temp<?x?x?xf64>) lowerext = (%2 : !stencil.temp<?x?x?xf64>): !stencil.temp<?x?x?xf64>, !stencil.temp<?x?x?xf64>
    """

    name = "stencil.combine"
    dim: Annotated[
        Operand, IntegerType
    ]  # TODO: how to use the ArrayLength constraint here? 0 <= dim <= 2
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


@irdl_op_definition
class HaloSwapOp(IRDLOperation):
    name = "stencil.halo_swap"

    input_stencil: Annotated[Operand, TempType]

    buff_lb: OptOpAttr[IndexAttr]
    buff_ub: OptOpAttr[IndexAttr]
    core_lb: OptOpAttr[IndexAttr]
    core_ub: OptOpAttr[IndexAttr]

    @staticmethod
    def get(input_stencil: SSAValue | Operation):
        return HaloSwapOp.build(operands=[input_stencil])


StencilExp = Dialect(
    [
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
        HaloSwapOp,
    ],
    [
        FieldType,
        TempType,
        ResultType,
        ElementType,
        IndexAttr,
        LoopAttr,
    ],
)
