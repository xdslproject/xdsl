from __future__ import annotations

from typing import Sequence, TypeVar, cast, Iterable, Iterator, List

from xdsl.dialects import builtin
from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IntAttr,
    IntegerAttr,
    ParametrizedAttribute,
    ArrayAttr,
    IntegerType,
    AnyFloat,
)
from xdsl.ir import Attribute, Operation, Dialect, TypeAttribute
from xdsl.ir import SSAValue

from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    ParameterDef,
    Attribute,
    Region,
    VerifyException,
    Generic,
    Annotated,
    Operand,
    OpAttr,
    OpResult,
    VarOperand,
    VarOpResult,
    OptOpAttr,
    Block,
    IRDLOperation,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.hints import isa


_FieldTypeElement = TypeVar("_FieldTypeElement", bound=Attribute, covariant=True)


class StencilType(Generic[_FieldTypeElement], ParametrizedAttribute, TypeAttribute):
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

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        parser.parse_char("<")
        dims, element_type = parser.parse_ranked_shape()
        parser.parse_char(">")
        return [ArrayAttr([IntegerAttr(d, 64) for d in dims]), element_type]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<")
        printer.print_list(
            (e.value.data for e in self.shape.data),
            lambda i: printer.print(i) if i != -1 else printer.print("?"),
            "x",
        )
        printer.print("x")
        printer.print_attribute(self.element_type)
        printer.print(">")

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
        return f"{self.name}<[{' '.join(str(d.value.data) for d in self.shape)}]>"


@irdl_attr_definition
class FieldType(
    Generic[_FieldTypeElement],
    StencilType[_FieldTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
):
    name = "stencil.field"

    def __repr__(self):
        return super().__repr__()


@irdl_attr_definition
class TempType(
    Generic[_FieldTypeElement],
    StencilType[_FieldTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
):
    name = "stencil.temp"

    def __repr__(self):
        return super().__repr__()


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
    name = "stencil.result"
    elem: ParameterDef[AnyFloat]

    def __init__(self, float_t: AnyFloat) -> None:
        super().__init__([float_t])


@irdl_attr_definition
class IndexAttr(ParametrizedAttribute, Iterable[int]):
    # TODO: can you have an attr and an op with the same name?
    name = "stencil.index"

    array: ParameterDef[ArrayAttr[IntAttr]]

    @staticmethod
    def parse_parameters(parser: Parser) -> list[Attribute]:
        """Parse the attribute parameters."""
        ints = parser.parse_comma_separated_list(
            parser.Delimiter.ANGLE, lambda: parser.parse_integer(allow_boolean=False)
        )
        return [ArrayAttr((IntAttr(i) for i in ints))]

    def print_parameters(self, printer: Printer) -> None:
        printer.print(f'<{", ".join((str(e.data) for e in self.array.data))}>')

    def verify(self) -> None:
        if len(self.array.data) < 1 or len(self.array.data) > 3:
            raise VerifyException(
                f"Expected 1 to 3 indexes for stencil.index, got {len(self.array.data)}."
            )

    @staticmethod
    def get(*indices: int | IntAttr):
        return IndexAttr(
            [
                ArrayAttr(
                    [(IntAttr(idx) if isinstance(idx, int) else idx) for idx in indices]
                )
            ]
        )

    @staticmethod
    def size_from_bounds(lb: IndexAttr, ub: IndexAttr) -> list[int]:
        return [ub.data - lb.data for lb, ub in zip(lb.array.data, ub.array.data)]

    # TODO : come to an agreement on, do we want to allow that kind of things
    # on Attributes? Author's opinion is a clear yes :P
    def __neg__(self) -> IndexAttr:
        return IndexAttr.get(*(-e.data for e in self.array.data))

    def __add__(self, o: IndexAttr) -> IndexAttr:
        return IndexAttr.get(
            *(se.data + oe.data for se, oe in zip(self.array.data, o.array.data))
        )

    def __sub__(self, o: IndexAttr) -> IndexAttr:
        return self + -o

    @staticmethod
    def min(a: IndexAttr, b: IndexAttr | None) -> IndexAttr:
        if b is None:
            return a
        return IndexAttr.get(
            *(min(ae.data, be.data) for ae, be in zip(a.array.data, b.array.data))
        )

    @staticmethod
    def max(a: IndexAttr, b: IndexAttr | None) -> IndexAttr:
        if b is None:
            return a
        return IndexAttr.get(
            *(max(ae.data, be.data) for ae, be in zip(a.array.data, b.array.data))
        )

    def as_tuple(self) -> tuple[int, ...]:
        return tuple(e.data for e in self.array.data)

    def __len__(self):
        return len(self.array)

    def __iter__(self) -> Iterator[int]:
        return (e.data for e in self.array.data)


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
                        ArrayAttr(IntAttr(value) for value in offset),
                    ]
                ),
            },
            result_types=[temp_type.element_type],
        )


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
        result_types: Sequence[TempType[Attribute]],
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


StencilExp = Dialect(
    [
        ExternalLoadOp,
        ExternalStoreOp,
        IndexOp,
        AccessOp,
        LoadOp,
        BufferOp,
        StoreOp,
        ApplyOp,
        StoreResultOp,
        ReturnOp,
    ],
    [
        FieldType,
        TempType,
        ResultType,
        IndexAttr,
    ],
)
