from __future__ import annotations

from operator import add, lt, neg
from typing import Generic, Iterable, Iterator, Sequence, TypeVar, cast

from xdsl.dialects import builtin, memref
from xdsl.dialects.builtin import AnyFloat, AnyIntegerAttr, ArrayAttr, IntAttr
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    Operation,
    OpResult,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.ir.core import ParametrizedAttribute
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    ParameterDef,
    VarOperand,
    VarOpResult,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import AttrParser
from xdsl.printer import Printer
from xdsl.traits import HasParent, IsolatedFromAbove, IsTerminator
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

_FieldTypeElement = TypeVar("_FieldTypeElement", bound=Attribute, covariant=True)


@irdl_attr_definition
class IndexAttr(ParametrizedAttribute, Iterable[int]):
    name = "stencil.index"

    array: ParameterDef[ArrayAttr[IntAttr]]

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        """Parse the attribute parameters."""
        ints = parser.parse_comma_separated_list(
            parser.Delimiter.ANGLE, lambda: parser.parse_integer(allow_boolean=False)
        )
        return [ArrayAttr(IntAttr(i) for i in ints)]

    def print_parameters(self, printer: Printer) -> None:
        printer.print(f'<{", ".join((str(e) for e in self))}>')

    def verify(self) -> None:
        l = len(self)
        if l < 1 or l > 3:
            raise VerifyException(
                f"Expected 1 to 3 indexes for stencil.index, got {l}."
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
        return [ub - lb for lb, ub in zip(lb, ub)]

    # TODO : come to an agreement on, do we want to allow that kind of things
    # on Attributes? Author's opinion is a clear yes :P
    def __neg__(self) -> IndexAttr:
        return IndexAttr.get(*(map(neg, self)))

    def __add__(self, o: IndexAttr) -> IndexAttr:
        return IndexAttr.get(*(map(add, self, o)))

    def __sub__(self, o: IndexAttr) -> IndexAttr:
        return self + -o

    def __lt__(self, o: IndexAttr) -> bool:
        return any(map(lt, self, o))

    @staticmethod
    def min(a: IndexAttr, b: IndexAttr | None) -> IndexAttr:
        if b is None:
            return a
        return IndexAttr.get(*map(min, a, b))

    @staticmethod
    def max(a: IndexAttr, b: IndexAttr | None) -> IndexAttr:
        if b is None:
            return a
        return IndexAttr.get(*map(max, a, b))

    def __len__(self):
        return len(self.array)

    def __iter__(self) -> Iterator[int]:
        return (e.data for e in self.array.data)


@irdl_attr_definition
class StencilBoundsAttr(ParametrizedAttribute):
    """
    This attribute represents known bounds over a stencil type.
    """

    name = "stencil.bounds"
    lb: ParameterDef[IndexAttr]
    ub: ParameterDef[IndexAttr]

    def _verify(self):
        if len(self.lb) != len(self.ub):
            raise VerifyException(
                "Incoherent stencil bounds: lower and upper bounds must have the "
                "same dimensionality."
            )
        for d in self.ub - self.lb:
            if d <= 0:
                raise VerifyException(
                    "Incoherent stencil bounds: upper bound must be strictly "
                    "greater than lower bound."
                )

    def __init__(self, bounds: Iterable[tuple[int | IntAttr, int | IntAttr]]):
        if bounds:
            lb, ub = zip(*bounds)
        else:
            lb, ub = (), ()
        super().__init__(
            [
                IndexAttr.get(*lb),
                IndexAttr.get(*ub),
            ]
        )


class StencilType(
    Generic[_FieldTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
    builtin.ShapedType,
    builtin.ContainerType[_FieldTypeElement],
):
    name = "stencil.type"
    bounds: ParameterDef[StencilBoundsAttr | IntAttr]
    """
    Represents the bounds information of a stencil.field or stencil.temp.

    A StencilBoundsAttr encodes known bounds, where an IntAttr encodes the
    rank of unknown bounds. A stencil.field or stencil.temp cannot be unranked!
    """
    element_type: ParameterDef[_FieldTypeElement]

    def get_num_dims(self) -> int:
        if isinstance(self.bounds, IntAttr):
            return self.bounds.data
        else:
            return len(self.bounds.ub.array.data)

    def get_shape(self) -> tuple[int, ...]:
        if isinstance(self.bounds, IntAttr):
            return (-1,) * self.bounds.data
        else:
            return tuple(self.bounds.ub - self.bounds.lb)

    def get_element_type(self) -> _FieldTypeElement:
        return self.element_type

    @classmethod
    def parse_parameters(cls, parser: AttrParser) -> list[Attribute]:
        def parse_interval() -> tuple[int, int] | int:
            if parser.parse_optional_punctuation("?"):
                return -1
            parser.parse_punctuation("[")
            l = parser.parse_integer(allow_boolean=False)
            parser.parse_punctuation(",")
            u = parser.parse_integer(allow_boolean=False)
            parser.parse_punctuation("]")
            return (l, u)

        parser.parse_characters("<")
        bounds = [parse_interval()]
        parser.parse_shape_delimiter()
        opt_type = parser.parse_optional_type()
        while opt_type is None:
            bounds.append(parse_interval())
            parser.parse_shape_delimiter()
            opt_type = parser.parse_optional_type()
        parser.parse_characters(">")
        if isa(bounds, list[tuple[int, int]]):
            bounds = StencilBoundsAttr(bounds)
        elif isa(bounds, list[int]):
            bounds = IntAttr(len(bounds))
        else:
            parser.raise_error("stencil types can only be fully dynamic or sized.")

        return [bounds, opt_type]

    def print_parameters(self, printer: Printer) -> None:
        printer.print("<")
        if isinstance(self.bounds, StencilBoundsAttr):
            printer.print_list(
                zip(self.bounds.lb, self.bounds.ub),
                lambda b: printer.print(f"[{b[0]},{b[1]}]"),
                "x",
            )
            printer.print("x")
        else:
            for _ in range(self.bounds.data):
                printer.print("?x")
        printer.print_attribute(self.element_type)
        printer.print(">")

    def __init__(
        self,
        bounds: Iterable[tuple[int | IntAttr, int | IntAttr]]
        | int
        | IntAttr
        | StencilBoundsAttr,
        element_type: _FieldTypeElement,
    ) -> None:
        """
            A StencilBoundsAttr encodes known bounds, where an IntAttr encodes the
        rank of unknown bounds. A stencil.field or stencil.temp cannot be unranked!

        ### examples:

        - `Field(3,f32)` is represented as `stencil.field<?x?x?xf32>`
        - `Field([(-1,17),(-2,18)],f32)` is represented as `stencil.field<[-1,17]x[-2,18]xf32>`,
        """
        if isinstance(bounds, Iterable):
            nbounds = StencilBoundsAttr(bounds)
        elif isinstance(bounds, int):
            nbounds = IntAttr(bounds)
        else:
            nbounds = bounds
        return super().__init__([nbounds, element_type])


@irdl_attr_definition
class FieldType(
    Generic[_FieldTypeElement],
    StencilType[_FieldTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
):
    """
    stencil.field represents memory from which stencil input values will be loaded,
    or to which stencil output values will be stored.

    stencil.temp are loaded from or stored to stencil.field
    """

    name = "stencil.field"


@irdl_attr_definition
class TempType(
    Generic[_FieldTypeElement],
    StencilType[_FieldTypeElement],
    ParametrizedAttribute,
    TypeAttribute,
):
    """
    stencil.temp represents stencil values, and is the type on which stencil.apply operates.
    It has value-semantics: it won't necesseraly be lowered to an actual buffer.
    """

    name = "stencil.temp"


@irdl_attr_definition
class ResultType(ParametrizedAttribute, TypeAttribute):
    name = "stencil.result"
    elem: ParameterDef[AnyFloat]

    def __init__(self, float_t: AnyFloat) -> None:
        super().__init__([float_t])


@irdl_op_definition
class ApplyOp(IRDLOperation):
    """
    This operation takes a stencil function plus parameters and applies
    the stencil function to the output temp.

    Example:

      %0 = stencil.apply (%arg0=%0 : !stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64> {
        ...
      }

    The computation bounds are defined by the bounds of the output types, which are
    constrained to be all equals.
    """

    name = "stencil.apply"
    args: VarOperand = var_operand_def(Attribute)
    region: Region = region_def()
    res: VarOpResult = var_result_def(TempType)

    traits = frozenset([IsolatedFromAbove()])

    @staticmethod
    def get(
        args: Sequence[SSAValue] | Sequence[Operation],
        body: Block,
        result_types: Sequence[TempType[Attribute]],
    ):
        assert len(result_types) > 0

        return ApplyOp.build(
            operands=[list(args)],
            regions=[Region(body)],
            result_types=[result_types],
        )

    def verify_(self) -> None:
        if len(self.res) < 1:
            raise VerifyException(
                f"Expected stencil.apply to have at least 1 result, got {len(self.res)}"
            )
        res_type = cast(TempType[Attribute], self.res[0].type)
        for other in self.res[1:]:
            other = cast(TempType[Attribute], other.type)
            if res_type.bounds != other.bounds:
                raise VerifyException(f"Expected all output types bounds to be equals.")

    def get_rank(self) -> int:
        res_type = self.res[0].type
        assert isa(res_type, TempType[Attribute])
        return res_type.get_num_dims()


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
        %0 = stencil.cast %in ([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64> # noqa
    """

    name = "stencil.cast"
    field: Operand = operand_def(FieldType)
    result: OpResult = result_def(FieldType)

    @staticmethod
    def get(
        field: SSAValue | Operation,
        bounds: StencilBoundsAttr,
        res_type: FieldType[_FieldTypeElement] | FieldType[Attribute] | None = None,
    ) -> CastOp:
        """ """
        field_ssa = SSAValue.get(field)
        assert isa(field_ssa.type, FieldType[Attribute])
        if res_type is None:
            res_type = FieldType(
                bounds,
                field_ssa.type.element_type,
            )
        return CastOp.build(
            operands=[field],
            result_types=[res_type],
        )

    def verify_(self) -> None:
        # this should be fine, verify() already checks them:
        assert isa(self.field.type, FieldType[Attribute])
        assert isa(self.result.type, FieldType[Attribute])

        if isinstance(self.result.type.bounds, IntAttr):
            raise VerifyException("Output type's size must be explicit")

        if self.field.type.element_type != self.result.type.element_type:
            raise VerifyException(
                "Input and output fields must have the same element types"
            )

        if self.field.type.get_num_dims() != self.result.type.get_num_dims():
            raise VerifyException("Input and output types must have the same rank")

        if (
            isinstance(self.field.type.bounds, StencilBoundsAttr)
            and self.field.type.bounds != self.result.type.bounds
        ):
            raise VerifyException(
                "If input shape is not dynamic, it must be the same as output"
            )


# Operations
@irdl_op_definition
class ExternalLoadOp(IRDLOperation):
    """
    This operation loads from an external field type, e.g. to bring data into the stencil

    Example:
      %0 = stencil.external_load %in : (!fir.array<128x128xf64>) -> !stencil.field<128x128xf64> # noqa
    """

    name = "stencil.external_load"
    field: Operand = operand_def(Attribute)
    result: OpResult = result_def(FieldType[Attribute] | memref.MemRefType[Attribute])

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
    temp: Operand = operand_def(FieldType)
    field: Operand = operand_def(Attribute)


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
    dim: AnyIntegerAttr = attr_def(AnyIntegerAttr)
    offset: IndexAttr = attr_def(IndexAttr)
    idx: OpResult = result_def(builtin.IndexType)


@irdl_op_definition
class AccessOp(IRDLOperation):
    """
    This operation accesses a value from a stencil.temp given the specified offset.
    offset. The offset is specified relative to the current position.

    The optional offset mapping will determine which offset corresponds to which
    result dimension and is needed when we are accessing an array which has fewer
    dimensions than the result. Dimensions are mapped from the inner loop, which is 0,
    incrementing with each outer nested loop. e.g. in a nest of three loops, 0 would be
    the inner loop, 1 the middle loop, and 2 the outer loop. We do not allow out of order
    mappings.

    Example:
      %0 = stencil.access %temp [-1, 0, 0] : !stencil.temp<?x?x?xf64> -> f64
    """

    name = "stencil.access"
    temp: Operand = operand_def(TempType)
    offset: IndexAttr = attr_def(IndexAttr)
    offset_mapping: ArrayAttr[IntAttr] | None = opt_attr_def(ArrayAttr[IntAttr])
    res: OpResult = result_def(Attribute)

    traits = frozenset([HasParent(ApplyOp)])

    @staticmethod
    def get(
        temp: SSAValue | Operation,
        offset: Sequence[int],
        offset_mapping: Sequence[int] | None = None,
    ):
        temp_type = SSAValue.get(temp).type
        assert isinstance(temp_type, TempType)
        temp_type = cast(TempType[Attribute], temp_type)

        attributes: dict[str, IndexAttr | ArrayAttr[IntAttr]] = {
            "offset": IndexAttr(
                [
                    ArrayAttr(IntAttr(value) for value in offset),
                ]
            ),
        }

        if offset_mapping is not None:
            attributes["offset_mapping"] = ArrayAttr(
                [IntAttr(value) for value in offset_mapping]
            )

        return AccessOp.build(
            operands=[temp],
            attributes=attributes,
            result_types=[temp_type.element_type],
        )

    def verify_(self) -> None:
        apply = self.parent_op()
        # As promised by HasParent(ApplyOp)
        assert isinstance(apply, ApplyOp)

        # TODO This should be handled by infra, having a way to verify things on ApplyOp
        # **before** its children.
        # cf https://github.com/xdslproject/xdsl/issues/1112
        apply.verify_()

        temp_type = self.temp.type
        assert isa(temp_type, TempType[Attribute])
        if temp_type.get_num_dims() != apply.get_rank():
            if self.offset_mapping is None:
                raise VerifyException(
                    f"Expected stencil.access operand to be of rank {apply.get_rank()} "
                    f"to match its parent apply, got {temp_type.get_num_dims()} without "
                    f"explict offset mapping provided"
                )

        if self.offset_mapping is not None and len(self.offset_mapping) != len(
            self.offset
        ):
            raise VerifyException(
                f"Expected stencil.access offset mapping be of length {len(self.offset)} "
                f"to match the provided offsets, but it is {len(self.offset_mapping)} "
                f"instead"
            )

        if self.offset_mapping is not None:
            prev_offset = None
            for offset in self.offset_mapping:
                if prev_offset is not None:
                    if offset.data >= prev_offset:
                        raise VerifyException(
                            f"Offset mapping in stencil.access must be strictly "
                            f"decreasing and unique, however {offset.data} follows "
                            f"{prev_offset} which is disallowed"
                        )
                prev_offset = offset.data

        if len(self.offset) != temp_type.get_num_dims():
            raise VerifyException(
                f"Expected offset's rank to be {temp_type.get_num_dims()} to match the "
                f"operand's rank, got {len(self.offset)}"
            )


@irdl_op_definition
class LoadOp(IRDLOperation):
    """
    This operation takes a field and returns its values.

    Example:
      %0 = stencil.load %field : (!stencil.field<70x70x60xf64>) -> !stencil.temp<?x?x?xf64>
    """

    name = "stencil.load"
    field: Operand = operand_def(FieldType)
    res: OpResult = result_def(TempType)

    @staticmethod
    def get(
        field: SSAValue | Operation,
        lb: IndexAttr | None = None,
        ub: IndexAttr | None = None,
    ):
        field_type = SSAValue.get(field).type
        assert isa(field_type, FieldType[Attribute])

        if lb is None or ub is None:
            res_type = TempType(field_type.get_num_dims(), field_type.element_type)
        else:
            res_type = TempType(zip(lb, ub), field_type.element_type)

        return LoadOp.build(
            operands=[field],
            result_types=[res_type],
        )

    def verify_(self) -> None:
        for use in self.field.uses:
            if isa(use.operation, StoreOp):
                raise VerifyException("Cannot Load and Store the same field!")
        field = self.field.type
        temp = self.res.type
        assert isa(field, FieldType[Attribute])
        assert isa(temp, TempType[Attribute])
        if isinstance(field.bounds, StencilBoundsAttr) and isinstance(
            temp.bounds, StencilBoundsAttr
        ):
            if temp.bounds.lb < field.bounds.lb or temp.bounds.ub > field.bounds.ub:
                raise VerifyException(
                    "The stencil.load is too big for the loaded field."
                )


@irdl_op_definition
class BufferOp(IRDLOperation):
    """
    Prevents fusion of consecutive stencil.apply operations.

    Example:
      %0 = stencil.buffer %buffered : (!stencil.temp<?x?x?xf64>) -> !stencil.temp<?x?x?xf64>
    """

    name = "stencil.buffer"
    temp: Operand = operand_def(TempType)
    res: OpResult = result_def(TempType)

    def __init__(self: IRDLOperation, temp: SSAValue | Operation):
        temp = SSAValue.get(temp)
        super().__init__(operands=[temp], result_types=[temp.type])

    def verify_(self) -> None:
        if self.temp.type != self.res.type:
            raise VerifyException(
                f"Expected operand and result type to be equal, got ({self.temp.type}) "
                f"-> {self.res.type}"
            )
        if not isinstance(self.temp.owner, ApplyOp):
            raise VerifyException(
                f"Expected stencil.buffer to buffer a stencil.apply's output, got "
                f"{self.temp.owner}"
            )
        if any(not isinstance(use.operation, BufferOp) for use in self.temp.uses):
            raise VerifyException(
                "A stencil.buffer's operand temp should only be buffered. You can use "
                "stencil.buffer's output instead!"
            )


@irdl_op_definition
class StoreOp(IRDLOperation):
    """
    This operation writes values to a field on a user defined range.

    Example:
      stencil.store %temp to %field ([0,0,0] : [64,64,60]) : !stencil.temp<?x?x?xf64> to !stencil.field<70x70x60xf64>  # noqa
    """

    name = "stencil.store"
    temp: Operand = operand_def(TempType)
    field: Operand = operand_def(FieldType)
    lb: IndexAttr = attr_def(IndexAttr)
    ub: IndexAttr = attr_def(IndexAttr)

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
            if isa(use.operation, LoadOp) and use.operation is not self:
                raise VerifyException("Can only store once to a field!")


@irdl_op_definition
class StoreResultOp(IRDLOperation):
    """
    The store_result operation either stores an operand value or nothing.

    Examples:
      stencil.store_result %0 : !stencil.result<f64>
      stencil.store_result : !stencil.result<f64>
    """

    name = "stencil.store_result"
    args: VarOperand = var_operand_def(Attribute)
    res: OpResult = result_def(ResultType)


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
    arg: VarOperand = var_operand_def(ResultType | AnyFloat)

    traits = frozenset([HasParent(ApplyOp), IsTerminator()])

    @staticmethod
    def get(res: Sequence[SSAValue | Operation]):
        return ReturnOp.build(operands=[list(res)])

    def verify_(self) -> None:
        types = [
            o.type.elem if isinstance(o.type, ResultType) else o.type for o in self.arg
        ]
        apply = cast(ApplyOp, self.parent_op())
        res_types = [cast(TempType[Attribute], r.type).element_type for r in apply.res]
        if len(types) != len(res_types):
            raise VerifyException(
                f"stencil.return expected {len(res_types)} operands to match the parent "
                f"stencil.apply result types, got {len(types)}"
            )
        if types != res_types:
            raise VerifyException(
                "stencil.return expected operand types to match the parent "
                "stencil.apply result element types."
            )


Stencil = Dialect(
    [
        CastOp,
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
