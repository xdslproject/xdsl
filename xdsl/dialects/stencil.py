from __future__ import annotations
from typing import (Annotated, TypeVar, Generic, List, Sequence, cast, Iterable, Iterator)

from xdsl.dialects.builtin import (IntegerType, IntegerAttr, ArrayAttr, ParametrizedAttribute, ParameterDef, TypeAttribute, AnyIntegerAttr)
from xdsl.ir import OpResult, SSAValue, Operation, Attribute, Dialect
from xdsl.irdl import irdl_op_definition, IRDLOperation, Operand, OpAttr, irdl_attr_definition
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


_FieldTypeVar = TypeVar("_FieldTypeVar", bound=Attribute)


@irdl_attr_definition
class FieldType(Generic[_FieldTypeVar], ParametrizedAttribute, TypeAttribute):
    name = "stencil.field"

    shape: ParameterDef[ArrayAttr[AnyIntegerAttr]]
    element_type: ParameterDef[_FieldTypeVar]

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
        typ: _FieldTypeVar,
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


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
        %0 = stencil.cast %in ([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64> # noqa
    """

    name: str = "stencil.cast"
    field: Annotated[Operand, FieldType]
    lb: OpAttr[IndexAttr]
    ub: OpAttr[IndexAttr]
    result: Annotated[OpResult, FieldType]

    @staticmethod
    def get(
        field: SSAValue | Operation,
        lb: IndexAttr,
        ub: IndexAttr,
        res_type: FieldType[_FieldTypeVar] | FieldType[Attribute] | None = None,
    ) -> CastOp:
        """ """
        field_ssa = SSAValue.get(field)
        assert isa(field_ssa.typ, FieldType[Attribute])
        if res_type is None:
            res_type = FieldType(
                tuple(ub_elm - lb_elm for lb_elm, ub_elm in zip(lb, ub)),
                field_ssa.typ.element_type,
            )
        return CastOp.build(
            operands=[field],
            attributes={"lb": lb, "ub": ub},
            result_types=[res_type],
        )

    def verify_(self) -> None:
        # this should be fine, verify() already checks them:
        assert isa(self.field.typ, FieldType[Attribute])
        assert isa(self.result.typ, FieldType[Attribute])

        if self.field.typ.element_type != self.result.typ.element_type:
            raise VerifyException(
                "Input and output fields have different element types"
            )

        if not len(self.lb) == len(self.ub):
            raise VerifyException("lb and ub must have the same dimensions")

        if not len(self.field.typ.shape) == len(self.lb):
            raise VerifyException("Input type and bounds must have the same dimensions")

        if not len(self.result.typ.shape) == len(self.ub):
            raise VerifyException(
                "Result type and bounds must have the same dimensions"
            )

        for i, (in_attr, lb, ub, out_attr) in enumerate(
            zip(
                self.field.typ.shape,
                self.lb,
                self.ub,
                self.result.typ.shape,
            )
        ):
            in_: int = in_attr.value.data
            out: int = out_attr.value.data

            if ub - lb != out:
                raise VerifyException(
                    "Bound math doesn't check out in dimensions {}! {} - {} != {}".format(
                        i, ub, lb, out
                    )
                )

            if in_ != -1:
                # TODO: find out if this is too strict
                raise VerifyException("Input must be dynamically shaped")


Stencil = Dialect([CastOp], [FieldType, IndexAttr])
