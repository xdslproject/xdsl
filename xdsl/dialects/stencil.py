from __future__ import annotations

from typing import Annotated, TypeVar
from xdsl.dialects.builtin import IntAttr

from xdsl.dialects.experimental.stencil import FieldType, StencilBoundsAttr
from xdsl.ir import OpResult, SSAValue, Operation, Attribute, Dialect
from xdsl.irdl import irdl_op_definition, IRDLOperation, Operand
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa


_FieldTypeVar = TypeVar("_FieldTypeVar", bound=Attribute)


@irdl_op_definition
class CastOp(IRDLOperation):
    """
    This operation casts dynamically shaped input fields to statically shaped fields.

    Example:
        %0 = stencil.cast %in ([-3, -3, 0] : [67, 67, 60]) : (!stencil.field<?x?x?xf64>) -> !stencil.field<70x70x60xf64> # noqa
    """

    name = "stencil.cast"
    field: Annotated[Operand, FieldType]
    result: Annotated[OpResult, FieldType]

    @staticmethod
    def get(
        field: SSAValue | Operation,
        bounds: StencilBoundsAttr,
        res_type: FieldType[_FieldTypeVar] | FieldType[Attribute] | None = None,
    ) -> CastOp:
        """ """
        field_ssa = SSAValue.get(field)
        assert isa(field_ssa.typ, FieldType[Attribute])
        if res_type is None:
            res_type = FieldType(
                bounds,
                field_ssa.typ.element_type,
            )
        return CastOp.build(
            operands=[field],
            result_types=[res_type],
        )

    def verify_(self) -> None:
        # this should be fine, verify() already checks them:
        assert isa(self.field.typ, FieldType[Attribute])
        assert isa(self.result.typ, FieldType[Attribute])

        if isinstance(self.result.typ.bounds, IntAttr):
            raise VerifyException("Output type's size must be explicit")

        if self.field.typ.element_type != self.result.typ.element_type:
            raise VerifyException(
                "Input and output fields must have the same element types"
            )

        if self.field.typ.get_num_dims() != self.result.typ.get_num_dims():
            raise VerifyException("Input and output types must have the same rank")

        if (
            isinstance(self.field.typ.bounds, StencilBoundsAttr)
            and self.field.typ.bounds != self.result.typ.bounds
        ):
            raise VerifyException(
                "If input shape is not dynamic, it must be the same as output"
            )


Stencil = Dialect([CastOp], [])
