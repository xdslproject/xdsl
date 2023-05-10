from __future__ import annotations

from typing import Annotated, TypeVar

from xdsl.dialects.experimental.stencil import FieldType, IndexAttr
from xdsl.ir import OpResult, SSAValue, Operation, Attribute, Dialect
from xdsl.irdl import irdl_op_definition, IRDLOperation, Operand, OpAttr
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

            if in_ != -1 and in_ != out:
                raise VerifyException(
                    "If input shape is not dynamic, it must be the same as output"
                )


Stencil = Dialect([CastOp], [])
