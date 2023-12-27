from __future__ import annotations

from abc import ABC

from xdsl.dialects.builtin import AnyTensorType, SSAValue
from xdsl.ir import (
    Attribute,
    Dialect,
    OpResult,
)
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


class ElementwiseBinOpBase(IRDLOperation, ABC):
    """Base class for element-wise binary operations on tensors with Numpy-style broadcasting."""

    lhs = operand_def()
    rhs = operand_def()
    res: OpResult = result_def(AnyTensorType)
    assembly_format = "`(` $lhs `,` $rhs `)` attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)"

    def __init__(
        self,
        lhs: SSAValue,
        rhs: SSAValue,
        attributes: dict[str, Attribute] = {},
    ):
        super().__init__(
            operands=[lhs, rhs],
            attributes=attributes,
            result_types=[[]],
        )

    def verify_(self) -> None:
        # Check that the arguments are broadcastable (using Numpy semantics) and that the result type is correct.
        res_shape: list[int] = []
        # Iterate over the shapes in reverse order and compute the result shape.
        lhs_shape: list[int] = [x.data for x in self.lhs.type.shape]
        rhs_shape: list[int] = [x.data for x in self.rhs.type.shape]
        i = max(len(lhs_shape), len(rhs_shape))
        while i > 0:
            i -= 1
            d1 = lhs_shape[i] if i >= 0 else 1
            d2 = rhs_shape[i] if i >= 0 else 1
            if d1 == d2:
                res_shape.append(d1)
                continue
            if d1 == 1:
                res_shape.append(d2)
                continue
            if d2 == 1:
                res_shape.append(d1)
                continue
            raise VerifyException(
                f"operands have incompatible shapes: {lhs_shape} and {rhs_shape}"
            )
        # Reverse the result shape and check that it matches the result type.
        res_shape.reverse()
        if not all(
            x == y for x, y in zip(res_shape, [v.data for v in self.res.type.shape])
        ):
            raise VerifyException(
                f"result shape {res_shape} does not match result type {self.res.type}"
            )


@irdl_op_definition
class Add(ElementwiseBinOpBase):
    name = "onnx.Add"


@irdl_op_definition
class Sub(ElementwiseBinOpBase):
    name = "onnx.Sub"


@irdl_op_definition
class Mul(ElementwiseBinOpBase):
    name = "onnx.Mul"


@irdl_op_definition
class Div(ElementwiseBinOpBase):
    name = "onnx.Div"


ONNX = Dialect("onnx", [Add, Sub, Mul, Div])
