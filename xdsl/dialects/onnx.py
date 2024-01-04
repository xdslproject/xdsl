from __future__ import annotations

from abc import ABC
from typing import Annotated

from xdsl.dialects.builtin import (
    AnyFloat,
    IntegerType,
    SSAValue,
    TensorType,
)
from xdsl.ir import (
    Attribute,
    Dialect,
)
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


class ElementwiseBinOpBase(IRDLOperation, ABC):
    """Base class for element-wise binary operations on tensors with Numpy-style broadcasting."""

    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    lhs = operand_def(TensorType[T])
    rhs = operand_def(TensorType[T])
    res = result_def(TensorType[T])
    assembly_format = "`(` $lhs `,` $rhs `)` attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($res)"

    def __init__(self, lhs: SSAValue, rhs: SSAValue, res_type: Attribute):
        super().__init__(
            operands=[lhs, rhs],
            result_types=[res_type],
        )

    def verify_(self) -> None:
        # Check that the arguments are broadcastable (using Numpy semantics) and that the result type is correct.
        res_shape: list[int] = []

        if (
            not isinstance(lhs_type := self.lhs.type, TensorType)
            or not isinstance(rhs_type := self.rhs.type, TensorType)
            or not isinstance(res_type := self.res.type, TensorType)
        ):
            assert (
                False
            ), "onnx elementwise binary operation operands and result must be of type TensorType"

        # Iterate over the shapes in reverse order and compute the result shape.
        lhs_shape = lhs_type.get_shape()
        rhs_shape = rhs_type.get_shape()
        i = max(len(lhs_shape), len(rhs_shape))
        while i > 0:
            i -= 1
            d1: int = lhs_shape[i] if i >= 0 else 1
            d2: int = rhs_shape[i] if i >= 0 else 1
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
        res_type_shape = list(res_type.get_shape())
        res_shape.reverse()
        if len(res_shape) != len(res_type_shape) or res_shape != res_type_shape:
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




@irdl_op_definition
class Relu(IRDLOperation):
    name = "onnx.Relu"
    """
    Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function,
     y = max(0, x), is applied to the tensor elementwise.
    """
    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    operand = operand_def(TensorType[T])
    res = result_def(TensorType[T])
    assembly_format = "`(` $operand`)` attr-dict `:` `(` type($operand) `)` `->` type($res)"

    def __init__(self, operand: SSAValue):
        super().__init__(
            operands=[operand],
            result_types=[operand.type],
        )

    def verify_(self) -> None:
        if (
                not isinstance(operand_type := self.operand.type, TensorType)
                or not isinstance(res_type := self.res.type, TensorType)
        ):
            assert (
                False
            ), "onnx elementwise operation operand and result must be of type TensorType"
            operand_type = cast(TensorType[Attribute], operand_type)
            res_type = cast(TensorType[Attribute], res_type)
            
            if operand_type != res_type:
                raise VerificationException("Mismatch between operand type and res type")

ONNX = Dialect(
    "onnx",
    [
        Add,
        Sub,
        Mul,
        Div,
        Relu,
    ],
)
