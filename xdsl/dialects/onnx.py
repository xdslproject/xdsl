from __future__ import annotations

from abc import ABC
from typing import Annotated, cast

from xdsl.dialects.builtin import (
    AnyFloat,
    FloatAttr,
    IntegerAttr,
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
    opt_attr_def,
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
    assembly_format = (
        "`(` $operand`)` attr-dict `:` `(` type($operand) `)` `->` type($res)"
    )

    def __init__(self, operand: SSAValue):
        super().__init__(
            operands=[operand],
            result_types=[operand.type],
        )

    def verify_(self) -> None:
        if not isinstance(
            operand_type := self.operand.type, TensorType
        ) or not isinstance(res_type := self.res.type, TensorType):
            assert (
                False
            ), "onnx elementwise operation operand and result must be of type TensorType"
            operand_type = cast(TensorType[Attribute], operand_type)
            res_type = cast(TensorType[Attribute], res_type)

            if operand_type != res_type:
                raise VerifyException("Mismatch between operand type and res type")


@irdl_op_definition
class Gemm(IRDLOperation):
    name = "onnx.Gemm"
    """
    General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
    A' = transpose(A) if transA else A
    B' = transpose(B) if transB else B
    Compute Y = alpha * A' * B' + beta * C,
    where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K),
    input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N).
    """
    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    tensor_a = operand_def(TensorType[T])
    tensor_b = operand_def(TensorType[T])
    tensor_c = operand_def(TensorType[T])

    alpha = opt_attr_def(FloatAttr)
    beta = opt_attr_def(FloatAttr)

    trans_a = opt_attr_def(IntegerAttr, attrname="transA")
    trans_b = opt_attr_def(IntegerAttr, attrname="transB")

    res_tensor = result_def(TensorType[T])
    assembly_format = (
        "`(` $tensor_a `,` $tensor_b `,`$tensor_c`)` attr-dict `:` `(` type($tensor_a) `,"
        "` type($tensor_b) `,`type($tensor_c)`)` `->` type($res_tensor) "
    )

    def __init__(
        self,
        tensor_a: SSAValue,
        tensor_b: SSAValue,
        tensor_c: SSAValue,
        alpha: Attribute,
        trans_a: Attribute,
        trans_b: Attribute,
        beta: Attribute,
    ):
        super().__init__(
            attributes={
                "transA": trans_a,
                "transB": trans_b,
                "alpha": alpha,
                "beta": beta,
            },
            operands=[tensor_a, tensor_b, tensor_c],
            result_types=[tensor_a.type],
        )

    def verify(self) -> None:
        # store dimensions of tensor A and tensor B
        res_shape: list[int] = []
        if (
            not isinstance(tensor_a_type := self.tensor_a.type, TensorType)
            or not isinstance(tensor_b_type := self.tensor_b.type, TensorType)
            or not isinstance(tensor_c_type := self.tensor_c.type, TensorType)
            or not isinstance(res_tensor_type := self.res_tensor.type, TensorType)
        ):
            assert (
                False
            ), "onnx elementwise operation operands and result must be of type TensorType"

        # check shape compatibility
        tensor_a_shape = tensor_a_type.get_shape()
        tensor_b_shape = tensor_b_type.get_shape()

        if tensor_a_type.get_num_dims() != 2:
            raise VerifyException("tensor A should be a 2D tensor")

        if tensor_b_type.get_num_dims() != 2:
            raise VerifyException("tensor B should be a 2D tensor")

        if self.trans_a is not None:
            list(tensor_a_shape).reverse()

        if self.trans_b is not None:
            list(tensor_b_shape).reverse()

        if self.beta is not None:
            c_dims = tensor_c_type.get_num_dims()
            if c_dims > 2:
                raise VerifyException("tensor C should be a 1D tensor or 2D tensor")
        if tensor_a_shape[1] != tensor_b_shape[0]:
            raise VerifyException(
                f"operands have incompatible shapes: {tensor_a_shape} and {tensor_b_shape}"
            )
        else:
            res_shape.append(tensor_a_shape[0])
            res_shape.append(tensor_b_shape[1])

        # Now check that tensor C is unidirectional broadcastable to tensor (A * B) (using Numpy semantics) and that
        # the result type is correct.
        # Iterate over the shapes in reverse order and compute the result shape.
        final_res_shape: list[int] = []
        res_shape = tuple(res_shape)
        tensor_c_shape = tensor_c_type.get_shape()
        i = max(len(res_shape), len(tensor_c_shape))
        while i > 0:
            i -= 1
            d1: int = res_shape[i] if i >= 0 else 1
            d2: int = tensor_c_shape[i] if i >= 0 else 1
            if d1 == d2 or d2 == 1 or d1 == 1:
                final_res_shape.append(max(d1, d2))
                continue
            raise VerifyException(
                f"operands have incompatible shapes: {res_shape} and {tensor_c_shape}"
            )
        res_tensor_type_shape = list(res_tensor_type.get_shape())
        final_res_shape.reverse()
        if (
            len(final_res_shape) != len(res_tensor_type_shape)
            or final_res_shape != res_tensor_type_shape
        ):
            raise VerifyException(
                f"result shape {final_res_shape} does not match result type {self.res_tensor.type}"
            )


ONNX = Dialect(
    "onnx",
    [
        Add,
        Sub,
        Mul,
        Div,
        Relu,
        Gemm,
    ],
)
