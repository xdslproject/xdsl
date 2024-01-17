from __future__ import annotations

from abc import ABC
from typing import Annotated, cast

from xdsl.dialects.builtin import (
    AnyFloat,
    AnyIntegerAttr,
    FloatAttr,
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


def verify_unidirectional_broadcast_shape(
    lhs: TensorType[Attribute], rhs: TensorType[Attribute], res: TensorType[Attribute]
) -> None:
    """
    Returns a unidirectional broadcastable shape
    """
    lhs_shape = lhs.get_shape()
    rhs_shape = rhs.get_shape()
    expected_shape = unidirectional_broadcast_shape(list(lhs_shape), list(rhs_shape))
    if expected_shape is None:
        raise VerifyException(
            f"operands have incompatible shapes: {lhs_shape} and {rhs_shape}"
        )
    res_type_shape = res.get_shape()
    if (
        len(expected_shape) != len(res_type_shape)
        or tuple(expected_shape) != res_type_shape
    ):
        raise VerifyException(
            f"result shape {expected_shape} does not match result type {res}"
        )


def verify_multidirectional_broadcast_shape(
    lhs: TensorType[Attribute], rhs: TensorType[Attribute], res: TensorType[Attribute]
) -> None:
    """
    Returns a multidirectional broadcastable shape
    """
    lhs_shape = lhs.get_shape()
    rhs_shape = rhs.get_shape()
    expected_shape = multidirectional_broadcast_shape(list(lhs_shape), list(rhs_shape))
    if expected_shape is None:
        raise VerifyException(
            f"operands have incompatible shapes: {lhs_shape} and {rhs_shape}"
        )
    res_type_shape = res.get_shape()
    if (
        len(expected_shape) != len(res_type_shape)
        or tuple(expected_shape) != res_type_shape
    ):
        raise VerifyException(
            f"result shape {expected_shape} does not match result type {res}"
        )


def unidirectional_broadcast_shape(lhs: list[int], rhs: list[int]) -> list[int] | None:
    """
    In ONNX, tensor B is unidirectional broadcastable to tensor A if one of the following is true:

    1. Tensor A and B both have exactly the same shape.
    2. Tensor A and B all have the same number of dimensions and
    the length of each dimensions is either a common length or B's length is 1.

    3.Tensor B has too few dimensions, and B can have its shapes prepended with a dimension of length 1 to satisfy
    property 2.

    When unidirectional broadcasting happens, the output's shape is the same as the shape of A (i.e.,
    the larger shape of two input tensors)

    https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    """
    # Check if Tensor A and B both have exactly the same shape
    if lhs == rhs:
        return lhs

    lhs_len = len(lhs)
    rhs_len = len(rhs)
    prefix_len = lhs_len - rhs_len
    if prefix_len < 0:
        # lhs must not be shorter than rhs
        return None
    res_shape = lhs[:prefix_len]
    for dl, dr in zip(lhs[prefix_len:], rhs):
        if dl == dr or dr == 1 or dl == 1:
            res_shape.append(max(dl, dr))
        else:
            return None
    return res_shape


def multidirectional_broadcast_shape(
    lhs: list[int], rhs: list[int]
) -> list[int] | None:
    """
    In ONNX, a set of tensors are multidirectional broadcastable to the same shape if one of the following is true:

    1.The tensors all have exactly the same shape.
    2.The tensors all have the same number of dimensions and the length of each dimensions is either a common length or 1.
    3.The tensors that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2.

    https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    """

    if len(lhs) > len(rhs):
        return unidirectional_broadcast_shape(rhs, lhs)
    else:
        return unidirectional_broadcast_shape(lhs, rhs)


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
        if (
            not isinstance(lhs_type := self.lhs.type, TensorType)
            or not isinstance(rhs_type := self.rhs.type, TensorType)
            or not isinstance(res_type := self.res.type, TensorType)
        ):
            assert (
                False
            ), "onnx elementwise binary operation operands and result must be of type TensorType"
        lhs_type = cast(TensorType[Attribute], lhs_type)
        rhs_type = cast(TensorType[Attribute], rhs_type)
        res_type = cast(TensorType[Attribute], res_type)
        verify_multidirectional_broadcast_shape(lhs_type, rhs_type, res_type)


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
    """
    Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function,
     y = max(0, x), is applied to the tensor elementwise.
    """

    name = "onnx.Relu"
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
        assert isinstance(operand_type := self.operand.type, TensorType)
        assert isinstance(res_type := self.res.type, TensorType)
        operand_type = cast(TensorType[Attribute], operand_type)
        res_type = cast(TensorType[Attribute], res_type)

        if operand_type != res_type:
            raise VerifyException(
                "Mismatch between operand type and res type of onnx.Relu"
            )


@irdl_op_definition
class Gemm(IRDLOperation):
    """
    General Matrix multiplication: https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3
    A' = transpose(A) if transA else A
    B' = transpose(B) if transB else B
    Compute Y = alpha * A' * B' + beta * C,
    where input tensor A has shape (M, K) or (K, M), input tensor B has shape (K, N) or (N, K),
    input tensor C is broadcastable to shape (M, N), and output tensor Y has shape (M, N).
    """

    name = "onnx.Gemm"
    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    tensor_a = operand_def(TensorType[T])
    tensor_b = operand_def(TensorType[T])
    tensor_c = operand_def(TensorType[T])

    alpha = opt_attr_def(FloatAttr[AnyFloat])
    beta = opt_attr_def(FloatAttr[AnyFloat])

    trans_a = opt_attr_def(AnyIntegerAttr, attr_name="transA")
    trans_b = opt_attr_def(AnyIntegerAttr, attr_name="transB")

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
            result_types=[tensor_c.type],
        )

    def verify_(self) -> None:
        if (
            not isinstance(tensor_a_type := self.tensor_a.type, TensorType)
            or not isinstance(tensor_b_type := self.tensor_b.type, TensorType)
            or not isinstance(tensor_c_type := self.tensor_c.type, TensorType)
            or not isinstance(res_tensor_type := self.res_tensor.type, TensorType)
        ):
            assert (
                False
            ), "onnx elementwise operation operands and result must be of type TensorType"

        tensor_a_type = cast(TensorType[Attribute], tensor_a_type)
        tensor_b_type = cast(TensorType[Attribute], tensor_b_type)
        tensor_c_type = cast(TensorType[Attribute], tensor_c_type)
        res_tensor_type = cast(TensorType[Attribute], res_tensor_type)

        # store dimensions of tensor A and tensor B
        res_shape: list[int] = []
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

        # Build tensor of tensor (A * B) computation
        tensors_res = TensorType(IntegerType(32), res_shape)
        verify_unidirectional_broadcast_shape(
            tensors_res, tensor_c_type, res_tensor_type
        )


ONNX = Dialect(
    "onnx",
    [
        Add,
        Div,
        Gemm,
        Mul,
        Relu,
        Sub,
    ],
)
