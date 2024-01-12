from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
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


def extract_shape_from_type(
    shape_type: list[int],
) -> list[int] | None:
    if shape_type is None:
        return None
    elif isinstance(shape_type, TensorType):
        return list(shape_type.get_shape())
    else:
        return list(shape_type)


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
    lhs_shape = extract_shape_from_type(lhs)
    rhs_shape = extract_shape_from_type(rhs)
    if lhs_shape == rhs_shape:
        return lhs_shape

    # Check if Tensor A and B have the same number of dimensions and the length of each dimensions is either a common
    # length or B's length is 1.
    res_shape: list[int] = []
    if len(lhs_shape) == len(rhs_shape):
        for d1, d2 in zip(lhs_shape, rhs_shape):
            if d1 == d2 or d2 == 1:
                res_shape.append(max(d1, d2))

    # If Tensor B has too few dimensions, and B can have its shapes prepended with a dimension of length 1 to satisfy
    # property 2.
    if len(rhs_shape) < len(lhs_shape):
        # Store difference in dimension of shapes
        shape_dimension_diffs = len(lhs_shape) - len(rhs_shape)
        prepend = [1] * shape_dimension_diffs
        prepend_b_shape = prepend + rhs_shape
        for d1, d2 in zip(lhs_shape, prepend_b_shape):
            if d1 == d2 or d2 == 1:
                res_shape.append(max(d1, d2))
            else:
                raise VerifyException(
                    f"operands have incompatible shapes: {tuple(lhs_shape)} and {tuple(rhs_shape)}"
                )
    return res_shape


def multidirectional_broadcast_shape(
    lhs: Sequence[int], rhs: Sequence[int]
) -> list[int] | None:
    """
    In ONNX, a set of tensors are multidirectional broadcastable to the same shape if one of the following is true:

    1.The tensors all have exactly the same shape.
    2.The tensors all have the same number of dimensions and the length of each dimensions is either a common length or 1.
    3.The tensors that have too few dimensions can have their shapes prepended with a dimension of length 1 to satisfy property 2.

    https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    """
    lhs_shape = extract_shape_from_type(lhs)
    rhs_shape = extract_shape_from_type(rhs)
    if len(lhs_shape) > len(rhs_shape):
        longer_shape, shorter_shape = lhs_shape, rhs_shape
    else:
        longer_shape, shorter_shape = rhs_shape, lhs_shape
    # Store difference in dimension of shapes
    shape_dimension_diffs = len(longer_shape) - len(shorter_shape)
    if len(lhs_shape) != len(rhs_shape):
        shorter_shape = [1] * shape_dimension_diffs + list(shorter_shape)
    res_shape: list[int] = []
    # Checking shape broadcasting compatibility
    for d1, d2 in zip(longer_shape, shorter_shape):
        if d1 == d2 or d1 == 1 or d2 == 1:
            res_shape.append(max(d1, d2))
        else:
            raise VerifyException(
                f"operands have incompatible shapes: {tuple(shorter_shape)} and {tuple(longer_shape)}"
            )
    return res_shape


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
        res_shape = multidirectional_broadcast_shape(lhs_type, rhs_type)
        res_type_shape = list(res_type.get_shape())
        if len(res_shape) != len(res_type_shape) or res_shape != res_type_shape:
            raise VerifyException(
                f"result shape {res_shape} does not match result type {res_type}"
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

        tensor_c_type = cast(TensorType[Attribute], tensor_c_type)
        final_res_shape = unidirectional_broadcast_shape(res_shape, tensor_c_type)
        res_type_shape = list(res_tensor_type.get_shape())
        if (
            len(final_res_shape) != len(res_type_shape)
            or final_res_shape != res_type_shape
        ):
            raise VerifyException(
                f"result shape {final_res_shape} does not match final result type {res_tensor_type}"
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
