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


class ShapeBroadcastVerifier:
    """
    https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
    """

    @staticmethod
    def verify_shapes(
        lhs_type: TensorType[Attribute] | list[int],
        rhs_type: TensorType[Attribute] | list[int],
        res_type: TensorType[Attribute],
    ) -> None:
        # Check that the arguments are broadcastable (using Numpy semantics) and that the result type is correct.
        res_shape: list[int] = []
        if isinstance(lhs_type, TensorType):
            lhs_shape = lhs_type.get_shape()
        else:
            lhs_shape = lhs_type

        if isinstance(rhs_type, TensorType):
            rhs_shape = rhs_type.get_shape()
        else:
            rhs_shape = rhs_type
        # Iterate over the shapes in reverse order and compute the result shape.
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
                f"result shape {res_shape} does not match result type {res_type}"
            )


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
        ShapeBroadcastVerifier.verify_shapes(lhs_type, rhs_type, res_type)


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
        res_tensor_type = cast(TensorType[Attribute], res_tensor_type)
        ShapeBroadcastVerifier.verify_shapes(res_shape, tensor_c_type, res_tensor_type)


@irdl_op_definition
class Reshape(IRDLOperation):
    """
    Reshape the input tensor similar to numpy.reshape.
    First input is the data tensor, second input is a shape tensor which specifies the output shape. It outputs the reshaped tensor.
    At most one dimension of the new shape can be -1. In this case, the value is
    inferred from the size of the tensor and the remaining dimensions. A dimension
    could also be 0, in which case the actual dimension value is unchanged (i.e. taken
    from the input tensor). Shape (second input) could be an empty shape, which means converting to a scalar.
    The input tensor's shape and the output tensor's shape are required to have the same number of elements.

     Attributes:
        - allowzero  int (default is 0):  By default, when any value in the 'shape' input is equal to zero
     the corresponding dimension value is copied from the input tensor dynamically. allowzero=1 indicates that if any
     value in the 'shape' input is set to zero, the zero value is honored, similar to NumPy.

    """

    name = "onnx.Reshape"
    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    data = operand_def(TensorType[T])
    shape = operand_def(TensorType)
    reshaped = result_def(TensorType[T])

    allow_zero = opt_attr_def(AnyIntegerAttr, attr_name="allowzero")

    assembly_format = "`(` $data `,` $shape `)` attr-dict `:` `(` type($data) `,` type($shape) `)` `->` type($reshaped)"

    def __init__(self, data: SSAValue, shape: SSAValue, allow_zero: Attribute):
        super().__init__(
            attributes={"allowzero": allow_zero},
            operands=[data, shape],
            result_types=[data.type],
        )

    def verify_(self) -> None:
        if (
            not isinstance(data_type := self.data.type, TensorType)
            or not isinstance(shape_type := self.shape.type, TensorType)
            or not isinstance(reshaped_type := self.reshaped.type, TensorType)
        ):
            assert (
                False
            ), "onnx elementwise operation operands and result must be of type TensorType"

        data_type = cast(TensorType[Attribute], data_type)
        reshaped_type = cast(TensorType[Attribute], reshaped_type)
        shape_type = cast(TensorType[Attribute], shape_type)

        if shape_type.element_type != IntegerType(64):
            raise VerifyException(
                "shape element type has to be a 64-bit signless integer"
            )

        data_type = data_type.get_shape()
        shape_type = shape_type.get_shape()
        shape_type_list = list(shape_type)
        reshaped_type = reshaped_type.get_shape()

        # The input tensor's shape and the output tensor's shape are required to have the same number of elements.
        if len(data_type) != len(reshaped_type):
            raise VerifyException(
                "Input tensor's shape and output tensor's shape must have the same number of elements"
            )

        # There is currently only support for rank one shape tensors in onnx-mlir
        if len(shape_type_list) != 1:
            raise VerifyException("Shape tensor must be 1D")

        # At most one dimension of the new shape can be -1.
        # In this case, the value is inferred from the size of the tensor and the remaining dimensions.
        count_minus_one = shape_type_list.count(-1)
        if count_minus_one == 1:
            index_of_minus_one = shape_type_list.index(-1)
            specified_dim = len(shape_type_list)
            total_elements = len(data_type)
            missing_dim = total_elements // specified_dim
            shape_type_list[index_of_minus_one] = missing_dim
        # Handle case where dimension is zero
        for i, dim in enumerate(shape_type_list):
            if dim == 0:
                if self.allow_zero:
                    # If allow_zero is set, explicitly set the dimension to zero  (i.e. not taken from input tensor)
                    shape_type_list[i] = 0
                else:
                    # dimension is 0, leave it unchanged  (i.e. taken from the input tensor).
                    shape_type_list[i] = data_type[i]
        # Shape (second input) could be an empty shape, which means converting to a scalar.
        if not shape_type_list:
            shape_type_list = [IntegerType(64)]


ONNX = Dialect(
    "onnx",
    [
        Add,
        Div,
        Gemm,
        Mul,
        Relu,
        Reshape,
        Sub,
    ],
)
