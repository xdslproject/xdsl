from __future__ import annotations

import math
from abc import ABC
from typing import Annotated, cast

from typing_extensions import Self

from xdsl.dialects.builtin import (
    Any,
    AnyFloat,
    AnyIntegerAttr,
    AnyTensorType,
    ArrayAttr,
    DenseIntOrFPElementsAttr,
    Float32Type,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    MemRefType,
    NoneType,
    SSAValue,
    StringAttr,
    SymbolRefAttr,
    TensorType,
)
from xdsl.ir import (
    Attribute,
    Dialect,
)
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    attr_def,
    base,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
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
            assert False, "onnx elementwise binary operation operands and result must be of type TensorType"
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
            assert False, "onnx elementwise operation operands and result must be of type TensorType"

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

        if self.trans_a is not None and self.trans_a.value.data == 1:
            tensor_a_shape = tuple(reversed(tensor_a_shape))

        if self.trans_b is not None and self.trans_b.value.data == 1:
            tensor_b_shape = tuple(reversed(tensor_b_shape))

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
        tensors_res = TensorType(tensor_a_type.element_type, res_shape)
        verify_unidirectional_broadcast_shape(
            tensors_res, tensor_c_type, res_tensor_type
        )


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
     value in the 'shape' input is set to zero, the zero value is honoured, similar to NumPy.

    """

    name = "onnx.Reshape"
    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    data = operand_def(TensorType[T])
    shape = operand_def(TensorType[IntegerType])
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
            assert False, "onnx elementwise operation operands and result must be of type TensorType"

        data_type = cast(TensorType[Attribute], data_type)
        reshaped_type = cast(TensorType[Attribute], reshaped_type)
        shape_type = cast(TensorType[Attribute], shape_type)

        if shape_type.element_type != IntegerType(64):
            raise VerifyException(
                "shape element type has to be a 64-bit signless integer"
            )

        data_type = data_type.get_shape()
        shape_type = shape_type.get_shape()
        reshaped_type = reshaped_type.get_shape()

        # Shape tensor rank can't be -1.
        if shape_type[0] == -1:
            raise VerifyException("Shape tensor rank must not be equal to -1")

        # There is currently only support for rank one shape tensors in onnx-mlir
        # Shape tensor must have a constant shape.
        if len(shape_type) != 1:
            raise VerifyException("Shape tensor must have a rank one")

        # The input tensor's shape and the output tensor's shape are required to have the same number of elements.
        if math.prod(data_type) != math.prod(reshaped_type):
            raise VerifyException(
                "Input tensor's shape and output tensor's shape must have the same number of elements"
            )


@irdl_op_definition
class Abs(IRDLOperation):
    """
    Absolute takes one input data (Tensor) and produces one output data (Tensor) where absolute value,
    y = abs(x), is applied to the tensor elementwise.
    """

    name = "onnx.Abs"
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
                "Mismatch between operand type and res type of onnx.Abs"
            )


@irdl_op_definition
class Conv(IRDLOperation):
    """
    The convolution operator consumes an input tensor and a filter, and computes the output.

    Attributes:

    - auto_pad  string (default is NOTSET): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or
    VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the
    input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between
    the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd
    number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.

    -dilations list of ints: dilation value along each spatial axis of the filter. If not present, the dilation
    defaults is 1 along each spatial axis.

    -group int (default is '1'): number of groups input channels and output channels are divided into.

    -kernel_shape list of ints: The shape of the convolution kernel. If not present, should be inferred from input W.

    -pads list of ints: Padding for the beginning and ending along each spatial axis, it can take any value greater
    than or equal to 0. The value represent the number of pixels added to the beginning and end part of the
    corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin
    the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis
    `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults
    to 0 along start and end of each spatial axis.

    -strides list of ints: Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

    """

    name = "onnx.Conv"
    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    data = operand_def(TensorType[T])
    weight = operand_def(TensorType[T])
    bias = operand_def(base(TensorType[T]) | base(NoneType))
    res = result_def(TensorType[T])

    auto_pad = attr_def(StringAttr)
    dilations = attr_def(ArrayAttr[AnyIntegerAttr])
    group = attr_def(AnyIntegerAttr)
    kernel_shape = attr_def(ArrayAttr[AnyIntegerAttr])
    pads = attr_def(ArrayAttr[AnyIntegerAttr])
    strides = attr_def(ArrayAttr[AnyIntegerAttr])

    assembly_format = (
        "`(` $data `,` $weight `,`$bias`)` attr-dict `:` `(` type($data) `,"
        "` type($weight) `,`type($bias)`)` `->` type($res) "
    )

    def __init__(
        self,
        data: SSAValue,
        weight: SSAValue,
        bias: SSAValue,
        auto_pad: Attribute,
        dilations: Attribute,
        group: Attribute,
        kernel_shape: Attribute,
        pads: Attribute,
        strides: Attribute,
    ):
        super().__init__(
            attributes={
                "auto_pad": auto_pad,
                "dilations": dilations,
                "group": group,
                "kernel_shape": kernel_shape,
                "pads": pads,
                "strides": strides,
            },
            operands=[data, weight, bias],
            result_types=[data.type],
        )

    def verify_(self) -> None:
        if (
            not isinstance(data_type := self.data.type, TensorType)
            or not isinstance(weight_type := self.weight.type, TensorType)
            or not isinstance(bias_type := self.bias.type, TensorType | NoneType)
            or not isinstance(res_type := self.res.type, TensorType)
        ):
            assert False, (
                "onnx elementwise operation operands (data, weight) and result (res) must be of type TensorType,"
                "operand (bias) must be of type TensorType or NoneType"
            )

        weight_type = cast(TensorType[Attribute], weight_type)
        data_type = cast(TensorType[Attribute], data_type)
        res_type = cast(TensorType[Attribute], res_type)

        # case that bias is a tensor type
        if isinstance(bias_type, TensorType):
            bias_type = bias_type.get_shape()
            if len(bias_type) != 1:
                raise VerifyException("bias must be 1D")

        weight_type = weight_type.get_shape()
        # kernel_shape
        kernel_shape_data: list[int] = []
        for value in self.kernel_shape:
            val = value.value.data
            kernel_shape_data.append(val)
        if list(weight_type[-2:]) != kernel_shape_data:
            raise VerifyException(
                "kernel shape rank and weight tensor rank are not the same"
            )

        # dilations
        for value in self.dilations:
            val = value.value.data
            if val <= 0:
                raise VerifyException("dilation value must be non zero positive")
        if len(self.dilations) != len(self.kernel_shape):
            raise VerifyException(
                "dilations rank and kernel shape rank are not the same"
            )

        # group
        if self.group.value.data < 1:
            raise VerifyException("group value must be nonnegative")

        # strides
        for value in self.strides:
            val = value.value.data
            if val <= 0:
                raise VerifyException("stride value must be non zero positive")
        if len(self.strides) != len(self.kernel_shape):
            raise VerifyException(
                "strides rank and kernel shape rank are not the same "
            )

        # pads
        for value in self.pads:
            val = value.value.data
            if val < 0:
                raise VerifyException("pads value must be nonnegative")
        if len(self.pads) != 2 * len(self.kernel_shape):
            raise VerifyException("pads rank is not twice the kernel shape rank")

        auto_pad_strings = ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"]
        if self.auto_pad.data not in auto_pad_strings:
            raise VerifyException(
                f"Invalid auto_pad string. Must be one of {auto_pad_strings}"
            )


@irdl_op_definition
class Constant(IRDLOperation):
    """
    Produce a constant tensor.

    Exactly one of the provided attributes, either value, sparse_value, or value_* must be specified.

    Attributes:
    - sparse_value: sparse_tensor
    The value for the elements of the output tensor in sparse format. (currently unsupported)
    - value : tensor
     The value for the elements of the output tensor.
    - value_float: float
     The value for the sole element for the scalar, float32, output tensor.
    - value_floats: list of floats
     The values for the elements for the 1D, float32, output tensor.
    - value_int : int
     The value for the sole element for the scalar, int64, output tensor.
    - value_ints : list of ints
     The values for the elements for the 1D, int64, output tensor.
    - value_string : string
     The value for the sole element for the scalar, UTF-8 string, output tensor.
    - value_strings: list of strings
     The values for the elements for the 1D, UTF-8 string, output tensor.
    """

    name = "onnx.Constant"
    output = result_def(AnyTensorType)

    value = opt_attr_def(DenseIntOrFPElementsAttr)
    value_float = opt_attr_def(FloatAttr[Float32Type])
    value_floats = opt_attr_def(ArrayAttr[FloatAttr[Float32Type]])
    value_int = opt_attr_def(IntegerAttr[IntegerType])
    value_ints = opt_attr_def(ArrayAttr[IntegerAttr[IntegerType]])
    value_string = opt_attr_def(StringAttr)
    value_strings = opt_attr_def(ArrayAttr[StringAttr])

    def __init__(
        self,
        value: Attribute | None,
        value_float: Attribute | None,
        value_floats: Attribute | None,
        value_int: Attribute | None,
        value_ints: Attribute | None,
        value_string: Attribute | None,
        value_strings: Attribute | None,
        output_type: Attribute | None,
    ):
        super().__init__(
            attributes={
                "value": value,
                "value_float": value_float,
                "value_floats": value_floats,
                "value_int": value_int,
                "value_ints": value_ints,
                "value_string": value_string,
                "value_strings": value_strings,
            },
            operands=[],
            result_types=[output_type],
        )

    def verify_(self) -> None:
        if self.value is not None and not isinstance(self.value.type, TensorType):
            raise VerifyException("value attribute type must be of type TensorType")

        if self.value_int is not None and self.value_int.type.width.data != 64:
            raise VerifyException(
                "value_int element type has to be a 64-bit signless integer"
            )

        if self.value_ints is not None:
            for value in self.value_ints:
                width = value.type.width.data
                if width != 64:
                    raise VerifyException(
                        "value_ints elements type has to be a 64-bit signless integer"
                    )

        attrs = [
            self.value,
            self.value_float,
            self.value_floats,
            self.value_int,
            self.value_ints,
            self.value_string,
            self.value_strings,
        ]
        used_attrs = sum(1 for attr in attrs if attr is not None)
        if used_attrs != 1:
            raise VerifyException(
                f"Only one value attribute must be provided, but {used_attrs} were specified"
            )

    def print(self, printer: Printer):
        if self.value is not None:
            printer.print(" ")
            printer.print(self.value)

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        v = parser.parse_attribute()
        if not isinstance(v, DenseIntOrFPElementsAttr):
            raise NotImplementedError()
        constant = cls(v, None, None, None, None, None, None, v.type)
        return constant


@irdl_op_definition
class MaxPoolSingleOut(IRDLOperation):
    """
    ONNX MaxPool operation with a single output.

     Attributes:

    - auto_pad string (default is 'NOTSET'):  auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or
    VALID. Where default value is NOTSET, which means explicit padding is used. SAME_UPPER or SAME_LOWER mean pad the
    input so that output_shape[i] = ceil(input_shape[i] / strides[i]) for each axis i. The padding is split between
    the two sides equally or almost equally (depending on whether it is even or odd). In case the padding is an odd
    number, the extra padding is added at the end for SAME_UPPER and at the beginning for SAME_LOWER.

    - ceil_mode  int (default is '1'): Whether to use ceil or floor (default) to compute the output shape.

    - dilations list of ints: Dilation value along each spatial axis of filter.

    - kernel_shape list of ints: The size of the kernel along each axis.

    - pads list of ints: Padding for the beginning and ending along each spatial axis, it can take any value greater
    than or equal to 0. The value represent the number of pixels added to the beginning and end part of the
    corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin
    the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis
    `i`. This attribute cannot be used simultaneously with auto_pad attribute. If not present, the padding defaults
    to 0 along start and end of each spatial axis.

    - storage_order int (default is '0') : The storage order of the tensor. 0 is row major, and 1 is column major.
    This attribute is used only to convert an n-tuple index value into a single integer value for producing the
    second output.

    - strides list of ints: Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis

    """

    name = "onnx.MaxPoolSingleOut"

    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    data = operand_def(base(TensorType[T]) | base(MemRefType[T]))
    output = result_def(base(TensorType[T]) | base(MemRefType[T]))

    auto_pad = attr_def(StringAttr)
    ceil_mode = attr_def(AnyIntegerAttr)
    dilations = attr_def(ArrayAttr[AnyIntegerAttr])
    kernel_shape = attr_def(ArrayAttr[AnyIntegerAttr])
    pads = attr_def(ArrayAttr[AnyIntegerAttr])
    storage_order = attr_def(AnyIntegerAttr)
    strides = attr_def(ArrayAttr[AnyIntegerAttr])

    assembly_format = (
        "`(` $data`)` attr-dict `:` `(` type($data) `)` `->` type($output)"
    )

    def __init__(
        self,
        data: SSAValue,
        auto_pad: Attribute,
        ceil_mode: Attribute,
        dilations: Attribute,
        kernel_shape: Attribute,
        pads: Attribute,
        storage_order: Attribute,
        strides: Attribute,
    ):
        super().__init__(
            attributes={
                "auto_pad": auto_pad,
                "ceil_mode": ceil_mode,
                "dilations": dilations,
                "kernel_shape": kernel_shape,
                "pads": pads,
                "storage_order": storage_order,
                "strides": strides,
            },
            operands=[data],
            result_types=[data.type],
        )

    def verify_(self) -> None:
        if not isinstance(
            data_type := self.data.type, TensorType | MemRefType
        ) or not isinstance(output_type := self.output.type, TensorType | MemRefType):
            assert False, (
                "onnx elementwise operation operands (data) and result (output) must be of type TensorType or "
                "MemRefTyoe "
            )

        data_type = cast(TensorType[Attribute], data_type)
        output_type = cast(TensorType[Attribute], output_type)

        # auto pad
        auto_pad_strings = ["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"]
        if self.auto_pad.data not in auto_pad_strings:
            raise VerifyException(
                f"Invalid auto_pad string. Must be one of {auto_pad_strings}"
            )

        # ceil mode
        if self.ceil_mode.value.data < 0 or self.ceil_mode.value.data > 1:
            raise VerifyException("ceil value must be either zero or one")

        # kernel shape
        if (input_dims := len(data_type.get_shape()) - 2) != (
            kernel_dims := len(self.kernel_shape)
        ):
            raise VerifyException(
                f"input data and kernel shape rank mismatch: ({input_dims}) vs ({kernel_dims})"
            )

        # dilations
        for value in self.dilations:
            val = value.value.data
            if val <= 0:
                raise VerifyException("dilation value must be non zero positive")

        if (dilations_dims := len(self.dilations)) != (
            kernel_dims := len(self.kernel_shape)
        ):
            raise VerifyException(
                f"dilations rank ({dilations_dims}) and kernel shape rank ({kernel_dims}) are not the "
                f"same "
            )

        # storage order
        # Not supported for storage order in column major mode in onnx-mlir (therefore row major mode only considered)
        if self.storage_order.value.data != 0:
            raise VerifyException("column major storage order not implemented yet")

        # strides
        for value in self.strides:
            val = value.value.data
            if val <= 0:
                raise VerifyException("stride value must be non zero positive")

        if (strides_dims := len(self.strides)) != (
            kernel_dims := len(self.kernel_shape)
        ):
            raise VerifyException(
                f"strides rank ({strides_dims}) and kernel shape rank ({kernel_dims}) are not the "
                f"same "
            )

        # pads
        for value in self.pads:
            val = value.value.data
            if val < 0:
                raise VerifyException("pads value must be nonnegative")

        if (pads_dims := len(self.pads)) != 2 * len(self.kernel_shape):
            raise VerifyException(
                f"pads rank ({pads_dims}) is not twice the kernel shape rank ({len(self.kernel_shape)})"
            )


@irdl_op_definition
class EntryPoint(IRDLOperation):
    """
    Indicate ONNX entry point
    The "onnx.EntryPoint" function indicates the main entry point of ONNX model.
    """

    name = "onnx.EntryPoint"
    func = attr_def(SymbolRefAttr)

    def __init__(self, func: Attribute):
        super().__init__(
            attributes={
                "func": func,
            },
        )


@irdl_op_definition
class MatMul(IRDLOperation):
    """
    The operation MatMul performs matrix multiplication between two input matrices, A and B, and returns the result as matrix Y.
    Matrix multiplication is a fundamental operation in linear algebra, where each element of the resulting matrix Y is computed by taking the
    dot product of the corresponding row of matrix A and column of matrix B.
    """

    name = "onnx.MatMul"

    # describe annotated type
    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]

    # input matrices
    matrix_A = operand_def(TensorType[T])
    matrix_B = operand_def(TensorType[T])

    # output matrices
    matrix_Y = result_def(TensorType[T])

    assembly_format = (
        "`(` $matrix_A `,` $matrix_B `)` attr-dict `:` `(` type($matrix_A) `,"
        "` type($matrix_B) `)` `->` type($matrix_Y) "
    )

    def __init__(
        self,
        matrix_A: SSAValue,
        matrix_B: SSAValue,
        matrix_Y_type: Attribute,
    ):
        super().__init__(
            operands=[matrix_A, matrix_B],
            result_types=[matrix_Y_type],
        )

    def verify_(self) -> None:
        # store dimensions of tensor A and tensor B
        res_shape: list[int] = []
        matrix_A_type = cast(TensorType[Any], self.matrix_A.type)
        matrix_B_type = cast(TensorType[Any], self.matrix_B.type)
        matrix_Y_type = cast(TensorType[Any], self.matrix_Y.type)

        # check shape compatibility
        matrix_A_shape = matrix_A_type.get_shape()
        matrix_B_shape = matrix_B_type.get_shape()

        if matrix_A_type.get_num_dims() != 2:
            raise VerifyException("input matrix A should be a 2D tensor")

        if matrix_B_type.get_num_dims() != 2:
            raise VerifyException("input matrix B should be a 2D tensor")

        if matrix_A_shape[1] != matrix_B_shape[0]:
            raise VerifyException(
                f"operands have incompatible shapes: {matrix_A_shape} and {matrix_B_shape}"
            )
        else:
            res_shape.append(matrix_A_shape[0])
            res_shape.append(matrix_B_shape[1])

        matrix_Y_type_shape = list(matrix_Y_type.get_shape())
        if (
            len(res_shape) != len(matrix_Y_type_shape)
            or res_shape != matrix_Y_type_shape
        ):
            raise VerifyException(
                f"result shape {res_shape} does not match result type {matrix_Y_type_shape}"
            )


@irdl_op_definition
class Transpose(IRDLOperation):
    """
    The transpose_tensor function takes a tensor as input and returns its transpose.
    Transposing a tensor means flipping its dimensions, so that rows become columns and vice versa.
    """

    name = "onnx.Transpose"

    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    tensor_input = operand_def(TensorType[T])

    perm = opt_attr_def(ArrayAttr[AnyIntegerAttr], attr_name="perm")

    tensor_output = result_def(TensorType[T])

    assembly_format = (
        "`(` $tensor_input `)` attr-dict `:` `(` type($tensor_input) "
        "`)` `->` type($tensor_output) "
    )

    def __init__(self, tensor_input: SSAValue, perm: Attribute):
        super().__init__(
            attributes={"perm": perm},
            operands=[tensor_input],
            result_types=[tensor_input.type],
        )

    def verify_(self) -> None:
        if not isinstance(
            tensor_input_type := self.tensor_input.type, TensorType
        ) or not isinstance(tensor_output_type := self.tensor_output.type, TensorType):
            assert False, "onnx elementwise operation operands and result must be of type TensorType"

        tensor_input_shape = tensor_input_type.get_shape()
        tensor_output_shape = tensor_output_type.get_shape()

        # numbers in perm cannot be repeated
        if self.perm is not None:
            for _, int_attr in enumerate(self.perm.data):
                attr_value = int_attr.value.data
                count = self.perm.data.count(int_attr)
                if count != 1:
                    raise VerifyException(
                        f"permutation can not contain more than one occurrence of the same dimension: dimension #{attr_value} appears {count} times."
                    )

            # numbers in perm must be between 0 and len(tensor_input_shape)-1
            perm_size = len(self.perm.data)
            for int_index, int_attr in enumerate(self.perm.data):
                int_index = int_index + 0
                int_attr_val = int_attr.value.data
                if int_attr_val < 0 or int_attr_val >= perm_size:
                    raise VerifyException(
                        f"permutation can only contain values between 0 and {perm_size}-1: dimension #{int_index} value is {int_attr_val}"
                    )

            # len(tensor_input_shape) must be equal to len(perm)
            perm_size = len(self.perm.data)
            input_size = len(tensor_input_shape)
            if perm_size != input_size:
                raise VerifyException(
                    f"permutation and inputs dimensions must have the same size: #dimensions input is {input_size}, #dimension perimutation is {perm_size}"
                )

            # check output shape
            for index_attr, int_attr in enumerate(self.perm.data):
                int_attr_val = int_attr.value.data
                if tensor_output_shape[index_attr] != tensor_input_shape[int_attr_val]:
                    raise VerifyException(
                        f"incorrect output shape: output dimension #{index_attr} should be equal to {tensor_input_shape[int_attr_val]}"
                    )


@irdl_op_definition
class Squeeze(IRDLOperation):
    """
    Squeeze the input tensor along the specified axes.

    Squeezing a tensor removes dimensions of size 1, effectively reducing the rank of the tensor and collapsing those dimensions.
    This operation is particularly useful for removing unnecessary singleton dimensions, which may arise from broadcasting or previous operations.

    Args:
        input_tensor: The input tensor to be squeezed. This tensor should be a multi-dimensional array-like object.
        axes: A list of axes along which to squeeze the tensor. If provided, only the specified axes will be squeezed. If not provided, all dimensions of size 1 will be squeezed.

    Returns:
        output_tensor: The squeezed tensor.
    """

    name = "onnx.Squeeze"

    T = Annotated[AnyFloat | IntegerType, ConstraintVar("T")]
    input_tensor = operand_def(TensorType[T])
    axes = opt_attr_def(base(AnyIntegerAttr), attr_name="axes")

    output_tensor = result_def(TensorType[T])

    assembly_format = "`(` $input_tensor `)` attr-dict `:` `(` type($input_tensor) `)` `->` type($output_tensor) "

    def __init__(
        self,
        input_tensor: SSAValue,
        axes: Attribute,
    ):
        super().__init__(
            attributes={
                "axes": axes,
            },
            operands=[input_tensor],
            result_types=[input_tensor.type],
        )

    def verify_(self) -> None:
        if not isinstance(input_tensor_type := self.input_tensor.type, TensorType):
            assert False, "onnx elementwise operation operands and result must be of type TensorType"

        input_tensor_shape = input_tensor_type.get_shape()

        if self.axes is not None:
            axes_value = self.axes.value.data

            # axes out of bounds: the axes value must between 0 and len(input_tensor.shape)-1
            if axes_value < 0 or axes_value >= len(input_tensor_shape):
                max_axes_value = len(input_tensor_shape) - 1
                raise VerifyException(
                    f"axes to squeeze must be between 0 and {max_axes_value}, axes: {axes_value}"
                )


@irdl_op_definition
class Sigmoid(IRDLOperation):
    """
    Applies the sigmoid function element-wise to all elements of the input tensor.
    The sigmoid function, denoted by sigma(x), is a common mathematical function used in machine learning and neural networks. It is defined as:
    sigma(x) = 1 / (1 + e^-x)
    where e is the base of the natural logarithm. The sigmoid function maps any real-valued number to the range of [0, 1].
    The sigmoid function is used as an activation function.

    Args:
    - input_tensor (TensorType): The input tensor to which the sigmoid function will be applied.

    Returns:
    - output_tensor (TensorType): The output tensor after applying the sigmoid function element-wise to the input tensor.
    """

    name = "onnx.Sigmoid"

    T = Annotated[AnyFloat, ConstraintVar("T")]
    input_tensor = operand_def(TensorType[T])
    output_tensor = result_def(TensorType[T])

    assembly_format = "`(` $input_tensor`)` attr-dict `:` `(` type($input_tensor) `)` `->` type($output_tensor) "

    def __init__(
        self,
        input_tensor: SSAValue,
    ):
        super().__init__(
            operands=[input_tensor],
            result_types=[input_tensor.type],
        )

    def verify_(self) -> None:
        if not isinstance(
            input_tensor_type := self.input_tensor.type, TensorType
        ) or not isinstance(output_tensor_type := self.output_tensor.type, TensorType):
            assert False, "onnx elementwise operation operands and result must be of type TensorType"

        input_tensor_shape = input_tensor_type.get_shape()
        output_tensor_shape = output_tensor_type.get_shape()

        # check if input tensor and output tensor have the same shape
        if input_tensor_shape != output_tensor_shape:
            raise VerifyException(
                f"tensor input shape {input_tensor_shape} is not equal to tensor output shape {output_tensor_shape}"
            )


ONNX = Dialect(
    "onnx",
    [
        Abs,
        Add,
        Constant,
        Conv,
        Div,
        EntryPoint,
        Gemm,
        MatMul,
        MaxPoolSingleOut,
        Mul,
        Relu,
        Reshape,
        Sub,
        Transpose,
        Squeeze,
        Sigmoid,
    ],
)
