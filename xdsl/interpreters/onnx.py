from typing import Any, cast

import numpy as np

from xdsl.dialects import onnx
from xdsl.dialects.builtin import TensorType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    ReturnedValues,
    impl,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray


@register_impls
class OnnxFunctions(InterpreterFunctions):
    @impl(onnx.Add)
    def run_add(self, interpreter: Interpreter, op: onnx.Add, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) + np.array(rhs.data)
        return ShapedArray(list(result), lhs.shape)

    @impl(onnx.Sub)
    def run_sub(self, interpreter: Interpreter, op: onnx.Sub, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) - np.array(rhs.data)
        return ShapedArray(list(result), lhs.shape)

    @impl(onnx.Mul)
    def run_mul(self, interpreter: Interpreter, op: onnx.Mul, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) * np.array(rhs.data)
        return ShapedArray(list(result), lhs.shape)

    @impl(onnx.Div)
    def run_div(self, interpreter: Interpreter, op: onnx.Div, args: tuple[Any, ...]):
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        result = np.array(lhs.data) / np.array(rhs.data)
        return ShapedArray(list(result), lhs.shape)

    @impl(onnx.Relu)
    def run_relu(self, interpreter: Interpreter, op: onnx.Relu, args: tuple[Any, ...]):
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        operand_data = np.array(operand.data)
        result = operand_data * (operand_data > 0)
        return ShapedArray(list(result), operand.shape)

    @impl(onnx.Constant)
    def run_constant(
        self, interpreter: Interpreter, op: onnx.Constant, args: tuple[Any, ...]
    ):
        attr_value = list(op.attributes.values())[0]
        constant_data = list(x.value.data for x in attr_value.data.data)
        result_type = op.output.type
        assert isinstance(result_type, TensorType)
        output_shape = list(result_type.get_shape())

        return ShapedArray(constant_data, output_shape)

    @impl(onnx.Reshape)
    def run_reshape(
        self, interpreter: Interpreter, op: onnx.Reshape, args: tuple[Any, ...]
    ):
        if op.allow_zero.value.data == 1:
            raise NotImplementedError(
                "allow_zero not yet supported in onnx.reshape interpreter"
            )
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        result_type = op.reshaped.type
        assert isinstance(result_type, TensorType)
        new_shape = list(result_type.get_shape())
        operand_data = np.array(operand.data)
        return ShapedArray(list(operand_data), new_shape)

    @impl(onnx.Gemm)
    def run_gemm(self, interpreter: Interpreter, op: onnx.Gemm, args: tuple[Any, ...]):
        a, b, c = args[0], args[1], args[2]

        alpha = op.alpha.value.data if op.alpha is not None else 1.0
        beta = op.beta.value.data if op.beta is not None else 1.0

        assert isinstance(a, ShapedArray)
        assert isinstance(b, ShapedArray)
        assert isinstance(c, ShapedArray)

        a = cast(ShapedArray[float], a)
        b = cast(ShapedArray[float], b)
        c = cast(ShapedArray[float], c)

        a = np.array(a.data)
        b = np.array(b.data)
        c = np.array(c.data)

        if op.trans_a is not None and op.trans_a.value.data == 1:
            a = np.transpose(a)

        if op.trans_b is not None and op.trans_b.value.data == 1:
            b = np.transpose(b)

        result = alpha * np.dot(a, b) + beta * c

        result_type = op.res_tensor.type
        result_shape = list(result_type.get_shape())
        assert isinstance(result_type, TensorType)

        return ShapedArray(list(result), result_shape)

    @impl(onnx.Conv)
    def run_conv(self, interpreter: Interpreter, op: onnx.Conv, args: tuple[Any, ...]):

        # initialise the attributes used
        auto_pad = op.auto_pad.data
        strides: list[int] = [value.value.data for value in op.strides]
        matrix, kernel, bias = args[0], args[1], args[2]
        pads: list[int] = [value.value.data for value in op.pads]

        matrix = cast(ShapedArray[float], matrix)
        kernel = cast(ShapedArray[float], kernel)
        bias = cast(ShapedArray[float], bias)

        matrix = np.array(matrix.data).reshape(matrix.shape)
        kernel = np.array(kernel.data).reshape(kernel.shape)

        if auto_pad != "NOTSET":
            if auto_pad == "SAME_UPPER" or auto_pad == "SAME_LOWER":
                out_height = np.ceil(matrix.shape[2] / strides[0])
                out_width = np.ceil(matrix.shape[3] / strides[1])

                pad_along_height = max(
                    (out_height - 1) * strides[0] + kernel.shape[2] - matrix.shape[2], 0
                )
                pad_along_width = max(
                    (out_width - 1) * strides[1] + kernel.shape[3] - matrix.shape[3], 0
                )

                if auto_pad == "SAME_UPPER":
                    pad_top = int(pad_along_height // 2)
                    pad_bottom = int(pad_along_height - pad_top)
                    pad_left = int(pad_along_width // 2)
                    pad_right = int(pad_along_width - pad_left)
                else:
                    pad_bottom = int(pad_along_height // 2)
                    pad_top = int(pad_along_height - pad_bottom)
                    pad_right = int(pad_along_width // 2)
                    pad_left = int(pad_along_width - pad_right)

                pads = [pad_top, pad_bottom, pad_left, pad_right]

            elif auto_pad == "VALID":
                pads = [0, 0, 0, 0]  # set padding to all zeros

        if pads:
            # case of asymmetric padding
            pad_values = [
                (pads[i], pads[i + len(pads) // 2]) for i in range(len(pads) // 2)
            ]

            # pad input matrix
            padded_matrix = np.pad(
                matrix,
                (
                    (0, 0),
                    (0, 0),
                    (pad_values[0][0], pad_values[0][1]),
                    (pad_values[1][0], pad_values[1][1]),
                ),
                mode="constant",
            )

            # padded shape case
            m_height, m_width = padded_matrix.shape[2:]

        else:
            m_height, m_width = matrix.shape[2:]

            padded_matrix = matrix

        # based on strides calculate the output shape
        out_height = int((m_height - kernel.shape[2]) // strides[0] + 1)
        out_width = int((m_width - kernel.shape[3]) // strides[1] + 1)

        output = np.zeros((matrix.shape[0], matrix.shape[1], out_height, out_width))

        # do convolution
        for k in range(matrix.shape[0]):
            for l in range(matrix.shape[1]):
                for i in range(0, m_height - kernel.shape[2] + 1, strides[0]):
                    for j in range(0, m_width - kernel.shape[3] + 1, strides[1]):
                        output[k, l, i // strides[0], j // strides[1]] = np.sum(
                            padded_matrix[
                                k, l, i : i + kernel.shape[2], j : j + kernel.shape[3]
                            ]
                            * kernel[k, l]
                        )

        if bias.data[0] is not None:
            output += np.array(bias.data[0])

        output_dims = [1, 1, output.shape[2], output.shape[3]]

        result = np.array(output).flatten()
        return ShapedArray(list(result), output_dims)

    @impl(onnx.MaxPoolSingleOut)
    def run_max_pool_single_out(
        self, interpreter: Interpreter, op: onnx.MaxPoolSingleOut, args: tuple[Any, ...]
    ):
        kernel: list[int] = [value.value.data for value in op.kernel_shape]
        strides: list[int] = [value.value.data for value in op.strides]

        if kernel.sort() != strides.sort():
            raise NotImplementedError(
                "Kernel shape and strides not equal computation not yet supported in onnx.MaxPoolSingleOut interpreter"
            )

        if op.storage_order.value.data == 1:
            raise NotImplementedError(
                "storage order not yet supported in onnx.MaxPoolSingleOut interpreter"
            )

        pads: list[int] = [value.value.data for value in op.pads]
        ky, kx = kernel[0], kernel[1]
        matrix = args[0]

        matrix = cast(ShapedArray[float], matrix)
        matrix = np.array(matrix.data).reshape(matrix.shape[2], matrix.shape[3])

        m, n = matrix.shape[:2]

        if op.ceil_mode.value.data == 1:

            def output_mode(x, y):
                return int(np.ceil(x / float(y)))

        elif op.ceil_mode.value.data == 0:

            def output_mode(x, y):
                return int(np.floor(x / float(y)))

        if all(element == 1 for element in pads):
            ny = output_mode(m, ky)
            nx = output_mode(n, kx)
            size = (ny * ky, nx * kx) + matrix.shape[2:]
            mat_pad = np.full(size, np.nan)
            mat_pad[:m, :n, ...] = matrix
        else:
            ny = m // ky
            nx = n // kx
            mat_pad = matrix[: ny * ky, : nx * kx, ...]

        new_shape = (ny, ky, nx, kx) + matrix.shape[2:]

        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
        flatten_res = np.array(result).flatten()
        output_dims = [1, 1, new_shape[1], new_shape[3]]
        return ShapedArray(list(flatten_res), output_dims)

    @impl(onnx.EntryPoint)
    def run_entry_point(
        self, interpreter: Interpreter, op: onnx.EntryPoint, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
