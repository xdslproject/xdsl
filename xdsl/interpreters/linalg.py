from itertools import product
from typing import Any, cast

from xdsl.dialects import linalg
from xdsl.dialects.builtin import TensorType
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray


def run_linalg_structured_op(
    interpreter: Interpreter,
    op: linalg.LinalgStructuredOperation,
    args: tuple[ShapedArray[float] | float, ...],
):
    """
    Helper function for interpreting ops inheriting from
    [`LinalgStructuredOperation`][xdsl.dialects.linalg.LinalgStructuredOperation].
    """
    body = op.body
    inputs_count = len(op.inputs)
    input_args = args[:inputs_count]
    output_args = args[inputs_count:]
    results = op.results
    indexing_maps = tuple(attr.data for attr in op.get_indexing_maps())
    loop_ranges = op.get_static_loop_ranges()

    if any(not isinstance(arg, ShapedArray) for arg in output_args):
        raise NotImplementedError("Only shaped out results are implemented")
    output_args = cast(tuple[ShapedArray[float], ...], output_args)
    if results:
        # If there are results, they must be tensors, initialised with the
        # `output_args`. If not, the results are stored in output_args directly.
        outputs = tuple(arg.copy() for arg in output_args)
    else:
        outputs = output_args

    loop_shaped_args = input_args + outputs

    output_indexing_maps = indexing_maps[inputs_count:]

    for indices in product(*(range(loop_range) for loop_range in loop_ranges)):
        loop_scalar_args = tuple(
            (
                i.load(indexing_map.eval(indices, ()))
                if isinstance(i, ShapedArray)
                else i
            )
            for i, indexing_map in zip(loop_shaped_args, indexing_maps, strict=True)
        )
        loop_results = interpreter.run_ssacfg_region(body, loop_scalar_args, "for_loop")
        for res, indexing_map in zip(loop_results, output_indexing_maps, strict=True):
            result_indices = indexing_map.eval(indices, ())
            outputs[0].store(result_indices, res)

    return outputs if results else ()


@register_impls
class LinalgFunctions(InterpreterFunctions):
    @impl(linalg.GenericOp)
    def run_generic(
        self, interpreter: Interpreter, op: linalg.GenericOp, args: tuple[Any, ...]
    ) -> PythonValues:
        if op.library_call is not None:
            raise NotImplementedError(
                "library_call not yet supported in linalg.generic interpreter"
            )
        return run_linalg_structured_op(interpreter, op, args)

    @impl_terminator(linalg.YieldOp)
    def run_yield(
        self, interpreter: Interpreter, op: linalg.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()

    @impl(linalg.AddOp)
    def run_add(
        self, interpreter: Interpreter, op: linalg.AddOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return run_linalg_structured_op(interpreter, op, args)

    @impl(linalg.FillOp)
    def run_fill(
        self, interpreter: Interpreter, op: linalg.FillOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        operand, res = args[0], args[1]
        assert isinstance(operand, ShapedArray)
        assert isinstance(res, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        res = cast(ShapedArray[float], res)
        if not all(res.data_ptr[i] == 0.0 for i in range(len(res.data))):
            raise NotImplementedError()
        for i in range(len(res.data)):
            res.data_ptr[i] = operand.data_ptr[0]
        if len(op.results) > 0:
            return (res,)
        return ()

    @impl(linalg.MulOp)
    def run_mul(
        self, interpreter: Interpreter, op: linalg.MulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return run_linalg_structured_op(interpreter, op, args)

    @impl(linalg.TransposeOp)
    def run_transpose(
        self, interpreter: Interpreter, op: linalg.TransposeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return run_linalg_structured_op(interpreter, op, args)

    @impl(linalg.MatmulOp)
    def run_mat_mul(
        self, interpreter: Interpreter, op: linalg.MatmulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        return run_linalg_structured_op(interpreter, op, args)

    @impl(linalg.PoolingNchwMaxOp)
    def run_pooling_nchw_max(
        self,
        interpreter: Interpreter,
        op: linalg.PoolingNchwMaxOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        input, kernel_filter, res = args[0], args[1], args[2]
        assert isinstance(input, ShapedArray)
        assert isinstance(kernel_filter, ShapedArray)
        assert isinstance(res, ShapedArray)
        input = cast(ShapedArray[float], input)
        kernel_filter = cast(ShapedArray[float], kernel_filter)
        res = cast(ShapedArray[float], res)
        if not all(res.data_ptr[i] == 0.0 for i in range(len(res.data))):
            raise NotImplementedError()
        strides_type = op.strides.type
        assert isinstance(strides_type, TensorType)
        (strides_shape,) = strides_type.get_shape()
        strides = op.strides.get_values()
        if strides_shape != 2:
            raise NotImplementedError("Only 2d max pooling supported")

        m_height, m_width = input.shape[2:]
        ky, kx = kernel_filter.shape[0], kernel_filter.shape[1]

        # convert input into a numpy like array
        input_data = [
            [input.data[r * m_width + c] for c in range(m_width)]
            for r in range(m_height)
        ]

        output: list[float] = []
        for k in range(0, m_height - ky + 1, strides[0]):
            for l in range(0, m_width - kx + 1, strides[0]):
                block_max_value = float("-inf")
                for i in range(k, k + ky):
                    for j in range(l, l + kx):
                        block_max_value = max(block_max_value, input_data[i][j])
                output.append(block_max_value)
        for i in range(len(output)):
            res.data_ptr[i] = output[i]
        if len(op.results) > 0:
            return (res,)
        return ()

    @impl(linalg.Conv2DNchwFchwOp)
    def run_conv_2d_nchw_fchw(
        self,
        interpreter: Interpreter,
        op: linalg.Conv2DNchwFchwOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        input, kernel_filter, res = args[0], args[1], args[2]
        assert isinstance(input, ShapedArray)
        assert isinstance(kernel_filter, ShapedArray)
        assert isinstance(res, ShapedArray)
        input = cast(ShapedArray[float], input)
        kernel_filter = cast(ShapedArray[float], kernel_filter)
        res = cast(ShapedArray[float], res)
        if not all(res.data_ptr[i] == 0.0 for i in range(len(res.data))):
            raise NotImplementedError()
        m_height, m_width = input.shape[2:]
        ky, kx = kernel_filter.shape[2], kernel_filter.shape[3]
        strides = op.strides.get_values()
        # convert input into a numpy like array
        input_data = [
            [input.data[r * m_width + c] for c in range(m_width)]
            for r in range(m_height)
        ]
        # convert kernel into a numpy like array
        kernel_data = [
            [
                kernel_filter.data[r * kernel_filter.shape[3] + c]
                for c in range(kernel_filter.shape[3])
            ]
            for r in range(kernel_filter.shape[2])
        ]
        output: list[float] = []
        for k in range(0, m_height - ky + 1, strides[0]):
            for l in range(0, m_width - kx + 1, strides[0]):
                conv_value: float = 0.0
                for i in range(k, k + ky):
                    for j in range(l, l + kx):
                        conv_value += input_data[i][j] * kernel_data[i - k][j - l]
                output.append(conv_value)
        for i in range(len(output)):
            res.data_ptr[i] = output[i]
        if len(op.results) > 0:
            return (res,)
        return ()
