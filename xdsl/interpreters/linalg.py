from itertools import product
from typing import Any, cast

from xdsl.dialects import linalg
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


@register_impls
class LinalgFunctions(InterpreterFunctions):
    @impl(linalg.Generic)
    def run_generic(
        self, interpreter: Interpreter, op: linalg.Generic, args: tuple[Any, ...]
    ) -> PythonValues:
        if op.library_call is not None:
            raise NotImplementedError(
                "library_call not yet supported in linalg.generic interpreter"
            )
        if op.res:
            raise NotImplementedError(
                "results not yet supported in linalg.generic interpreter"
            )

        inputs_count = len(op.inputs)

        outputs: tuple[ShapedArray[float], ...] = args[inputs_count:]

        indexing_maps = op.get_indexing_maps()
        output_indexing_maps = indexing_maps[inputs_count:]

        loop_ranges = op.get_static_loop_ranges()

        for indices in product(*(range(loop_range) for loop_range in loop_ranges)):
            loop_args = tuple(
                (
                    (cast(ShapedArray[Any], i)).load(indexing_map.eval(indices, ()))
                    if isinstance(i, ShapedArray)
                    else i
                )
                for i, indexing_map in zip(args, indexing_maps, strict=True)
            )
            loop_results = interpreter.run_ssacfg_region(op.body, loop_args, "for_loop")
            for res, indexing_map in zip(
                loop_results, output_indexing_maps, strict=True
            ):
                result_indices = indexing_map.eval(indices, ())
                outputs[0].store(result_indices, res)

        return ()

    @impl_terminator(linalg.YieldOp)
    def run_yield(
        self, interpreter: Interpreter, op: linalg.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()

    @impl(linalg.AddOp)
    def run_add(
        self, interpreter: Interpreter, op: linalg.AddOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        (lhs, rhs, res) = (args[0], args[1], args[2])
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        assert isinstance(res, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        res = cast(ShapedArray[float], res)
        assert all(res.data_ptr[i] == 0.0 for i in range(len(res.data)))
        assert lhs.shape == rhs.shape == res.shape
        for i in range(len(lhs.data)):
            res.data_ptr[i] = lhs.data_ptr[i] + rhs.data_ptr[i]
        if len(op.results) > 0:
            return (res,)
        return ()

    @impl(linalg.FillOp)
    def run_fill(
        self, interpreter: Interpreter, op: linalg.FillOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        operand, res = args[0], args[1]
        assert isinstance(operand, ShapedArray)
        assert isinstance(res, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        res = cast(ShapedArray[float], res)
        assert all(res.data_ptr[i] == 0.0 for i in range(len(res.data)))
        for i in range(len(res.data)):
            res.data_ptr[i] = operand.data_ptr[0]
        if len(op.results) > 0:
            return (res,)
        return ()

    @impl(linalg.MulOp)
    def run_mul(
        self, interpreter: Interpreter, op: linalg.MulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs, res = args[0], args[1], args[2]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        assert isinstance(res, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        res = cast(ShapedArray[float], res)
        assert all(res.data_ptr[i] == 0.0 for i in range(len(res.data)))
        assert lhs.shape == rhs.shape == res.shape
        for i in range(len(lhs.data)):
            res.data_ptr[i] = lhs.data_ptr[i] * rhs.data_ptr[i]
        if len(op.results) > 0:
            return (res,)
        return ()

    @impl(linalg.TransposeOp)
    def run_transpose(
        self, interpreter: Interpreter, op: linalg.TransposeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        operand, res = args[0], args[1]
        assert isinstance(operand, ShapedArray)
        assert isinstance(res, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        res = cast(ShapedArray[float], res)
        assert all(res.data_ptr[i] == 0.0 for i in range(len(res.data)))
        assert len(operand.shape) == 2
        assert len(res.shape) == 2
        rows, cols = operand.shape
        for i in range(rows):
            for j in range(cols):
                res.data_ptr[j * rows + i] = operand.data_ptr[i * cols + j]
        if len(op.results) > 0:
            return (res,)
        return ()

    @impl(linalg.MatmulOp)
    def run_mat_mul(
        self, interpreter: Interpreter, op: linalg.MatmulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs, res = args[0], args[1], args[2]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        assert isinstance(res, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        res = cast(ShapedArray[float], res)
        assert all(res.data_ptr[i] == 0.0 for i in range(len(res.data)))
        rows = lhs.shape[0]
        cols = rhs.shape[1]
        assert rows == cols
        for i in range(rows):
            for j in range(cols):
                res.data_ptr[i * cols + j] = sum(
                    lhs.data_ptr[i * lhs.shape[1] + k]
                    * rhs.data_ptr[k * rhs.shape[1] + j]
                    for k in range(lhs.shape[1])
                )

        if len(op.results) > 0:
            return (res,)
        return ()

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
        assert all(res.data_ptr[i] == 0.0 for i in range(len(res.data)))

        strides_shape = op.strides.get_shape()
        strides = tuple(value.value.data for value in op.strides.data)
        if len(strides_shape) != 2:
            raise NotImplementedError("Only 2d max pooling supported")

        m_height, m_width = input.shape[2:]
        out_height = int((m_height - kernel_filter.shape[2]) // strides[0] + 1)
        out_width = int((m_width - kernel_filter.shape[3]) // strides[0] + 1)

        ky, kx = kernel_filter.shape[0], kernel_filter.shape[1]

        output = [
            [[[0] * out_width for _ in range(out_height)] for _ in range(m_width)]
            for _ in range(m_height[0])
        ]

        for k in range(input.shape[0]):
            for l in range(input.shape[1]):
                for i in range(0, m_height - ky + 1, strides[0]):
                    for j in range(0, m_width - kx + 1, strides[1]):
                        output[k][l][i // strides[0]][j // strides[0]] = max(
                            input.data[k][l][i : i + ky][j + kx]
                        )

        print(output)
