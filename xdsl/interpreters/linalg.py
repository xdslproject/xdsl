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
        assert all(res.data_ptr[i] == 0 for i in range(len(res.data)))
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        res = cast(ShapedArray[float], res)
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
        assert all(res.data_ptr[i] == 0 for i in range(len(res.data)))
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
        assert all(res.data_ptr[i] == 0 for i in range(len(res.data)))
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
        assert all(res.data_ptr[i] == 0 for i in range(len(res.data)))
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
        assert all(res.data_ptr[i] == 0 for i in range(len(res.data)))
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
