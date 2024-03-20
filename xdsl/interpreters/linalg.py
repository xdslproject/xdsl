import math
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
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        return (
            ShapedArray(list(l + r for l, r in zip(lhs.data, rhs.data)), lhs.shape),
        )

    @impl(linalg.FillOp)
    def run_fill(
        self, interpreter: Interpreter, op: linalg.FillOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        (operand,) = args
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        result_type = op.res[0].type
        assert isinstance(result_type, TensorType)
        result_shape = list(result_type.get_shape())
        return (
            ShapedArray(list(operand.data * math.prod(result_shape)), result_shape),
        )

    @impl(linalg.MulOp)
    def run_mul(
        self, interpreter: Interpreter, op: linalg.MulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape == rhs.shape
        return (
            ShapedArray(list(l * r for l, r in zip(lhs.data, rhs.data)), lhs.shape),
        )

    @impl(linalg.TransposeOp)
    def run_transpose(
        self, interpreter: Interpreter, op: linalg.TransposeOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        operand = args[0]
        assert isinstance(operand, ShapedArray)
        operand = cast(ShapedArray[float], operand)
        assert len(operand.shape) == 2

        transposed_data: list[float] = []

        for c in range(operand.shape[1]):
            for r in range(operand.shape[0]):
                transposed_data.append(operand.load((r, c)))

        return (ShapedArray(list(transposed_data), operand.shape[::-1]),)

    @impl(linalg.MatmulOp)
    def run_mat_mul(
        self, interpreter: Interpreter, op: linalg.MatmulOp, args: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        lhs, rhs = args[0], args[1]
        assert isinstance(lhs, ShapedArray)
        assert isinstance(rhs, ShapedArray)
        lhs = cast(ShapedArray[float], lhs)
        rhs = cast(ShapedArray[float], rhs)
        assert lhs.shape[1] == rhs.shape[0]

        # reshape the arrays
        a = [
            lhs.data[x : x + lhs.shape[1]]
            for x in range(0, len(lhs.data), lhs.shape[1])
        ]
        b = [
            rhs.data[x : x + rhs.shape[1]]
            for x in range(0, len(rhs.data), rhs.shape[1])
        ]

        # initialise a result list
        matrix_result: list[list[int]] = [
            [0 for _ in range(rhs.shape[1])] for _ in range(lhs.shape[0])
        ]

        # do matmul
        for i in range(lhs.shape[0]):
            for j in range(rhs.shape[1]):
                for k in range(lhs.shape[1]):
                    matrix_result[i][j] += a[i][k] * b[k][j]

        # flatten the result
        result: list[float] = [ele for row in matrix_result for ele in row]
        return (ShapedArray(result, list([lhs.shape[0], rhs.shape[1]])),)
