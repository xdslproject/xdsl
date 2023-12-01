from itertools import product
from typing import Any

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
        if not all(it.data == linalg.IteratorType.PARALLEL for it in op.iterator_types):
            raise NotImplementedError(
                'Only "parallel" iterator types supported in linalg.generic interpreter'
            )
        if op.library_call is not None:
            raise NotImplementedError(
                "library_call not yet supported in linalg.generic interpreter"
            )
        if op.res:
            raise NotImplementedError(
                "results not yet supported in linalg.generic interpreter"
            )

        inputs_count = len(op.inputs)

        inputs: tuple[ShapedArray[float], ...] = args[:inputs_count]
        outputs: tuple[ShapedArray[float], ...] = args[inputs_count:]

        indexing_maps = op.get_indexing_maps()
        input_indexing_maps = indexing_maps[:inputs_count]
        output_indexing_maps = indexing_maps[inputs_count:]

        loop_ranges = op.get_static_loop_ranges()

        for indices in product(*(range(loop_range) for loop_range in loop_ranges)):
            loop_args = tuple(
                i.load(indexing_map.eval(indices, ()))
                for i, indexing_map in zip(inputs, input_indexing_maps, strict=True)
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
