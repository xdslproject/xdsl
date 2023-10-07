from itertools import product
from typing import Any

from xdsl.dialects import stream
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
class StreamFunctions(InterpreterFunctions):
    @impl(stream.GenericOp)
    def run_generic(
        self, interpreter: Interpreter, op: stream.GenericOp, args: tuple[Any, ...]
    ) -> PythonValues:
        # bla
        assert not op.stream_inputs
        assert not op.stream_outputs
        assert op.iter_count is None

        inputs: tuple[ShapedArray[float], ...] = interpreter.get_values(
            op.memref_inputs
        )
        outputs: tuple[ShapedArray[float], ...] = interpreter.get_values(
            op.memref_outputs
        )
        indexing_maps = op.get_indexing_maps()

        assert inputs, "inputs should not be empty"
        assert len(inputs) == len(indexing_maps) - 1
        assert len(outputs) == 1, "can only handle single output map for now"

        loop_ranges = op.get_static_loop_ranges()

        for indices in product(*(range(loop_range) for loop_range in loop_ranges)):
            loop_args = tuple(
                i.load(tuple(indexing_map.eval(list(indices), [])))
                for i, indexing_map in zip(inputs, indexing_maps)
            )
            (loop_results,) = interpreter.run_ssacfg_region(
                op.body, loop_args, "for_loop"
            )
            result_indices = indexing_maps[-1].eval(list(indices), [])
            outputs[0].store(tuple(result_indices), loop_results)

        return ()

    @impl_terminator(stream.YieldOp)
    def run_br(
        self, interpreter: Interpreter, op: stream.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
