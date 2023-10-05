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
        assert list(op.iterator_types.data) == [
            linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)
        ] * len(args), f"{op.iterator_types.data} {len(args)}"
        assert op.library_call is None
        assert not op.res

        inputs: tuple[ShapedArray[float], ...] = interpreter.get_values(op.inputs)
        outputs: tuple[ShapedArray[float], ...] = interpreter.get_values(op.outputs)
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

    @impl_terminator(linalg.Yield)
    def run_br(self, interpreter: Interpreter, op: linalg.Yield, args: tuple[Any, ...]):
        return ReturnedValues(args), ()
