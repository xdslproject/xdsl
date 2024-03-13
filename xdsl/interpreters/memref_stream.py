from itertools import product
from typing import Any, cast

from xdsl.dialects import memref_stream
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
class MemrefStreamFunctions(InterpreterFunctions):
    @impl(memref_stream.GenericOp)
    def run_generic(
        self,
        interpreter: Interpreter,
        op: memref_stream.GenericOp,
        args: tuple[Any, ...],
    ) -> PythonValues:

        inputs_count = len(op.inputs)

        outputs: tuple[ShapedArray[float], ...] = args[inputs_count:]

        indexing_maps = tuple(attr.data for attr in op.indexing_maps)
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

    @impl_terminator(memref_stream.YieldOp)
    def run_yield(
        self, interpreter: Interpreter, op: memref_stream.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
