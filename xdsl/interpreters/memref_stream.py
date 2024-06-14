from itertools import product
from typing import Any, cast

from xdsl.dialects import memref_stream
from xdsl.dialects.builtin import UnitAttr
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

        outer_ubs, inner_ubs = op.get_static_loop_ranges()

        inits = op.inits.data

        if inner_ubs:
            inputs: tuple[ShapedArray[float] | float, ...] = args[:inputs_count]
            input_indexing_maps = indexing_maps[:inputs_count]
            for outer_indices in product(*(range(outer_ub) for outer_ub in outer_ubs)):
                output_loop_args = tuple(
                    (
                        (cast(ShapedArray[int | float], o)).load(
                            indexing_map.eval(outer_indices, ())
                        )
                        if isinstance(init, UnitAttr)
                        else init.value.data
                    )
                    for o, indexing_map, init in zip(
                        outputs, output_indexing_maps, inits, strict=True
                    )
                )
                for inner_indices in product(
                    *(range(inner_ub) for inner_ub in inner_ubs)
                ):
                    input_loop_args = tuple(
                        (
                            (cast(ShapedArray[Any], i)).load(
                                indexing_map.eval(outer_indices + inner_indices, ())
                            )
                            if isinstance(i, ShapedArray)
                            else i
                        )
                        for i, indexing_map in zip(
                            inputs, input_indexing_maps, strict=True
                        )
                    )

                    loop_results = interpreter.run_ssacfg_region(
                        op.body, input_loop_args + output_loop_args, "for_loop"
                    )
                    output_loop_args = loop_results
                print(output_loop_args, output_indexing_maps, outputs)
                for res, indexing_map, output in zip(
                    output_loop_args, output_indexing_maps, outputs, strict=True
                ):
                    result_indices = indexing_map.eval(outer_indices, ())
                    output.store(result_indices, res)
        else:
            for indices in product(*(range(outer_ub) for outer_ub in outer_ubs)):
                loop_args = tuple(
                    (
                        (cast(ShapedArray[Any], i)).load(indexing_map.eval(indices, ()))
                        if isinstance(i, ShapedArray)
                        else i
                    )
                    for i, indexing_map in zip(args, indexing_maps, strict=True)
                )
                loop_results = interpreter.run_ssacfg_region(
                    op.body, loop_args, "for_loop"
                )
                for res, indexing_map, output in zip(
                    loop_results, output_indexing_maps, outputs, strict=True
                ):
                    result_indices = indexing_map.eval(indices, ())
                    output.store(result_indices, res)

        return ()

    @impl_terminator(memref_stream.YieldOp)
    def run_yield(
        self, interpreter: Interpreter, op: memref_stream.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
