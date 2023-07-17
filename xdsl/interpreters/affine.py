from typing import Any, cast

from xdsl.dialects import affine
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    ReturnedValues,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.shaped_array import ShapedArray


@register_impls
class AffineFunctions(InterpreterFunctions):
    @impl(affine.Store)
    def run_store(
        self, interpreter: Interpreter, op: affine.Store, args: tuple[Any, ...]
    ):
        value, memref, *affine_dims = args

        affine_map = op.map

        assert affine_map is not None
        memref = cast(ShapedArray[Any], memref)
        indices = affine_map.data.eval(affine_dims, [])

        indices = tuple(indices)
        memref.store(indices, value)

        return ()

    @impl(affine.Load)
    def run_load(
        self, interpreter: Interpreter, op: affine.Load, args: tuple[Any, ...]
    ):
        memref, *affine_dims = args

        affine_map = op.map

        assert affine_map is not None
        memref = cast(ShapedArray[Any], memref)
        indices = affine_map.data.eval(affine_dims, [])
        indices = tuple(indices)
        value = memref.load(indices)

        return (value,)

    @impl(affine.For)
    def run_for(self, interpreter: Interpreter, op: affine.For, args: tuple[Any, ...]):
        assert not args, "Arguments not supported yet"
        assert not op.results, "Results not supported yet"

        lower_bound = op.lower_bound.data.eval([], [])
        upper_bound = op.upper_bound.data.eval([], [])
        assert len(lower_bound) == 1
        assert len(upper_bound) == 1

        lower_bound = lower_bound[0]
        upper_bound = upper_bound[0]
        step = op.step.value.data

        for i in range(lower_bound, upper_bound, step):
            for_results = interpreter.run_ssacfg_region(op.body, (i,))
            if for_results:
                raise NotImplementedError("affine block results not supported yet")

        return ()

    @impl_terminator(affine.Yield)
    def run_yield(
        self, interpreter: Interpreter, op: affine.Yield, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
