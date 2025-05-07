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
    @impl(affine.StoreOp)
    def run_store(
        self, interpreter: Interpreter, op: affine.StoreOp, args: tuple[Any, ...]
    ):
        value, memref, *affine_dims = args

        affine_map = op.map

        assert affine_map is not None
        memref = cast(ShapedArray[Any], memref)
        indices = affine_map.data.eval(affine_dims, [])

        indices = tuple(indices)
        memref.store(indices, value)

        return ()

    @impl(affine.LoadOp)
    def run_load(
        self, interpreter: Interpreter, op: affine.LoadOp, args: tuple[Any, ...]
    ):
        memref, *affine_dims = args

        affine_map = op.map

        assert affine_map is not None
        memref = cast(ShapedArray[Any], memref)
        indices = affine_map.data.eval(affine_dims, [])
        indices = tuple(indices)
        value = memref.load(indices)

        return (value,)

    @impl(affine.ForOp)
    def run_for(
        self, interpreter: Interpreter, op: affine.ForOp, args: tuple[Any, ...]
    ):
        assert not args, "Arguments not supported yet"
        assert not op.results, "Results not supported yet"

        lower_bound = op.lowerBoundMap.data.eval([], [])
        upper_bound = op.upperBoundMap.data.eval([], [])
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

    @impl(affine.ApplyOp)
    def run_apply(
        self, interpreter: Interpreter, op: affine.ApplyOp, args: tuple[Any, ...]
    ):
        return op.map.data.eval(args, ())

    @impl_terminator(affine.YieldOp)
    def run_yield(
        self, interpreter: Interpreter, op: affine.YieldOp, args: tuple[Any, ...]
    ):
        return ReturnedValues(args), ()
