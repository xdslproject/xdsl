from xdsl.dialects import transform
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl_callable,
    impl_terminator,
    register_impls,
)


@register_impls
class TransformFunctions(InterpreterFunctions):
    @impl_callable(transform.NamedSequenceOp)
    def run_named_sequence_op(
        self,
        interpreter: Interpreter,
        op: transform.NamedSequenceOp,
        args: PythonValues,
    ) -> PythonValues:
        return interpreter.run_ssacfg_region(op.body, args, op.sym_name.data)

    @impl_terminator(transform.YieldOp)
    def run_apply_yield_op(
        self, interpreter: Interpreter, op: transform.YieldOp, args: PythonValues
    ) -> PythonValues:
        # transform.yield does not return anything
        return (), ()
