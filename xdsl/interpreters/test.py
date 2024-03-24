from xdsl.dialects import test
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)


@register_impls
class TestFunctions(InterpreterFunctions):
    @impl(test.TestOp)
    def run_test(
        self, interpreter: Interpreter, op: test.TestOp, args: PythonValues
    ) -> PythonValues:
        return (range(len(op.res)),)
