from math import exp

from xdsl.dialects import math
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    impl,
    register_impls,
)


@register_impls
class MathFunctions(InterpreterFunctions):
    @impl(math.ExpOp)
    def run_constant(
        self, interpreter: Interpreter, op: math.ExpOp, args: PythonValues
    ) -> PythonValues:
        (arg,) = args
        return (exp(arg),)
