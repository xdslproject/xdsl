from math import exp, sqrt

from xdsl.dialects import math
from xdsl.dialects.builtin import AnyFloat, FloatAttr
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
    def run_exp(
        self, interpreter: Interpreter, op: math.ExpOp, args: PythonValues
    ) -> PythonValues:
        (arg,) = args
        assert isinstance(op.result.type, AnyFloat)
        return (FloatAttr(exp(arg.value.data), op.result.type),)

    @impl(math.SqrtOp)
    def run_sqrt(
        self, interpreter: Interpreter, op: math.SqrtOp, args: PythonValues
    ) -> PythonValues:
        (arg,) = args
        assert isinstance(op.result.type, AnyFloat)
        return (FloatAttr(sqrt(arg.value.data), op.result.type),)

    @impl(math.LogOp)
    def run_log(
        self, interpreter: Interpreter, op: math.LogOp, args: PythonValues
    ) -> PythonValues:
        (arg,) = args
        from math import log

        assert isinstance(op.result.type, AnyFloat)
        return (FloatAttr(log(arg.value.data), op.result.type),)
