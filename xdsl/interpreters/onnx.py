from xdsl.dialects import onnx
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    ReturnedValues,
    impl,
    register_impls,
)


@register_impls
class OnnxFunctions(InterpreterFunctions):
    @impl(onnx.Add)
    def run_add(self, interpreter: Interpreter, op: onnx.Add, args: PythonValues):
        return ReturnedValues(args), ()

    @impl(onnx.Sub)
    def run_sub(self, interpreter: Interpreter, op: onnx.Sub, args: PythonValues):
        return ReturnedValues(args), ()

    @impl(onnx.Mul)
    def run_mul(self, interpreter: Interpreter, op: onnx.Mul, args: PythonValues):
        return ReturnedValues(args), ()

    @impl(onnx.Div)
    def run_div(self, interpreter: Interpreter, op: onnx.Div, args: PythonValues):
        return ReturnedValues(args), ()

    @impl(onnx.Relu)
    def run_relu(self, interpreter: Interpreter, op: onnx.Relu, args: PythonValues):
        return ReturnedValues(args), ()
