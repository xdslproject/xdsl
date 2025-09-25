from io import StringIO

from xdsl.builder import Builder
from xdsl.dialects import arith
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.printf import PrintFormatOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.printf import PrintfFunctions


def _print(module: ModuleOp) -> str:
    output = StringIO()
    interpreter = Interpreter(module, file=output)
    interpreter.register_implementations(PrintfFunctions())
    interpreter.register_implementations(ArithFunctions())
    interpreter.run_ssacfg_region(module.body, ())
    return output.getvalue()


def test_print_empty():
    @ModuleOp
    @Builder.implicit_region
    def empty():
        pass

    assert _print(empty) == ""


def test_print_constant():
    @ModuleOp
    @Builder.implicit_region
    def hello():
        PrintFormatOp("hello")

    assert _print(hello) == "hello"


def test_print_format():
    @ModuleOp
    @Builder.implicit_region
    def hello():
        one = arith.ConstantOp.from_int_and_width(1, 32).result
        two = arith.ConstantOp.from_int_and_width(2, 32).result
        PrintFormatOp("{} {} {}", one, one, two)

    assert _print(hello) == "1 1 2"
