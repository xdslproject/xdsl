from io import StringIO

from xdsl.builder import Builder
from xdsl.dialects import arith
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.print import PrintLnOp
from xdsl.interpreter import Interpreter
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.print import PrintFunctions


def _print(module: ModuleOp) -> str:
    output = StringIO()
    interpreter = Interpreter(module, file=output)
    interpreter.register_implementations(PrintFunctions())
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
        PrintLnOp("hello")

    assert _print(hello) == "hello\n"


def test_print_format():
    @ModuleOp
    @Builder.implicit_region
    def hello():
        one = arith.Constant.from_int_and_width(1, 32).result
        two = arith.Constant.from_int_and_width(2, 32).result
        PrintLnOp("{} {} {}", one, one, two)

    assert _print(hello) == "1 1 2\n"
