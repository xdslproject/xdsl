from io import StringIO

from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.arith import *
from xdsl.diagnostic import Diagnostic


def test_negative_integer():
    """Test that we can parse negative integers."""
    prog = \
        """module() {
    %0 : !i32 = arith.constant() ["value" = -42 : !i32]
    }"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = -42 : !i32]
}"""

    ctx = MLContext()
    arith = Arith(ctx)
    builtin = Builtin(ctx)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected.strip()
