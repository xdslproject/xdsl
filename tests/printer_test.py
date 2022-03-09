from xdsl.printer import Printer
from xdsl.dialects.arith import *


def test_forgotten_op():
    ctx = MLContext()
    arith = Arith(ctx)

    lit = Constant.from_int_constant(42, 32)
    add = Addi.get(lit, lit)

    add.verify()
    try:
        printer = Printer()
        printer.print_op(add)
    except KeyError:
        return

    assert False, "Exception expected"
