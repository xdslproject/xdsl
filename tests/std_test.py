from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.std import *
from xdsl.dialects.arith import *

test_prog = """
module() {
  builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"]
  {

    %7 : !i1 = arith.constant() ["value" = 0 : !i1]
    %8 : !i1 = arith.constant() ["value" = 1 : !i1]
    %9 : !i1 = std.and(%7 : !i1, %8 : !i1)
    %10 : !i1 = std.or(%7 : !i1, %8 : !i1)
    %11 : !i1 = std.xor(%7 : !i1, %8 : !i1)
  }

  builtin.func() ["sym_name" = "rec", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"]
  {
  ^1(%20: !i32):
    %21 : !i32 = std.call(%20 : !i32) ["callee" = @rec] 
    std.return(%21 :!i32)
  }
}
"""


def test_main():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    printer = Printer()
    printer.print_op(module)
    print()

    print("Done")
