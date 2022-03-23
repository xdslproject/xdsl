from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.dialects.arith import *

test_prog = """
module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[], []>, "sym_visibility" = "private"]
  {

    %7 : !i1 = arith.constant() ["value" = 0 : !i1]
    %8 : !i1 = arith.constant() ["value" = 1 : !i1]
    %9 : !i1 = arith.andi(%7 : !i1, %8 : !i1)
    %10 : !i1 = arith.ori(%7 : !i1, %8 : !i1)
    %11 : !i1 = arith.xori(%7 : !i1, %8 : !i1)
  }

  func.func() ["sym_name" = "rec", "function_type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"]
  {
  ^1(%20: !i32):
    %21 : !i32 = func.call(%20 : !i32) ["callee" = @rec] 
    func.return(%21 :!i32)
  }
}
"""


def test_main():
    ctx = MLContext()
    builtin = Builtin(ctx)
    func = Func(ctx)
    arith = Arith(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    printer = Printer()
    printer.print_op(module)
    print()

    print("Done")
