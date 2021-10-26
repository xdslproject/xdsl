from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.std import *

# TODO setup file check tests
test_prog = """
module() {
  builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"]
  {
    %0 : !i32 = std.constant() ["value" = 42 : !i32]
    %1 : !i32 = std.constant() ["value" = 42 : !i32]
    %2 : !i32 = std.addi(%0 : !i32, %1 : !i32) 
    %3 : !i32 = std.subi(%0 : !i32, %1 : !i32) 
    %4 : !i32 = std.muli(%0 : !i32, %1 : !i32) 
    %5 : !i32 = std.floordivi_signed(%0 : !i32, %1 : !i32) 
    %6 : !i32 = std.remi_signed(%0 : !i32, %1 : !i32) 

    %7 : !i1 = std.constant() ["value" = 0 : !i1]
    %8 : !i1 = std.constant() ["value" = 1 : !i1]
    %9 : !i1 = std.and(%7 : !i1, %8 : !i1)
    %10 : !i1 = std.or(%7 : !i1, %8 : !i1)
    %11 : !i1 = std.xor(%7 : !i1, %8 : !i1)

    %12 : !i1 = std.cmpi(%1 : !i32, %2 : !i32) ["predicate" = 5 : !i64]
  }

  builtin.func() ["sym_name" = "rec", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"]
  {
  ^1(%20: !i32):
    %21 : !i32 = std.call(%20 : !i32) ["callee" = @rec] 
    std.return(%21 :!i32)
  }

  builtin.func() ["sym_name" = "floats", "type" = !fun<[!f32, !f32], []>, "sym_visibility" = "private"]
  {
  ^2(%30 : !f32, %31 : !f32):
    %32 : !f32 = std.addf(%30 : !f32, %31 : !f32)
    %33 : !f32 = std.mulf(%30 : !f32, %32 : !f32)
  }
}
"""


def test_main():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    printer = Printer()
    printer.print_op(module)
    print()

    print("Done")
