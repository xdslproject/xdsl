from xdsl.dialects.builtin import *
from xdsl.parser import Printer, Parser
from xdsl.dialects.std import *
from xdsl.dialects.scf import *

# TODO setup file check tests
test_prog = """
module() {
  builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"]
  {
    %0 : !i1 = std.constant() ["value" = 1 : !i1]
    scf.if(%0 : !i1) 
    {
        %1 : !i32 = std.constant() ["value" = 42 : !i32]
    } 
    {
        %2 : !i32 = std.constant() ["value" = 24 : !i32]
    }
    %5 : !i32 = scf.if(%0 : !i1) 
    {
        %3 : !i32 = std.constant() ["value" = 42 : !i32]
        scf.yield(%3 : !i32)
    } 
    {
        %4 : !i32 = std.constant() ["value" = 24 : !i32]
        scf.yield(%4 : !i32)
    }
  }

  builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"]
  {
    %init : !i32 = std.constant()["value" = 0 : !i32]
    %res : !i32 = scf.while(%init : !i32)
    {
    ^0(%arg : !i32):
      %zero : !i32 = std.constant()["value" = 0 : !i32]
      %c : !i1 = std.cmpi(%zero : !i32, %arg : !i32) ["predicate" = 1 : !i64]
      scf.condition(%c : !i1, %zero : !i32)
    }
    {
    ^1(%arg2 : !i32):
       scf.yield(%arg2 : !i32)
    }
  }
}
"""


def test_scf():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    scf = Scf(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    printer = Printer()
    printer.print_op(module)
    print()

    print("Done")
