from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.dialects.arith import *
from xdsl.dialects.scf import *

# TODO setup file check tests
test_prog = """
module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[], []>, "sym_visibility" = "private"]
  {
    %0 : !i1 = arith.constant() ["value" = 1 : !i1]
    scf.if(%0 : !i1) 
    {
        %1 : !i32 = arith.constant() ["value" = 42 : !i32]
    } 
    {
        %2 : !i32 = arith.constant() ["value" = 24 : !i32]
    }
    %5 : !i32 = scf.if(%0 : !i1) 
    {
        %3 : !i32 = arith.constant() ["value" = 42 : !i32]
        scf.yield(%3 : !i32)
    } 
    {
        %4 : !i32 = arith.constant() ["value" = 24 : !i32]
        scf.yield(%4 : !i32)
    }
  }

  func.func() ["sym_name" = "test", "function_type" = !fun<[], []>, "sym_visibility" = "private"]
  {
    %init : !i32 = arith.constant()["value" = 0 : !i32]
    %res : !i32 = scf.while(%init : !i32)
    {
    ^0(%arg : !i32):
      %zero : !i32 = arith.constant()["value" = 0 : !i32]
      %c : !i1 = arith.cmpi(%zero : !i32, %arg : !i32) ["predicate" = 1 : !i64]
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
    func = Func(ctx)
    arith = Arith(ctx)
    scf = Scf(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    printer = Printer()
    printer.print_op(module)
    print()

    print("Done")
