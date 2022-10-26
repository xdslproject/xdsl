from io import StringIO
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.dialects.symref import Symref
from xdsl.ir import MLContext
from xdsl.passes.desymref import DesymrefPass
from xdsl.parser import Parser
from xdsl.printer import Printer

ctx = MLContext()
_builtin = Builtin(ctx)
_func = Func(ctx)
_arith = Arith(ctx)
_affine = Affine(ctx)
_symref = Symref(ctx)
_cf = Cf(ctx)

def verify(prog: str):
    parser = Parser(ctx, prog)
    module = parser.parse_op()
    module.verify()

def desymref(prog: str) -> str:
  parser = Parser(ctx, prog)
  module = parser.parse_op()

  for func in  module.body.ops:
      DesymrefPass.run(func)

  file = StringIO("")
  printer = Printer(stream=file)
  printer.print_op(module)
  return file.getvalue().strip()

def compare_and_verify(prog: str, expected_prog: str):
  actual_prog: str = desymref(prog)
  verify(actual_prog)
  assert actual_prog == expected_prog.strip()


def test_dead_declare():
  prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "test_dead_declare", "function_type" = !fun<[!f32], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !f32):
    symref.declare() ["sym_name" = "dead"]
    func.return(%0 : !f32)
  }
}"""
  expected_prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "test_dead_declare", "function_type" = !fun<[!f32], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !f32):
    func.return(%0 : !f32)
  }
}"""
  compare_and_verify(prog, expected_prog)


def test_dead_updates():
  prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "test_dead_updates", "function_type" = !fun<[!f32, !f32, !f32], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !f32, %1 : !f32, %2 : !f32):
    symref.declare() ["sym_name" = "a"]
    symref.update(%0 : !f32) ["symbol" = @a]
    affine.for() ["lower_bound" = 0 : !index, "upper_bound" = 10 : !index, "step" = 1 : !index] {
    ^1(%3 : !index):
      symref.update(%1 : !f32) ["symbol" = @a]
      affine.for() ["lower_bound" = 0 : !index, "upper_bound" = 5 : !index, "step" = 1 : !index] {
      ^2(%4 : !index):
        symref.update(%2 : !f32) ["symbol" = @a]
      }
    }
    func.return(%0 : !f32)
  }
}"""
  expected_prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "test_dead_updates", "function_type" = !fun<[!f32, !f32, !f32], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !f32, %1 : !f32, %2 : !f32):
    affine.for() ["lower_bound" = 0 : !index, "upper_bound" = 10 : !index, "step" = 1 : !index] {
    ^1(%3 : !index):
      affine.for() ["lower_bound" = 0 : !index, "upper_bound" = 5 : !index, "step" = 1 : !index] {
      ^2(%4 : !index):
      }
    }
    func.return(%0 : !f32)
  }
}"""
  compare_and_verify(prog, expected_prog)


def test_single_block():
  prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "use", "function_type" = !fun<[!f32], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !f32):
    func.return(%0 : !f32)
  }

  func.func() ["sym_name" = "test_single_block", "function_type" = !fun<[!f32, !f32, !f32], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !f32, %1 : !f32, %2 : !f32):
    symref.declare() ["sym_name" = "a"]
    symref.update(%0 : !f32) ["symbol" = @a]
    symref.declare() ["sym_name" = "b"]
    symref.update(%1 : !f32) ["symbol" = @b]
    %3 : !f32 = symref.fetch() ["symbol" = @a]
    %4 : !f32 = func.call(%3 : !f32) ["callee" = @test_single_block]
    symref.update(%4 : !f32) ["symbol" = @b]
    symref.update(%2 : !f32) ["symbol" = @a]
    %5 : !f32 = symref.fetch() ["symbol" = @a]
    symref.update(%5 : !f32) ["symbol" = @b]
    %6 : !f32 = symref.fetch() ["symbol" = @b]
    func.return(%5 : !f32)
  }
}"""
  expected_prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "use", "function_type" = !fun<[!f32], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !f32):
    func.return(%0 : !f32)
  }
  func.func() ["sym_name" = "test_single_block", "function_type" = !fun<[!f32, !f32, !f32], [!f32]>, "sym_visibility" = "private"] {
  ^1(%1 : !f32, %2 : !f32, %3 : !f32):
    %4 : !f32 = func.call(%3 : !f32) ["callee" = @test_single_block]
    func.return(%3 : !f32)
  }
}"""
  compare_and_verify(prog, expected_prog)


def test_single_update():
  prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[!i1], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i1):
    %1: !f32 = arith.constant() ["value" = !float_data<10.0>]
    symref.declare() ["sym_name" = "a"]
    symref.update(%1 : !f32) ["symbol" = @a]
    cf.cond_br(%0 : !i1) (^1, ^2) ["operand_segment_sizes" = !dense<!vector<[3 : !i64], !i32>, [1 : !i32, 0 : !i32, 0 : !i32]>]
  ^1:
    %2 : !f32 = symref.fetch() ["symbol" = @a]
    cf.br(%2 : !f32) (^3)
  ^2:
    %3 : !f32 = symref.fetch() ["symbol" = @a]
    cf.br(%3 : !f32) (^3)
  ^3(%4: !f32):
    func.return(%4 : !f32)
  }
}"""

  expected_prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[!i1], [!f32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i1):
    %1 : !f32 = arith.constant() ["value" = !float_data<10.0>]
    cf.cond_br(%0 : !i1) (^1, ^2) ["operand_segment_sizes" = !dense<!vector<[3 : !i64], !i32>, [1 : !i32, 0 : !i32, 0 : !i32]>]
  ^1:
    cf.br(%1 : !f32) (^3)
  ^2:
    cf.br(%1 : !f32) (^3)
  ^3(%2 : !f32):
    func.return(%2 : !f32)
  }
}"""
  # TODO: we can see that both (and only) predecessors take the same SSA value as block
  # arguments (basically it is a PHI with the same values). We can simplify this further
  # by checking if all predecessors have the same value at that position in arguments.
  compare_and_verify(prog, expected_prog)
