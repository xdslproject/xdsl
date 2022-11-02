
from io import StringIO
from xdsl.dialects.affine import Affine
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.dialects.scf import Scf
from xdsl.dialects.symref import Symref
from xdsl.ir import MLContext
from xdsl.passes.desymref import DesymrefPass
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter

# prog = \
# """
# builtin.module() {
#   func.func() ["sym_name" = "test", "function_type" = !fun<[!si32, !si32, !i1], [!f32]>, "sym_visibility" = "private"] {
#   ^0(%0 : !f32, %1 : !f32, %2: !i1):
#     symref.declare() ["sym_name" = "a"]
#     %3 : !f32 = arith.constant() ["value" = !float_data<0.0>]
#     symref.update(%3 : !f32) ["symbol" = @a]
#     cf.cond_br(%2: !i1)(^1, ^2) 
#   ^1:
#     cf.br()(^3)
#   ^2:
#     %4 : !f32 = arith.constant() ["value" = !float_data<1.0>]
#     symref.update(%4 : !f32) ["symbol" = @a]
#     cf.br()(^3)
#   ^3:
#     %5 : !f32 = symref.fetch() ["symbol" = @a]
#     func.return(%5 : !f32)
#   }
# }"""

# def test(c1, c2, x, y)
#   a = 0
#   b = 10
#   if c1:
#     if c2:
#       a = a + b
#     else:
#       a = x
#     b = a
#   a = x + b

# def test(c1, c2, x, y)
#   a = 0
#   b = 10
#   if c1:
#     if c2:
#       a = a + b
#     else:
#       a = x
#     b = a
#   a = y + b

# prog = \
# """
# builtin.module() {
#   func.func() ["sym_name" = "test", "function_type" = !fun<[!i1, !i32, !i32], [!i32]>, "sym_visibility" = "private"] {
#   ^0(%0 : !i1, %1 : !i1, %2 : !i32, %3 : !i32):

#     // a: i32 = 0
#     symref.declare() ["sym_name" = "a"]
#     %4 : !i32 = arith.constant() ["value" = 0 : !i32]
#     symref.update(%4 : !i32) ["symbol" = @a]

#     // b: i32 = 10
#     symref.declare() ["sym_name" = "b"]
#     %5 : !i32 = arith.constant() ["value" = 10 : !i32]
#     symref.update(%5 : !i32) ["symbol" = @b]

#     // if c1:
#     cf.cond_br(%0: !i1)(^1, ^5) ["operand_segment_sizes" = !dense<!vector<[3 : !i64], !i32>, [1 : !i32, 0 : !i32, 0 : !i32]>]

#   ^1:
#     // if c2: .. else: ..
#     cf.cond_br(%1: !i1)(^2, ^3) ["operand_segment_sizes" = !dense<!vector<[3 : !i64], !i32>, [1 : !i32, 0 : !i32, 0 : !i32]>]

#   ^2:
#     %6 : !i32 = symref.fetch() ["symbol" = @a]
#     %7 : !i32 = symref.fetch() ["symbol" = @b]
#     %8 : !i32 = arith.addi(%6 : !i32, %7 : !i32)
#     symref.update(%8 : !i32) ["symbol" = @a]
#     cf.br() (^4)

#   ^3:
#     symref.update(%2 : !i32) ["symbol" = @a]
#     cf.br() (^4)

#   ^4:
#     %9 : !i32 = symref.fetch() ["symbol" = @a]
#     symref.update(%9 : !i32) ["symbol" = @b]
#     cf.br()(^5)

#   ^5:
#     %10 : !i32 = symref.fetch() ["symbol" = @b]
#     %11 : !i32 = arith.addi(%3 : !i32, %10 : !i32)
#     symref.update(%11 : !i32) ["symbol" = @a]
#     %12 : !i32 = symref.fetch() ["symbol" = @a]
#     func.return(%12 : !i32)
#   }
# }"""

prog1 = \
"""
builtin.module() {
  func.func() ["sym_name" = "test", "function_type" = !fun<[], []>, "sym_visibility" = "private"] {
  ^0:
    symref.declare() ["sym_name" = "0"]
    %w : !i1 = arith.constant() ["value" = 1 : !i1]
    cf.cond_br(%w: !i1)(^1, ^5) ["operand_segment_sizes" = !dense<!vector<[3 : !i64], !i32>, [1 : !i32, 0 : !i32, 0 : !i32]>]

  ^1:
    symref.declare() ["sym_name" = "1"]
    %v : !i1 = arith.constant() ["value" = 0 : !i1]
    cf.cond_br(%v: !i1)(^2, ^3) ["operand_segment_sizes" = !dense<!vector<[3 : !i64], !i32>, [1 : !i32, 0 : !i32, 0 : !i32]>]

  ^2:
    symref.declare() ["sym_name" = "2"]
    cf.br() (^4)

  ^3:
    symref.declare() ["sym_name" = "3"]
    cf.br() (^4)

  ^4:
    symref.declare() ["sym_name" = "4"]
    cf.br()(^5)

  ^5:
    symref.declare() ["sym_name" = "5"]
    func.return()
  }
}"""

prog = \
"""
builtin.module() {
  func.func() ["sym_name" = "foo", "function_type" = !fun<[!i32, !i32, !i32], []>, "sym_visibility" = "private"] {
^0(%0 : !i32, %1 : !i32, %2 : !i32):
  symref.declare() ["sym_name" = "a"]
  symref.update(%0 : !i32) ["symbol" = @a]
  symref.declare() ["sym_name" = "b"]
  symref.update(%1 : !i32) ["symbol" = @b]
  symref.declare() ["sym_name" = "c"]
  symref.update(%2 : !i32) ["symbol" = @c]
  %3 : !i32 = symref.fetch() ["symbol" = @b]
  %4 : !i32 = symref.fetch() ["symbol" = @a]
  %5 : !i1 = arith.cmpi(%4 : !i32, %3 : !i32) ["predicate" = 0 : !i64]
  scf.if(%5 : !i1) {
    %6 : !i32 = symref.fetch() ["symbol" = @b]
    symref.update(%6 : !i32) ["symbol" = @c]
  } {
    %7 : !i32 = symref.fetch() ["symbol" = @c]
    %8 : !i32 = symref.fetch() ["symbol" = @a]
    %9 : !i1 = arith.cmpi(%8 : !i32, %7 : !i32) ["predicate" = 0 : !i64]
    scf.if(%9 : !i1) {
      %10 : !i32 = symref.fetch() ["symbol" = @a]
      symref.update(%10 : !i32) ["symbol" = @c]
    } {
      %11 : !i32 = symref.fetch() ["symbol" = @c]
      symref.update(%11 : !i32) ["symbol" = @c]
    }
  }
}
}"""

ctx = MLContext()
builtin = Builtin(ctx)
func = Func(ctx)
arith = Arith(ctx)
affine = Affine(ctx)
symref = Symref(ctx)
cf = Cf(ctx)
scf = Scf(ctx)

parser = Parser(ctx, prog)
module = parser.parse_op()
rewriter = Rewriter()

for func in  module.body.ops:
    DesymrefPass.run(func)

file = StringIO("")
printer = Printer(stream=file)
printer.print_op(module)
module.verify()
print(file.getvalue().strip())
