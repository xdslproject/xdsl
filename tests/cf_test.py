from io import StringIO

from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.cf import *
from xdsl.dialects.std import *
from xdsl.dialects.arith import *


def get_example_cf_program_unconditional_noargs(ctx: MLContext, std: Std,
                                                cf: Cf) -> (Operation, str):

    a = Block()
    b = Block()

    a.add_op(Branch.get(b))
    b.add_op(Branch.get(a))

    f = FuncOp.from_region("test", [], [], [a, b])

    prog = """
builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"] {
^0:
  cf.br() (^1)
^1:
  cf.br() (^0)
}
    """
    return f, prog


def get_example_cf_program_unconditional_args(ctx: MLContext, std: Std,
                                              cf: Cf) -> (Operation, str):

    a = Block.from_arg_types([IntegerType.from_width(32)])
    b = Block.from_arg_types([IntegerType.from_width(32)])

    a.add_op(Branch.get(b, a.args[0]))
    b.add_op(Branch.get(a, b.args[0]))

    f = FuncOp.from_region("test", [], [], [a, b])

    prog = """
builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"] {
^0(%0 : !i32):
  cf.br(%0 : !i32) (^1)
^1(%1 : !i32):
  cf.br(%1 : !i32) (^0)
}
    """
    return f, prog


def get_example_cf_program_conditional_args(ctx: MLContext, std: Std,
                                            cf: Cf) -> (Operation, str):

    a = Block.from_arg_types([IntegerType.from_width(32)])
    b = Block.from_arg_types([IntegerType.from_width(32)])

    a.add_op(Branch.get(b, a.args[0]))
    new_constant_op = Constant.from_int_constant(1, IntegerType.from_width(1))
    b.add_op(new_constant_op)
    b.add_op(
        ConditionalBranch.get(new_constant_op, a, [b.args[0]], b, [b.args[0]]))

    f = FuncOp.from_region("test", [], [], [a, b])

    prog = """
builtin.func() ["sym_name" = "test", "type" = !fun<[], []>, "sym_visibility" = "private"] {
^0(%0 : !i32):
  cf.br(%0 : !i32) (^1)
^1(%1 : !i32):
  %2 : !i1 = arith.constant() ["value" = 1 : !i1]
  cf.cond_br(%2 : !i1, %1 : !i32, %1 : !i32) (^0, ^1) ["operand_segment_sizes" = !dense<!vector<[2 : !index], !i32>, [1 : !i32, 1 : !i32]>]
}
    """
    return f, prog


def test_get():
    ctx = MLContext()
    std = Std(ctx)
    cf = Cf(ctx)

    f, prog = get_example_cf_program_unconditional_noargs(ctx, std, cf)

    f.verify()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(f)
    assert file.getvalue().strip() == prog.strip()

    f, prog = get_example_cf_program_unconditional_args(ctx, std, cf)

    f.verify()
    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(f)
    assert file.getvalue().strip() == prog.strip()

    f, prog = get_example_cf_program_conditional_args(ctx, std, cf)

    f.verify()
    printer = Printer()
    printer.print_op(f)
    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(f)
    assert file.getvalue().strip() == prog.strip()


test_prog = """
module() {
  builtin.func() ["sym_name" = "br", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"]
  {
  ^2(%22: !i32):
    cf.br(%22: !i32)(^2)
  }

  builtin.func() ["sym_name" = "cond_br", "type" = !fun<[!i32], [!i32]>, "sym_visibility" = "private"]
  {
  ^3(%cond : !i1, %arg: !i32):
    cf.cond_br(%cond: !i1, %cond: !i1, %arg : !i32, %arg : !i32, %arg : !i32, %arg : !i32)(^3, ^4) ["operand_segment_sizes" = !dense<!vector<[2 : !i64], !i32>, [2 : !i32, 3 : !i32]>]
  ^4(%24 : !i32, %25 : !i32, %26 : !i32):
    std.return(%24 : !i32)
  }
}
"""


def test_main():
    ctx = MLContext()
    builtin = Builtin(ctx)
    cf = Cf(ctx)
    std = Std(ctx)

    parser = Parser(ctx, test_prog)
    module = parser.parse_op()

    module.verify()
    printer = Printer()
    printer.print_op(module)
