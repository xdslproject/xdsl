import pytest

from xdsl.ir import MLContext, Operation
from xdsl.parser import XDSLParser
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.dialects.arith import Arith
from xdsl.dialects.cf import Cf
from xdsl.rewriting.composable_rewriting.immutable_ir.immutable_ir import get_immutable_copy

program_region = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
}
"""
program_region_2 = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 2 : !i32]
}
"""
program_region_2_diff_name = \
"""builtin.module() {
  %cst : !i32 = arith.constant() ["value" = 2 : !i32]
}
"""
program_region_2_diff_type = \
"""builtin.module() {
  %0 : !i64 = arith.constant() ["value" = 2 : !i64]
}
"""
program_add = \
"""builtin.module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
}
"""
program_add_2 = \
"""builtin.module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
}
"""
program_func = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
program_successors = \
"""builtin.module() {
    func.func() ["sym_name" = "unconditional_br", "function_type" = !fun<[], []>, "sym_visibility" = "private"] {
    ^0:
        cf.br() (^1)
    ^1:
        cf.br() (^0)
    }
}
"""


@pytest.mark.parametrize("program_str", [(program_region), (program_region_2),
                                         (program_region_2_diff_type),
                                         (program_region_2_diff_name),
                                         (program_add), (program_add_2),
                                         (program_func), (program_successors)])
def test_immutable_ir(program_str: str):
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Func)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Cf)

    parser = XDSLParser(ctx, program_str)
    program: Operation = parser.parse_op()
    immutable_program = get_immutable_copy(program)
    mutable_program = immutable_program.to_mutable()

    assert program.is_structurally_equivalent(mutable_program)
