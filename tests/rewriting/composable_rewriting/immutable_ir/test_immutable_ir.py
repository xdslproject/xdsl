import pytest

from xdsl.context import Context
from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.cf import Cf
from xdsl.dialects.func import Func
from xdsl.dialects.test import Test
from xdsl.ir import Operation
from xdsl.parser import Parser
from xdsl.rewriting.composable_rewriting.immutable_ir.immutable_ir import (  # noqa
    get_immutable_copy,
)

program_region = """
"builtin.module"() ({
  %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
}) : () -> ()
"""

program_region_2 = """
"builtin.module"() ({
  %0 = "arith.constant"() <{"value" = 2 : i32}> : () -> i32
}) : () -> ()
"""

program_region_2_diff_name = """
"builtin.module"() ({
  %0 = "arith.constant"() <{"value" = 2 : i32}> : () -> i32
}) : () -> ()
"""

program_region_2_diff_type = """
"builtin.module"() ({
  %0 = "arith.constant"() <{"value" = 2 : i64}> : () -> i64
}) : () -> ()
"""

program_add = """
"builtin.module"() ({
  %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
  %1 = "arith.constant"() <{"value" = 2 : i32}> : () -> i32
  %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
}) : () -> ()
"""

program_add_2 = """
"builtin.module"() ({
  %0 = "arith.constant"() <{"value" = 1 : i32}> : () -> i32
  %1 = "arith.constant"() <{"value" = 2 : i32}> : () -> i32
  %2 = "arith.addi"(%1, %0) : (i32, i32) -> i32
}) : () -> ()
"""

program_func = """
"builtin.module"() ({
  "func.func"() <{"sym_name" = "test", "function_type" = (i32, i32) -> i32, "sym_visibility" = "private"}> ({
  ^0(%0 : i32, %1 : i32):
    %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
    "func.return"(%2) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
"""

program_successors = """
"builtin.module"() ({
  "func.func"() <{"sym_name" = "unconditional_br", "function_type" = () -> (), "sym_visibility" = "private"}> ({
  ^0:
    "cf.br"() [^1] : () -> ()
  ^1:
    "cf.br"() [^0] : () -> ()
  }) : () -> ()
}) : () -> ()
"""

program_attr_and_prop = """
"builtin.module"() ({
  "test.op"() <{"prop1" = i32}> {"attr1" = i64} : () -> ()
}) : () -> ()
"""


@pytest.mark.parametrize(
    "program_str",
    [
        program_region,
        program_region_2,
        program_region_2_diff_type,
        program_region_2_diff_name,
        program_add,
        program_add_2,
        program_func,
        program_successors,
        program_attr_and_prop,
    ],
)
def test_immutable_ir(program_str: str):
    ctx = Context()
    ctx.load_dialect(Test)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)
    ctx.load_dialect(Arith)
    ctx.load_dialect(Cf)

    parser = Parser(ctx, program_str)
    program: Operation = parser.parse_op()
    immutable_program = get_immutable_copy(program)
    mutable_program = immutable_program.to_mutable()

    assert program.is_structurally_equivalent(mutable_program)
