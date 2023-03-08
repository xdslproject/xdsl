import pytest

from conftest import assert_print_op

from xdsl.dialects.arith import Arith
from xdsl.dialects.builtin import Builtin
from xdsl.frontend.passes.desymref import Desymrefier
from xdsl.frontend.symref import Symref
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter


def run_on_prog_and_compare(prog: str, expected_prog: str):
    ctx = MLContext()
    ctx.register_dialect(Arith)
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Symref)

    parser = Parser(ctx, prog)
    desymrefier = Desymrefier(Rewriter())
    op = parser.parse_op()
    desymrefier.desymrefy(op)
    assert_print_op(op, expected_prog, None)


def test_desymrefy_op_no_regions():
    prog = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""
    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""
    run_on_prog_and_compare(prog, expected)


def test_prune_definitions_no_reads_I():
    # a: i32
    prog = \
"""builtin.module() {
  %0 : !i32 = symref.declare() ["sym_name" = "a"]
}"""
    expected = "builtin.module() {}"
    run_on_prog_and_compare(prog, expected)


def test_prune_definitions_no_reads_II():
    # a: i32
    # a = 42
    # a = 11
    # a = 23
    prog = \
"""builtin.module() {
  %0 : !i32 = symref.declare() ["sym_name" = "a"]
  %1 : !i32 = arith.constant() ["value" = 42 : !i32]
  symref.update(%1 : !i32) ["symbol" = @a]
  %2 : !i32 = arith.constant() ["value" = 11 : !i32]
  symref.update(%1 : !i32) ["symbol" = @a]
  %3 : !i32 = arith.constant() ["value" = 23 : !i32]
  symref.update(%1 : !i32) ["symbol" = @a]
}"""
    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.constant() ["value" = 11 : !i32]
  %2 : !i32 = arith.constant() ["value" = 23 : !i32]
}"""
    run_on_prog_and_compare(prog, expected)


def test_prune_definitions_single_write():
    # a: i32
    # a = 42
    #   = a + a
    #   = a * 7
    prog = \
"""builtin.module() {
  %0 : !i32 = symref.declare() ["sym_name" = "a"]
  %1 : !i32 = arith.constant() ["value" = 42 : !i32]
  symref.update(%1 : !i32) ["symbol" = @a]
  %2 : !i32 = symref.fetch() ["symbol" = @a]
  %3 : !i32 = arith.addi(%2 : !i32, %2 : !i32)
  %4 : !i32 = arith.constant() ["value" = 7 : !i32]
  %5 : !i32 = arith.muli(%2 : !i32, %4 : !i32)
}"""
    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
  %2 : !i32 = arith.constant() ["value" = 7 : !i32]
  %3 : !i32 = arith.muli(%0 : !i32, %2 : !i32)
}"""
    run_on_prog_and_compare(prog, expected)


def test_prune_definitions_I():
    # a: i32
    # a = 11
    # b: i32
    # b = 22
    #   = a + b
    prog = \
"""builtin.module() {
  %0 : !i32 = symref.declare() ["sym_name" = "a"]
  %1 : !i32 = arith.constant() ["value" = 11 : !i32]
  symref.update(%1 : !i32) ["symbol" = @a]
  %2 : !i32 = symref.declare() ["sym_name" = "b"]
  %3 : !i32 = arith.constant() ["value" = 22 : !i32]
  symref.update(%3 : !i32) ["symbol" = @b]
  %4 : !i32 = symref.fetch() ["symbol" = @b]
  %5 : !i32 = symref.fetch() ["symbol" = @a]
  %6 : !i32 = arith.addi(%4 : !i32, %5 : !i32)
}"""
    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 11 : !i32]
  %1 : !i32 = arith.constant() ["value" = 22 : !i32]
  %2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
}"""
    run_on_prog_and_compare(prog, expected)


def test_prune_definitions_II():
    # a, b, c: i32
    # a = 0
    # b = 1
    # c = 2
    # a = b + c
    # b = a * b
    # c = b + c
    prog = \
"""builtin.module() {
  %0 : !i32 = symref.declare() ["sym_name" = "a"]
  %1 : !i32 = arith.constant() ["value" = 0 : !i32]
  symref.update(%1 : !i32) ["symbol" = @a]
  %2 : !i32 = symref.declare() ["sym_name" = "b"]
  %3 : !i32 = arith.constant() ["value" = 1 : !i32]
  symref.update(%3 : !i32) ["symbol" = @b]
  %4 : !i32 = symref.declare() ["sym_name" = "c"]
  %5 : !i32 = arith.constant() ["value" = 2 : !i32]
  symref.update(%5 : !i32) ["symbol" = @c]
  %6 : !i32 = symref.fetch() ["symbol" = @b]
  %7 : !i32 = symref.fetch() ["symbol" = @c]
  %8 : !i32 = arith.addi(%6 : !i32, %7 : !i32)
  symref.update(%8 : !i32) ["symbol" = @a]
  %9 : !i32 = symref.fetch() ["symbol" = @a]
  %10 : !i32 = symref.fetch() ["symbol" = @b]
  %11 : !i32 = symref.fetch() ["symbol" = @c]
  %12 : !i32 = arith.muli(%9 : !i32, %10 : !i32)
  symref.update(%12 : !i32) ["symbol" = @b]
  %13 : !i32 = arith.addi(%12 : !i32, %11 : !i32)
  symref.update(%13 : !i32) ["symbol" = @c]
}"""
    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 0 : !i32]
  %1 : !i32 = arith.constant() ["value" = 1 : !i32]
  %2 : !i32 = arith.constant() ["value" = 2 : !i32]
  %3 : !i32 = arith.addi(%1 : !i32, %2 : !i32)
  %4 : !i32 = arith.muli(%3 : !i32, %1 : !i32)
  %5 : !i32 = arith.addi(%4 : !i32, %2 : !i32)
}"""
    run_on_prog_and_compare(prog, expected)
  

def test_prune_unused_reads():
    prog = \
"""builtin.module() {
  %0 : !i32 = symref.fetch() ["symbol" = @a]
  %1 : !i32 = symref.fetch() ["symbol" = @b]
  %2 : !i32 = symref.fetch() ["symbol" = @b]
}"""
    expected = \
    expected = "builtin.module() {}"
    run_on_prog_and_compare(prog, expected)


def test_prune_uses_without_definitions_no_reads():
    prog = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 0 : !i32]
  symref.update(%0 : !i32) ["symbol" = @a]
  %1 : !i32 = arith.constant() ["value" = 1 : !i32]
  symref.update(%1 : !i32) ["symbol" = @a]
  %2 : !i32 = arith.constant() ["value" = 2 : !i32]
  symref.update(%2 : !i32) ["symbol" = @c]
  symref.update(%2 : !i32) ["symbol" = @c]
  symref.update(%2 : !i32) ["symbol" = @c]
}"""
    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 0 : !i32]
  %1 : !i32 = arith.constant() ["value" = 1 : !i32]
  symref.update(%1 : !i32) ["symbol" = @a]
  %2 : !i32 = arith.constant() ["value" = 2 : !i32]
  symref.update(%2 : !i32) ["symbol" = @c]
}"""
    run_on_prog_and_compare(prog, expected)


def test_prune_uses_without_definitions_no_writes():
    prog = \
"""builtin.module() {
  %0 : !i32 = symref.fetch() ["symbol" = @b]
  %1 : !i32 = arith.muli(%0 : !i32, %0 : !i32)
  %2 : !i32 = symref.fetch() ["symbol" = @b]
  %3 : !i32 = arith.constant() ["value" = 5 : !i32]
  %4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
}"""
    expected = \
"""builtin.module() {
  %0 : !i32 = symref.fetch() ["symbol" = @b]
  %1 : !i32 = arith.muli(%0 : !i32, %0 : !i32)
  %2 : !i32 = arith.constant() ["value" = 5 : !i32]
  %3 : !i32 = arith.addi(%0 : !i32, %2 : !i32)
}"""
    run_on_prog_and_compare(prog, expected)


def test_prune_definitions_II():
    # a = d
    # b = 1
    # c = 2
    # a = b + c
    # a = a * b
    # c = b + c
    prog = \
"""builtin.module() {
  %0 : !i32 = symref.fetch() ["symbol" = @d]
  symref.update(%0 : !i32) ["symbol" = @a]

  %1 : !i32 = arith.constant() ["value" = 1 : !i32]
  symref.update(%1 : !i32) ["symbol" = @b]

  %2 : !i32 = arith.constant() ["value" = 2 : !i32]
  symref.update(%2 : !i32) ["symbol" = @c]

  %3 : !i32 = symref.fetch() ["symbol" = @b]
  %4 : !i32 = symref.fetch() ["symbol" = @c]
  %5 : !i32 = arith.addi(%3 : !i32, %4 : !i32)
  symref.update(%5 : !i32) ["symbol" = @a]

  %6 : !i32 = symref.fetch() ["symbol" = @a]
  %7 : !i32 = symref.fetch() ["symbol" = @b]
  %8 : !i32 = arith.muli(%6 : !i32, %7 : !i32)
  symref.update(%8 : !i32) ["symbol" = @a]

  %9 : !i32 = symref.fetch() ["symbol" = @b]
  %10 : !i32 = symref.fetch() ["symbol" = @c]
  %11 : !i32 = arith.addi(%9 : !i32, %10 : !i32)
  symref.update(%11 : !i32) ["symbol" = @c]
}"""
    expected = \
"""builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  symref.update(%0 : !i32) ["symbol" = @b]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
  %3 : !i32 = arith.muli(%2 : !i32, %0 : !i32)
  symref.update(%3 : !i32) ["symbol" = @a]
  %4 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
  symref.update(%4 : !i32) ["symbol" = @c]
}"""
    run_on_prog_and_compare(prog, expected)
