# rewrites from:
# expansions, written in C++:
# https://github.com/llvm/llvm-project/blob/872d69e5d4e251bd70c6cf5dbf1b3ea34f976aa2/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp

from __future__ import annotations
from io import StringIO
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
from xdsl.passes.arith_expansion import *
import difflib


def apply_strategy_and_compare(program: str, expected_program: str,
                               strategy: Strategy):
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()
    imm_module: IOp = get_immutable_copy(module)

    rr = strategy.apply(imm_module)
    assert rr.isSuccess()

    # for debugging
    printer = Printer()
    print(f'Result after applying "{strategy}":')
    printer.print_op(rr.result_op.get_mutable_copy())
    print()

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(rr.result_op.get_mutable_copy())

    diff = difflib.Differ().compare(file.getvalue().splitlines(True),
                                    expected_program.splitlines(True))
    if file.getvalue().strip() != expected_program.strip():
        print("Did not get expected output! Diff:")
        print(''.join(diff))
        assert False


def test_expand_ceildivui():
    # https://github.com/llvm/llvm-project/blob/872d69e5d4e251bd70c6cf5dbf1b3ea34f976aa2/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L28

    before = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.ceildivui(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    expanded = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.constant() ["value" = 0 : !i32]
    %3 : !i1 = arith.cmpi(%0 : !i32, %2 : !i32) ["predicate" = 0 : !i64]
    %4 : !i32 = arith.constant() ["value" = 1 : !i32]
    %5 : !i32 = arith.subi(%0 : !i32, %4 : !i32)
    %6 : !i32 = arith.divui(%5 : !i32, %1 : !i32)
    %7 : !i32 = arith.addi(%6 : !i32, %4 : !i32)
    %8 : !i32 = arith.select(%3 : !i1, %2 : !i32, %7 : !i32)
    func.return(%8 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded,
                               topToBottom(ExpandCeilDivUI()))


def test_expand_ceildivsi():
    # https://github.com/llvm/llvm-project/blob/872d69e5d4e251bd70c6cf5dbf1b3ea34f976aa2/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L47

    before = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.ceildivsi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    expanded = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.constant() ["value" = 1 : !i32]
    %3 : !i32 = arith.constant() ["value" = 0 : !i32]
    %4 : !i32 = arith.constant() ["value" = -1 : !i32]
    %5 : !i1 = arith.cmpi(%1 : !i32, %3 : !i32) ["predicate" = 4 : !i64]
    %6 : !i32 = arith.select(%5 : !i1, %4 : !i32, %2 : !i32)
    %7 : !i32 = arith.addi(%6 : !i32, %0 : !i32)
    %8 : !i32 = arith.divsi(%7 : !i32, %1 : !i32)
    %9 : !i32 = arith.addi(%2 : !i32, %8 : !i32)
    %10 : !i32 = arith.subi(%3 : !i32, %0 : !i32)
    %11 : !i32 = arith.divsi(%10 : !i32, %1 : !i32)
    %12 : !i32 = arith.subi(%3 : !i32, %11 : !i32)
    %13 : !i1 = arith.cmpi(%0 : !i32, %3 : !i32) ["predicate" = 2 : !i64]
    %14 : !i1 = arith.cmpi(%0 : !i32, %3 : !i32) ["predicate" = 4 : !i64]
    %15 : !i1 = arith.cmpi(%1 : !i32, %3 : !i32) ["predicate" = 2 : !i64]
    %16 : !i1 = arith.cmpi(%1 : !i32, %3 : !i32) ["predicate" = 4 : !i64]
    %17 : !i1 = arith.andi(%13 : !i1, %15 : !i1)
    %18 : !i1 = arith.andi(%14 : !i1, %16 : !i1)
    %19 : !i1 = arith.ori(%17 : !i1, %18 : !i1)
    %20 : !i32 = arith.select(%19 : !i1, %9 : !i32, %12 : !i32)
    func.return(%20 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded,
                               topToBottom(ExpandCeilDivSI()))


def test_expand_floordivsi():
    # https://github.com/llvm/llvm-project/blob/872d69e5d4e251bd70c6cf5dbf1b3ea34f976aa2/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L102

    before = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.floordivsi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    expanded = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.constant() ["value" = 1 : !i32]
    %3 : !i32 = arith.constant() ["value" = 0 : !i32]
    %4 : !i32 = arith.constant() ["value" = -1 : !i32]
    %5 : !i1 = arith.cmpi(%1 : !i32, %3 : !i32) ["predicate" = 2 : !i64]
    %6 : !i32 = arith.select(%5 : !i1, %2 : !i32, %4 : !i32)
    %7 : !i32 = arith.subi(%6 : !i32, %0 : !i32)
    %8 : !i32 = arith.divsi(%7 : !i32, %1 : !i32)
    %9 : !i32 = arith.subi(%4 : !i32, %8 : !i32)
    %10 : !i32 = arith.divsi(%0 : !i32, %1 : !i32)
    %11 : !i1 = arith.cmpi(%0 : !i32, %3 : !i32) ["predicate" = 2 : !i64]
    %12 : !i1 = arith.cmpi(%0 : !i32, %3 : !i32) ["predicate" = 4 : !i64]
    %13 : !i1 = arith.cmpi(%1 : !i32, %3 : !i32) ["predicate" = 2 : !i64]
    %14 : !i1 = arith.cmpi(%1 : !i32, %3 : !i32) ["predicate" = 4 : !i64]
    %15 : !i1 = arith.andi(%11 : !i1, %14 : !i1)
    %16 : !i1 = arith.andi(%12 : !i1, %13 : !i1)
    %17 : !i1 = arith.ori(%15 : !i1, %16 : !i1)
    %18 : !i32 = arith.select(%17 : !i1, %9 : !i32, %10 : !i32)
    func.return(%18 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded,
                               topToBottom(ExpandFloorDivSI()))


def test_expand_minui():
    # https://github.com/llvm/llvm-project/blob/872d69e5d4e251bd70c6cf5dbf1b3ea34f976aa2/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L176
    before = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.minui(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    expanded = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i1 = arith.cmpi(%0 : !i32, %1 : !i32) ["predicate" = 6 : !i64]
    %3 : !i32 = arith.select(%2 : !i1, %0 : !i32, %1 : !i32)
    func.return(%3 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded, topToBottom(ExpandMaxMinI()))


def test_expand_minsi():
    before = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.minsi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    expanded = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i1 = arith.cmpi(%0 : !i32, %1 : !i32) ["predicate" = 2 : !i64]
    %3 : !i32 = arith.select(%2 : !i1, %0 : !i32, %1 : !i32)
    func.return(%3 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded, topToBottom(ExpandMaxMinI()))


def test_expand_maxui():
    before = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.maxui(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    expanded = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i1 = arith.cmpi(%0 : !i32, %1 : !i32) ["predicate" = 8 : !i64]
    %3 : !i32 = arith.select(%2 : !i1, %0 : !i32, %1 : !i32)
    func.return(%3 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded, topToBottom(ExpandMaxMinI()))


def test_expand_maxsi():
    before = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.maxsi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    expanded = \
"""builtin.module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i1 = arith.cmpi(%0 : !i32, %1 : !i32) ["predicate" = 4 : !i64]
    %3 : !i32 = arith.select(%2 : !i1, %0 : !i32, %1 : !i32)
    func.return(%3 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded, topToBottom(ExpandMaxMinI()))


if __name__ == "__main__":
    test_expand_ceildivui()
    test_expand_ceildivsi()
    test_expand_floordivsi()
    test_expand_minui()
    test_expand_minsi()
    test_expand_maxui()
    test_expand_maxsi()