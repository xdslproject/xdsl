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

    @dataclass(frozen=True)
    class ExpandCeilDivUI(Strategy):
        # /// Expands CeilDivUIOp (n, m) into
        # ///  n == 0 ? 0 : ((n-1) / m) + 1
        # This rewrite is actually not 100% correct, as unsigned types are not supported currently
        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.CeilDivUI,
                         operands=[lhs, rhs],
                         results=[IResult(type)]):

                    cst0 = new_cst(0)
                    cst1 = new_cst(1)
                    result = new_op(
                        arith.Select,
                        operands=[
                            new_cmpi("eq", lhs, cst0), cst0,
                            new_bin_op(
                                arith.Addi,
                                new_bin_op(arith.DivUI,
                                           new_bin_op(arith.Subi, lhs, cst1),
                                           rhs), cst1)
                        ],
                        result_types=[type])

                    return success(cst0 + cst1 + result)
                case _:
                    return failure(self)
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
    %3 : !i32 = arith.constant() ["value" = 1 : !i32]
    %4 : !i32 = arith.subi(%0 : !i32, %3 : !i32)
    %5 : !i32 = arith.divui(%4 : !i32, %1 : !i32)
    %6 : !i32 = arith.addi(%5 : !i32, %3 : !i32)
    %7 : !i1 = arith.cmpi(%0 : !i32, %2 : !i32) ["predicate" = 0 : !i64]
    %8 : !i32 = arith.select(%7 : !i1, %2 : !i32, %6 : !i32)
    func.return(%8 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded,
                               topToBottom(ExpandCeilDivUI()))


def test_expand_ceildivsi():
    # https://github.com/llvm/llvm-project/blob/872d69e5d4e251bd70c6cf5dbf1b3ea34f976aa2/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L47

    @dataclass(frozen=True)
    class ExpandCeilDivSI(Strategy):
        # /// Expands CeilDivSIOp (n, m) into
        # ///   1) x = (m > 0) ? -1 : 1
        # ///   2) (n*m>0) ? ((n+x) / m) + 1 : - (-n / m)

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.CeilDivSI,
                         operands=[lhs, rhs],
                         results=[IResult(type)]):
                    cst_0 = new_cst(0)
                    cst_1 = new_cst(1)
                    cst_n1 = new_cst(-1)
                    compare = new_cmpi("sgt", rhs, cst_0)
                    # Compute x = (b>0) ? -1 : 1.
                    x = new_op(arith.Select,
                               operands=[compare, cst_n1, cst_1],
                               result_types=[type])
                    # Compute positive res: 1 + ((x+a)/b).
                    xPlusLhs = new_bin_op(arith.Addi, x, lhs)
                    xPlusLhsDivRhs = new_bin_op(arith.DivSI, xPlusLhs, rhs)
                    posRes = new_bin_op(arith.Addi, cst_1, xPlusLhsDivRhs)
                    # Compute negative res: - ((-a)/b).
                    minusLhs = new_bin_op(arith.Subi, cst_0, lhs)
                    minusLhsDivRhs = new_bin_op(arith.DivSI, minusLhs, rhs)
                    negRes = new_bin_op(arith.Subi, cst_0, minusLhsDivRhs)
                    # Do (a<0 && b<0) || (a>0 && b>0) instead of n*m to avoid overflow
                    lhsNeg = new_cmpi("slt", lhs, cst_0)
                    lhsPos = new_cmpi("sgt", lhs, cst_0)
                    rhsNeg = new_cmpi("slt", rhs, cst_0)
                    rhsPos = new_cmpi("sgt", rhs, cst_0)
                    firstTerm = new_bin_op(arith.AndI, lhsNeg, rhsNeg)
                    secondTerm = new_bin_op(arith.AndI, lhsPos, rhsPos)
                    compareRes = new_bin_op(arith.OrI, firstTerm, secondTerm)
                    result = new_op(arith.Select,
                                    operands=[compareRes, posRes, negRes],
                                    result_types=[type])
                    return success(cst_0 + cst_1 + cst_n1 + result)
                case _:
                    return failure(self)

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
    %2 : !i32 = arith.constant() ["value" = 0 : !i32]
    %3 : !i32 = arith.constant() ["value" = 1 : !i32]
    %4 : !i32 = arith.constant() ["value" = -1 : !i32]
    %5 : !i32 = arith.subi(%2 : !i32, %0 : !i32)
    %6 : !i32 = arith.divsi(%5 : !i32, %1 : !i32)
    %7 : !i32 = arith.subi(%2 : !i32, %6 : !i32)
    %8 : !i1 = arith.cmpi(%1 : !i32, %2 : !i32) ["predicate" = 4 : !i64]
    %9 : !i32 = arith.select(%8 : !i1, %4 : !i32, %3 : !i32)
    %10 : !i32 = arith.addi(%9 : !i32, %0 : !i32)
    %11 : !i32 = arith.divsi(%10 : !i32, %1 : !i32)
    %12 : !i32 = arith.addi(%3 : !i32, %11 : !i32)
    %13 : !i1 = arith.cmpi(%1 : !i32, %2 : !i32) ["predicate" = 4 : !i64]
    %14 : !i1 = arith.cmpi(%0 : !i32, %2 : !i32) ["predicate" = 4 : !i64]
    %15 : !i1 = arith.andi(%14 : !i1, %13 : !i1)
    %16 : !i1 = arith.cmpi(%1 : !i32, %2 : !i32) ["predicate" = 2 : !i64]
    %17 : !i1 = arith.cmpi(%0 : !i32, %2 : !i32) ["predicate" = 2 : !i64]
    %18 : !i1 = arith.andi(%17 : !i1, %16 : !i1)
    %19 : !i1 = arith.ori(%18 : !i1, %15 : !i1)
    %20 : !i32 = arith.select(%19 : !i1, %12 : !i32, %7 : !i32)
    func.return(%20 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded,
                               topToBottom(ExpandCeilDivSI()))


def test_expand_floordivsi():
    # https://github.com/llvm/llvm-project/blob/872d69e5d4e251bd70c6cf5dbf1b3ea34f976aa2/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L102
    @dataclass(frozen=True)
    class ExpandFloorDivSI(Strategy):
        # /// Expands FloorDivSIOp (n, m) into
        # ///   1)  x = (m<0) ? 1 : -1
        # ///   2)  return (n*m<0) ? - ((-n+x) / m) -1 : n / m

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.FloorDivSI,
                         operands=[lhs, rhs],
                         results=[IResult(type)]):
                    cst_0 = new_cst(0)
                    cst_1 = new_cst(1)
                    cst_n1 = new_cst(-1)
                    # Compute x = (b<0) ? 1 : -1.
                    compare = new_cmpi("slt", rhs, cst_0)
                    x = new_op(arith.Select,
                               operands=[compare, cst_1, cst_n1],
                               result_types=[type])
                    # Compute negative res: -1 - ((x-a)/b).
                    xMinusLhs = new_bin_op(arith.Subi, x, lhs)
                    xMinusLhsDivRhs = new_bin_op(arith.DivSI, xMinusLhs, rhs)
                    negRes = new_bin_op(arith.Subi, cst_n1, xMinusLhsDivRhs)
                    # Compute positive res: a/b.
                    posRes = new_bin_op(arith.DivSI, lhs, rhs)
                    # Compute (a>0 && b<0) || (a>0 && b<0) instead of n*m<0
                    lhsNeg = new_cmpi("slt", lhs, cst_0)
                    lhsPos = new_cmpi("sgt", lhs, cst_0)
                    rhsNeg = new_cmpi("slt", rhs, cst_0)
                    rhsPos = new_cmpi("sgt", rhs, cst_0)
                    firstTerm = new_bin_op(arith.AndI, lhsNeg, rhsNeg)
                    secondTerm = new_bin_op(arith.AndI, lhsPos, rhsPos)
                    compareRes = new_bin_op(arith.OrI, firstTerm, secondTerm)
                    result = new_op(arith.Select,
                                    operands=[compareRes, negRes, posRes],
                                    result_types=[type])

                    return success(result)
                case _:
                    return failure(self)

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
    %2 : !i32 = arith.divsi(%0 : !i32, %1 : !i32)
    %3 : !i32 = arith.constant() ["value" = -1 : !i32]
    %4 : !i32 = arith.constant() ["value" = 1 : !i32]
    %5 : !i32 = arith.constant() ["value" = 0 : !i32]
    %6 : !i1 = arith.cmpi(%1 : !i32, %5 : !i32) ["predicate" = 2 : !i64]
    %7 : !i32 = arith.select(%6 : !i1, %4 : !i32, %3 : !i32)
    %8 : !i32 = arith.subi(%7 : !i32, %0 : !i32)
    %9 : !i32 = arith.divsi(%8 : !i32, %1 : !i32)
    %10 : !i32 = arith.subi(%3 : !i32, %9 : !i32)
    %11 : !i1 = arith.cmpi(%1 : !i32, %5 : !i32) ["predicate" = 4 : !i64]
    %12 : !i1 = arith.cmpi(%0 : !i32, %5 : !i32) ["predicate" = 4 : !i64]
    %13 : !i1 = arith.andi(%12 : !i1, %11 : !i1)
    %14 : !i1 = arith.cmpi(%1 : !i32, %5 : !i32) ["predicate" = 2 : !i64]
    %15 : !i1 = arith.cmpi(%0 : !i32, %5 : !i32) ["predicate" = 2 : !i64]
    %16 : !i1 = arith.andi(%15 : !i1, %14 : !i1)
    %17 : !i1 = arith.ori(%16 : !i1, %13 : !i1)
    %18 : !i32 = arith.select(%17 : !i1, %10 : !i32, %2 : !i32)
    func.return(%18 : !i32)
  }
}
"""
    apply_strategy_and_compare(before, expanded,
                               topToBottom(ExpandFloorDivSI()))


class ExpandMaxMinI(Strategy):
    # https://github.com/llvm/llvm-project/blob/872d69e5d4e251bd70c6cf5dbf1b3ea34f976aa2/mlir/lib/Dialect/Arithmetic/Transforms/ExpandOps.cpp#L176

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.MinUI | arith.MinSI | arith.MaxUI
                     | arith.MaxSI,
                     operands=[lhs, rhs],
                     results=[IResult(type)]):
                match op.op_type:
                    case arith.MinUI:
                        pred = "ult"
                    case arith.MinSI:
                        pred = "slt"
                    case arith.MaxUI:
                        pred = "ugt"
                    case arith.MaxSI:
                        pred = "sgt"
                    case _:
                        return failure(self)
                cmp = new_cmpi(pred, lhs, rhs)
                result = new_op(arith.Select,
                                operands=[cmp, lhs, rhs],
                                result_types=[type])
                return success(result)
            case _:
                return failure(self)


def test_expand_minui():
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