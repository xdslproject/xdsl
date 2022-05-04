from __future__ import annotations
from io import StringIO
from optparse import Option
from pprint import pprint
import xdsl.dialects as dialects
import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
import xdsl.dialects.scf as scf
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.elevate import *
from xdsl.immutable_ir import *


def apply_strategy_and_compare(program: str, expected_program: str,
                               strategy: Strategy) -> IOp:
    ctx = MLContext()
    Builtin(ctx)
    Func(ctx)
    Arith(ctx)
    scf.Scf(ctx)

    parser = Parser(ctx, program)
    module: Operation = parser.parse_op()
    imm_module: IOp = get_immutable_copy(module)

    rr = strategy.apply(imm_module)
    assert (rr.isSuccess() and isinstance((resultOp := rr.result[-1]), IOp))

    # for debugging
    printer = Printer()
    printer.print_op(resultOp.get_mutable_copy())

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(resultOp.get_mutable_copy())
    assert file.getvalue().strip() == expected_program.strip()

    return resultOp


@dataclass
class CommuteAdd(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
            case IOp(op_type=arith.Addi,
                     operands=IList([operand0, operand1])):
                b = IBuilder()
                b.from_op(op, operands=[operand1, operand0])
                return success(b)
            case _:
                return failure("CommuteAdd")


@dataclass
class FoldConstantAdd(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
          case IOp(op_type=arith.Addi,
                  operands=IList([IVal(op=IOp(op_type=arith.Constant, 
                                             attributes={"value": IntegerAttr() as attr1}) as c1), 
                                  IVal(op=IOp(op_type=arith.Constant, 
                                             attributes={"value": IntegerAttr() as attr2}))])):
            b = IBuilder()
            b.from_op(c1,
                      attributes={
                          "value":
                          IntegerAttr.from_params(
                              attr1.value.data + attr2.value.data,
                              attr1.typ)
                      })
            return success(b)
          case _:
            return failure("FoldConstantAdd")


@dataclass
class ChangeConstantTo42(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        match op:
          case IOp(op_type=arith.Constant, attributes={"value": IntegerAttr() as attr}):
              b = IBuilder()
              b.from_op(op,
                        attributes={
                            "value":
                            IntegerAttr.from_params(42,
                                                    attr.typ)
                        })
              return success(b)
          case _:
              return failure("ChangeConstant")


def test_double_commute():
    """Tests a strategy which swaps the two operands of an arith.addi."""

    before = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 4 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.constant() ["value" = 1 : !i32]
  %3 : !i32 = arith.addi(%2 : !i32, %1 : !i32)
  %4 : !i32 = arith.addi(%3 : !i32, %0 : !i32)
  func.return(%4 : !i32)
}
"""
    once_commuted = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 2 : !i32]
  %1 : !i32 = arith.constant() ["value" = 1 : !i32]
  %2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
  %3 : !i32 = arith.constant() ["value" = 4 : !i32]
  %4 : !i32 = arith.addi(%3 : !i32, %2 : !i32)
  func.return(%4 : !i32)
}
"""
    apply_strategy_and_compare(program=before,
                               expected_program=once_commuted,
                               strategy=topdown(CommuteAdd()))
    apply_strategy_and_compare(program=once_commuted,
                               expected_program=before,
                               strategy=topdown(CommuteAdd()))
                               
    apply_strategy_and_compare(program=before,
                               expected_program=before,
                               strategy=seq(topdown(CommuteAdd()),
                                            topdown(CommuteAdd())))


def test_commute_block_args():
    """Tests a strategy which swaps the two operands of 
    an arith.addi which are block_args."""

    before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    commuted = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i32, %1 : !i32):
    %2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
    func.return(%2 : !i32)
  }
}
"""
    apply_strategy_and_compare(program=before,
                               expected_program=commuted,
                               strategy=topdown(CommuteAdd()))
    apply_strategy_and_compare(program=commuted,
                               expected_program=before,
                               strategy=topdown(CommuteAdd()))
    newModule = apply_strategy_and_compare(program=before,
                                           expected_program=before,
                                           strategy=seq(
                                               topdown(CommuteAdd()),
                                               topdown(CommuteAdd())))


def test_rewriting_with_blocks():

    # TODO: using this program does acutally lose the if region somehow!
    error = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i1):
    %7 : !i1 = arith.constant() ["value" = 1 : !i1]
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 0 : !i32]
      scf.yield(%2 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""


    before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i1):
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 0 : !i32]
      scf.yield(%2 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""
    constant_changed = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32, !i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0 : !i1):
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 42 : !i32]
      scf.yield(%2 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""

    apply_strategy_and_compare(program=before,
                               expected_program=constant_changed,
                               strategy=topdown(ChangeConstantTo42()))


def test_constant_folding():
    before = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
func.return(%4 : !i32)
}
"""
    once_folded = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 4 : !i32]
  %1 : !i32 = arith.constant() ["value" = 3 : !i32]
  %2 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
  func.return(%2 : !i32)
}
"""
    twice_folded = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 7 : !i32]
  func.return(%0 : !i32)
}
"""

    apply_strategy_and_compare(program=before,
                               expected_program=once_folded,
                               strategy=topdown(FoldConstantAdd()))

    apply_strategy_and_compare(program=once_folded,
                               expected_program=twice_folded,
                               strategy=topdown(FoldConstantAdd()))
                                            
    apply_strategy_and_compare(program=before,
                               expected_program=twice_folded,
                               strategy=seq(topdown(FoldConstantAdd()),
                                            topdown(FoldConstantAdd())))


if __name__ == "__main__":
    test_double_commute()
    test_commute_block_args()
    test_rewriting_with_blocks()
    # TODO: rewriting with successors
    test_constant_folding()