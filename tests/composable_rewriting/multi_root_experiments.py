from __future__ import annotations
from io import StringIO
from xdsl.dialects import memref
import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
import xdsl.dialects.scf as scf
import xdsl.dialects.affine as affine
import xdsl.dialects.func as func
import xdsl.dialects.tensat as tensat
from xdsl.parser import Parser
from xdsl.printer import Printer

from xdsl.elevate import *
import xdsl.elevate as elevate
from xdsl.immutable_ir import *
from xdsl.immutable_utils import *
import difflib

###
#
#   This is not a test. It is a file for prototyping and experimentation. To be removed in a non draft PR
#
###


def rewriting_with_immutability_experiments():
# rewrite is a commutation of 2 unrelated additions. Both of them are root operations in the rewrite
# (at least unrelated in the sense that we do not match them using the op that uses both of them)
    before_commute_additions = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.constant() ["value" = 3 : !i32]
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
%5 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
%6 : !i32 = arith.addi(%4 : !i32, %5 : !i32)
func.return(%6 : !i32)
}
"""
    expected_commute_additions = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.constant() ["value" = 3 : !i32]
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%1 : !i32, %0 : !i32)        // root0
%5 : !i32 = arith.addi(%3 : !i32, %2 : !i32)        // root1
%6 : !i32 = arith.addi(%4 : !i32, %5 : !i32)
func.return(%6 : !i32)
}
"""


# Source: (matmul ?input1 ?input2 ), (matmul ?input1 ?input3)
# Target: 
# (split0 (split 1 (matmul ?input1 (concat2 1 ?input2 ?input3)))),
# (split1 (split 1 (matmul ?input1 (concat2 1 ?input2 ?input3))))
    before_tensat_double_matmul = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 10 : !i32]       // input1
%1 : !i32 = arith.constant() ["value" = 11 : !i32]       // input2
%2 : !i32 = arith.constant() ["value" = 12 : !i32]       // input3
%3 : !i32 = tensat.matmul(%0 : !i32, %1 : !i32)
%4 : !i32 = tensat.matmul(%0 : !i32, %2 : !i32)
%5 : !i32 = arith.addi(%3 : !i32, %4 : !i32)
func.return(%5 : !i32)
}
"""
    expected_tensat_double_matmul = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 10 : !i32]
  %1 : !i32 = arith.constant() ["value" = 11 : !i32]
  %2 : !i32 = arith.constant() ["value" = 12 : !i32]
  %3 : !i32 = tensat.concat(%2 : !i32, %1 : !i32)
  %4 : !i32 = tensat.matmul(%0 : !i32, %3 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = tensat.split(%5 : !i32, %4 : !i32)
  %7 : !i32 = arith.constant() ["value" = 2 : !i32]
  %8 : !i32 = tensat.split(%7 : !i32, %4 : !i32)
  %9 : !i32 = arith.addi(%6 : !i32, %8 : !i32)
  func.return(%9 : !i32)
}
"""


    @dataclass(frozen=True)
    class DoubleCommute(Strategy):
        fstAdd: IOp

        @dataclass(frozen=True)
        class Match(Matcher):

            def impl(self, op: IOp) -> MatchResult:
                match op:
                    case IOp(op_type=arith.Addi,
                            operands=[ISSAValue(), ISSAValue()]):
                        return match_success([op])
                    case _:
                        return match_failure(self)

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.Addi,
                        operands=[operand0, operand1]) if op != self.fstAdd:
                    # Commuting both additions
                    result = success(from_op(self.fstAdd, operands=[self.fstAdd.operands[1], self.fstAdd.operands[0]]), matched_op=self.fstAdd)
                    result += success(from_op(op, operands=[operand1, operand0]))
                    return result
                case _:
                    return failure(self)

    @dataclass(frozen=True)
    class TripleCommute(Strategy):
        fstAdd: IOp
        sndAdd: IOp

        @dataclass(frozen=True)
        class Match0(Matcher):

            def impl(self, op: IOp) -> MatchResult:
                match op:
                    case IOp(op_type=arith.Addi,
                            operands=[ISSAValue(), ISSAValue()]):
                        return match_success([op])
                    case _:
                        return match_failure(self)

        @dataclass(frozen=True)
        class Match1(Matcher):
            fstAdd: IOp

            def impl(self, op: IOp) -> MatchResult:
                match op:
                    case IOp(op_type=arith.Addi,
                            operands=[ISSAValue(), ISSAValue()]) if op != self.fstAdd:
                        return match_success([self.fstAdd, op])
                    case _:
                        return match_failure(self)

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.Addi,
                        operands=[operand0, operand1]) if op != self.fstAdd and  op != self.sndAdd:
                    
                    # This success has to be at the top or the rewrite fails. Why is that?
                    result = success(from_op(op, operands=[operand1, operand0]))
                    result += success(from_op(self.fstAdd, operands=[self.fstAdd.operands[1], self.fstAdd.operands[0]]), matched_op=self.fstAdd)
                    result += success(from_op(self.sndAdd, operands=[self.sndAdd.operands[1], self.sndAdd.operands[0]]), matched_op=self.sndAdd)
                    return result
                case _:
                    return failure(self)

    @dataclass(frozen=True)
    class DoubleMatMulSameFstInput(Strategy):
        fstMM: IOp

        @dataclass(frozen=True)
        class Match(Matcher):
            def apply(self, op: IOp) -> MatchResult:
                match op:
                    case IOp(op_type=tensat.MatMul,
                            operands=[ISSAValue(), ISSAValue()]):
                        # can we somehow enforce that only existing operations can be used in a match?
                        return match_success([op])
                    case _:
                        return match_failure(self)

        def impl(self, op: IOp) -> RewriteResult:
            assert len(self.fstMM.operands) == 2
            fstMMInput0: ISSAValue = self.fstMM.operands[0]
            fstMMInput1: ISSAValue = self.fstMM.operands[1]
            match op:
                case IOp(op_type=tensat.MatMul,
                        operands=[input0, input1]) if op != self.fstMM and input0 == fstMMInput0:
                    axis = new_cst(1)
                    axis2 = new_cst(2)
                    concat = new_op(tensat.Concat, operands=[input1, fstMMInput1], result_types=[i32]) # i.e. input_2, input_3 in figure
                    matmul = new_op(tensat.MatMul, operands=[input0, concat], result_types=[i32])
                    split1 = new_op(tensat.Split, operands=[axis, matmul], result_types=[i32])
                    split2 = new_op(tensat.Split, operands=[axis2, matmul], result_types=[i32])[-2:] 
                    # split2 will only contain the splitOp and the constant. 
                    # Only possible if the IR of the first replacement is in scope 

                    result = success(split1, matched_op=self.fstMM)
                    result += success(split2)
                    return result
                case _:
                    return failure(self)

    @dataclass(frozen=True)
    class MatchAdd(Matcher):
        def impl(self, op: IOp) -> MatchResult:
            match op:
                case IOp(op_type=arith.Addi,
                        operands=[ISSAValue(), ISSAValue()]):
                    return match_success([op])
                case _:
                    return match_failure(self)

    @dataclass(frozen=True)
    class Commute(Replacer):
        def impl(self, match: Match) -> RewriteResult:
            # Has to be in this order because there is a bug somewhere in the handling of replacements. Ideally it should not make a difference
            result: RewriteResult = success(from_op(match[-1], operands=[match[-1].operands[1], match[-1].operands[0]]), matched_op=match[-1])
            for op in reversed(match[:-1]):
                result += success(from_op(op, operands=[op.operands[1], op.operands[0]]), matched_op=op)
            return result

    ctx = MLContext()
    builtin.Builtin(ctx)
    func.Func(ctx)
    arith.Arith(ctx)
    scf.Scf(ctx)
    affine.Affine(ctx)
    memref.MemRef(ctx)
    tensat.Tensat(ctx)

    parser = Parser(ctx, before_commute_additions)
    beforeM: Operation = parser.parse_op()
    immBeforeM: IOp = get_immutable_copy(beforeM)

    print("before:")
    printer = Printer()
    printer.print_op(beforeM)

    # DoubleCommute of two additions. To use parse the string "before_additions"
    # rrImmM1 = multi_seq(matchTopToBottom(DoubleCommute.Match()), 
    #     lambda matched_ops: topToBottom(DoubleCommute(*matched_ops))).apply(immBeforeM)
    
    # TripleCommute of three additions. To use parse the string "before_additions"
    # rrImmM1 = multiRoot(matchSeq(matchTopToBottom(TripleCommute.Match0()), 
    #                                 lambda matched_ops: matchTopToBottom(TripleCommute.Match1(*matched_ops))),
    #                                 lambda matched_ops: topToBottom(TripleCommute(*matched_ops))).apply(immBeforeM)

    # New design with Replacer:
    # SingleCommute
    # rrImmM1 = multiRoot_new(matchTopToBottom(MatchAdd()), Commute()).apply(immBeforeM)

    # DoubleCommute
    # rrImmM1 = multiRoot_new(
    #                 matchSeq(matchTopToBottom(MatchAdd()), 
    #                          lambda fst_add: matchCombine(fst_add, matchTopToBottom(matchNeq(MatchAdd(), fst_add)))), 
    #                 Commute()).apply(immBeforeM)

    # TripleCommute
    rrImmM1 = multiRoot_new(
                matchSeq(matchTopToBottom(MatchAdd()), 
                    lambda fst_add: matchCombine(fst_add, 
                        matchSeq(matchTopToBottom(matchNeq(MatchAdd(), fst_add)), 
                            lambda snd_add: matchCombine(snd_add, matchTopToBottom(matchNeq(matchNeq(MatchAdd(), fst_add), snd_add)))))), 
                Commute()).apply(immBeforeM)

    # rrImmM1 = multi_seq(
    #     matchTopToBottom(DoubleMatMulSameFstInput.Match()), 
    #     lambda matched_ops: topToBottom(DoubleMatMulSameFstInput(*matched_ops))).apply(immBeforeM)


    print(rrImmM1)
    
    printer = Printer()
    printer.print_op(rrImmM1.result_op.get_mutable_copy())

if __name__ == "__main__":
    rewriting_with_immutability_experiments()
