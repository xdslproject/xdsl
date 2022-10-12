from __future__ import annotations
from io import StringIO
import xdsl.dialects.arith as arith
import xdsl.dialects.scf as scf
import xdsl.dialects.onnx.dialect as onnx
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
    onnx.Onnx(ctx)

    parser = Parser(ctx, program, allow_unregistered_ops=True)
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

###############################################################################
#                                 Rewrites                                    #
###############################################################################

@dataclass(frozen=True)
class DoubleMatMulSameFstInput(Strategy):
    fstMM: IOp

    @dataclass(frozen=True)
    class Match(Matcher):
        def apply(self, op: IOp) -> MatchResult:
            match op:
                case IOp(op_type=onnx.ONNXMatMulOp,
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
            case IOp(op_type=onnx.ONNXMatMulOp,
                    operands=[input0, input1]) if op != self.fstMM and input0 == fstMMInput0:
                axis = new_cst(1)
                axis2 = new_cst(2)
                concat = new_op(onnx.ONNXConcatOp, operands=[input1, fstMMInput1], result_types=[i32]) # i.e. input_2, input_3 in figure
                matmul = new_op(onnx.ONNXMatMulOp, operands=[input0, concat], result_types=[i32])
                split1 = new_op(onnx.ONNXSplitOp, operands=[axis, matmul], result_types=[i32])
                split2 = new_op(onnx.ONNXSplitOp, operands=[axis2, matmul], result_types=[i32])[-2:] 
                # split2 will only contain the splitOp and the constant. 
                # Only possible if the IR of the first replacement is in scope 

                result = success(split1, matched_op=self.fstMM)
                result += success(split2)
                return result
            case _:
                return failure(self)

@dataclass(frozen=True)
class MatchMatMul(Matcher):
    def impl(self, op: IOp) -> MatchResult:
        match op:
            case IOp(op_type=onnx.ONNXMatMulOp,
                    operands=[ISSAValue(), ISSAValue()]):
                return match_success([op])
            case _:
                return match_failure(self)

@dataclass(frozen=True)
class MatchSndMatMulSameLHS(Matcher):
    fstMM: IOp

    def impl(self, op: IOp) -> MatchResult:
        match op:
            case IOp(op_type=onnx.ONNXMatMulOp,
                    operands=[input0, _]) if op != self.fstMM and input0 == self.fstMM.operands[0]:
                return match_success([op])
            case _:
                return match_failure(self)

@dataclass(frozen=True)
class FuseMatMulsSameLHS(Replacer):
    def impl(self, match: Match) -> RewriteResult:
        if len(match) != 2:
            return failure(self)
        fstMM = match[0]
        sndMM = match[1]

        axis = new_cst(1)
        axis2 = new_cst(2)
        concat = new_op(onnx.ONNXConcatOp, operands=[sndMM.operands[1], fstMM.operands[1]], result_types=[i32]) # i.e. input_2, input_3 in figure
        matmul = new_op(onnx.ONNXMatMulOp, operands=[sndMM.operands[0], concat], result_types=[i32])
        split1 = new_op(onnx.ONNXSplitOp, operands=[axis, matmul], result_types=[i32])
        split2 = new_op(onnx.ONNXSplitOp, operands=[axis2, matmul], result_types=[i32])[-2:] 
        # split2 will only contain the splitOp and the constant. 
        # Only possible if the IR of the first replacement is in scope 
        
        result = success(split1, matched_op=fstMM)
        result += success(split2)
        return result


###############################################################################
#                                  Tests                                      #
#   Source: https://github.com/uwplse/tensat/blob/master/multi_cleaned.txt    #
###############################################################################


def test_commute_matmuls():
    """
    Source: (matmul 0 (matmul 0 ?input_4 ?input_5) ?input_4), (matmul 0 ?input_5 (matmul 0 ?input_4 ?input_5))
    Target:
        (matmul 0 ?input_4 (matmul 0 ?input_5 ?input_4)),
        (matmul 0 (matmul 0 ?input_5 ?input_4) ?input_5)
    """

    before = \
"""module() {
%0 : !i32 = onnx.Constant() ["value" = 10 : !i32]       // input4
%1 : !i32 = onnx.Constant() ["value" = 11 : !i32]       // input5
%2 : !i32 = onnx.MatMul(%0 : !i32, %1 : !i32)
%3 : !i32 = onnx.MatMul(%2 : !i32, %0 : !i32)
%4 : !i32 = onnx.MatMul(%1 : !i32, %2 : !i32)
}
"""
    after = \
"""module() {
  %0 : !i32 = onnx.Constant() ["value" = 10 : !i32]
  %1 : !i32 = onnx.Constant() ["value" = 11 : !i32]
  %2 : !i32 = onnx.MatMul(%1 : !i32, %0 : !i32)
  %3 : !i32 = onnx.MatMul(%0 : !i32, %2 : !i32)
  %4 : !i32 = onnx.MatMul(%2 : !i32, %1 : !i32)
}
"""


def test_two_matmul_same_left_input():
    # example from lines 3-4
    """
Source: (matmul ?input1 ?input2 ), (matmul ?input1 ?input3)
Target: 
    (split 0 (split 1 (matmul ?input1 (concat2 1 ?input2 ?input3)))),
    (split 1 (split 1 (matmul ?input1 (concat2 1 ?input2 ?input3))))
"""

    before = \
"""builtin.module() {
%0 : !i32 = onnx.Constant() ["value" = 10 : !i32]       // input1
%1 : !i32 = onnx.Constant() ["value" = 11 : !i32]       // input2
%2 : !i32 = onnx.Constant() ["value" = 12 : !i32]       // input3
%3 : !i32 = onnx.MatMul(%0 : !i32, %1 : !i32)
%4 : !i32 = onnx.MatMul(%0 : !i32, %2 : !i32)
}
"""
    after = \
"""builtin.module() {
  %0 : !i32 = onnx.Constant() ["value" = 10 : !i32]
  %1 : !i32 = onnx.Constant() ["value" = 11 : !i32]
  %2 : !i32 = onnx.Constant() ["value" = 12 : !i32]
  %3 : !i32 = onnx.Concat(%2 : !i32, %1 : !i32)
  %4 : !i32 = onnx.MatMul(%0 : !i32, %3 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = onnx.Split(%5 : !i32, %4 : !i32)
  %7 : !i32 = arith.constant() ["value" = 2 : !i32]
  %8 : !i32 = onnx.Split(%7 : !i32, %4 : !i32)
}
"""
    # old structure
    apply_strategy_and_compare(program=before,
                               expected_program=after,
                               strategy=multiRoot(
        matchTopToBottom(DoubleMatMulSameFstInput.Match()), 
        lambda matched_ops: topToBottom(DoubleMatMulSameFstInput(*matched_ops))))

    # new structure with matcher and replacer
    # TODO: maybe we can find a generic way to express the matching of the second MM
    apply_strategy_and_compare(program=before,
                               expected_program=after,
                               strategy=multiRoot_new(matchSeq(matchTopToBottom(MatchMatMul()), 
                                                            lambda fstMM: matchCombine(fstMM, matchTopToBottom(MatchSndMatMulSameLHS(*fstMM)))), 
                                            FuseMatMulsSameLHS()))




def test_two_matmul_same_right_input():
    # example from lines 5-6
    """
    Source: (matmul 0 ?input_1 ?input_4), (matmul 0 ?input_2 ?input_4)
    Target:
        (split 0 (split 0 (matmul 0 (concat 0 2 ?input_1 ?input_2) ?input_4)))
        (split 1 (split 0 (matmul 0 (concat 0 2 ?input_1 ?input_2) ?input_4)))
    """

    before = \
"""module() {
%0 : !i32 = onnx.Constant() ["value" = 10 : !i32]       // input1
%1 : !i32 = onnx.Constant() ["value" = 11 : !i32]       // input2
%2 : !i32 = onnx.Constant() ["value" = 12 : !i32]       // input4
%3 : !i32 = onnx.MatMul(%0 : !i32, %2 : !i32)
%4 : !i32 = onnx.MatMul(%1 : !i32, %2 : !i32)
}
"""

    after = \
"""module() {
  %0 : !i32 = onnx.Constant() ["value" = 10 : !i32]
  %1 : !i32 = onnx.Constant() ["value" = 11 : !i32]
  %2 : !i32 = onnx.Constant() ["value" = 12 : !i32]
  %3 : !i32 = onnx.concat(%0 : !i32, %1 : !i32)
  %4 : !i32 = onnx.MatMul(%3 : !i32, %2 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = onnx.split(%5 : !i32, %4 : !i32)
  %7 : !i32 = arith.constant() ["value" = 2 : !i32]
  %8 : !i32 = onnx.split(%7 : !i32, %4 : !i32)
}
"""    

def test_two_conv_same_right_input():
    # example from lines 7-8
    """
    Source: (conv2d 1 1 0 0 ?input_7 ?input_10), (conv2d 1 1 0 0 ?input_8 ?input_10)
    Target:
        (split_0 (split 0 (conv2d 1 1 0 0 (concat 0 4 ?input_7 ?input_8) ?input_10)))
        (split_1 (split 0 (conv2d 1 1 0 0 (concat 0 4 ?input_7 ?input_8) ?input_10)))
    """

    # I use the normal onnx.conv because there is no conv2D. Not sure how exactly the
    # rewrite maps to ONNX in the implementation of Tensat

    before = \
"""module() {
%0 : !i32 = onnx.Constant() ["value" = 10 : !i32]       // input7
%1 : !i32 = onnx.Constant() ["value" = 11 : !i32]       // input8
%2 : !i32 = onnx.Constant() ["value" = 12 : !i32]       // input10
%3 : !i32 = onnx.conv(%0 : !i32, %2 : !i32)
%4 : !i32 = onnx.conv(%1 : !i32, %2 : !i32)
}
"""

    after = \
"""module() {
  %0 : !i32 = onnx.Constant() ["value" = 10 : !i32]
  %1 : !i32 = onnx.Constant() ["value" = 11 : !i32]
  %2 : !i32 = onnx.Constant() ["value" = 12 : !i32]
  %3 : !i32 = onnx.concat(%0 : !i32, %1 : !i32)
  %4 : !i32 = onnx.conv(%3 : !i32, %2 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = onnx.split(%5 : !i32, %4 : !i32)
  %7 : !i32 = arith.constant() ["value" = 2 : !i32]
  %8 : !i32 = onnx.split(%7 : !i32, %4 : !i32)
}
"""  

def test_two_conv_same_right_input_relu():
    # example from lines 9-10
    """
    Source: (relu (conv2d 1 1 0 0 ?input_7 ?input_10)), (conv2d 1 1 0 2 ?input_8 ?input_10)
    Target:
        (split_0 (split 0 (conv2d 1 1 0 2 (concat 0 4 ?input_7 ?input_8) ?input_10)))
        (split_1 (split 0 (conv2d 1 1 0 2 (concat 0 4 ?input_7 ?input_8) ?input_10)))
    """

    before = \
"""module() {
%0 : !i32 = onnx.Constant() ["value" = 10 : !i32]       // input7
%1 : !i32 = onnx.Constant() ["value" = 11 : !i32]       // input8
%2 : !i32 = onnx.Constant() ["value" = 12 : !i32]       // input10
%3 : !i32 = onnx.conv(%0 : !i32, %3 : !i32)
%4 : !i32 = onnx.relu(%3 : !i32)
%5 : !i32 = onnx.conv(%1 : !i32, %2 : !i32)
}
"""

    after = \
"""module() {
  %0 : !i32 = onnx.Constant() ["value" = 10 : !i32]
  %1 : !i32 = onnx.Constant() ["value" = 11 : !i32]
  %2 : !i32 = onnx.Constant() ["value" = 12 : !i32]
  %3 : !i32 = onnx.concat(%0 : !i32, %1 : !i32)
  %4 : !i32 = onnx.conv(%3 : !i32, %2 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = onnx.split(%5 : !i32, %4 : !i32)
  %7 : !i32 = arith.constant() ["value" = 2 : !i32]
  %8 : !i32 = onnx.split(%7 : !i32, %4 : !i32)
}
"""  


def test_two_conv_same_left_input():
    # example from lines 11-12
    """
    Source: (conv2d 1 1 0 0 ?input_7 ?input_10), (conv2d 1 1 0 0 ?input_7 ?input_11)
    Target:
        (split_0 (split 1 (conv2d 1 1 0 0 ?input_7 (concat 0 4 ?input_10 ?input_11))))
        (split_1 (split 1 (conv2d 1 1 0 0 ?input_7 (concat 0 4 ?input_10 ?input_11))))
    """

    before = \
"""module() {
%0 : !i32 = onnx.Constant() ["value" = 10 : !i32]       // input7
%1 : !i32 = onnx.Constant() ["value" = 11 : !i32]       // input10
%2 : !i32 = onnx.Constant() ["value" = 12 : !i32]       // input11
%3 : !i32 = onnx.conv(%0 : !i32, %1 : !i32)
%4 : !i32 = onnx.conv(%0 : !i32, %2 : !i32)
}
"""

    after = \
"""module() {
  %0 : !i32 = onnx.Constant() ["value" = 10 : !i32]
  %1 : !i32 = onnx.Constant() ["value" = 11 : !i32]
  %2 : !i32 = onnx.Constant() ["value" = 12 : !i32]
  %3 : !i32 = onnx.concat(%1 : !i32, %2 : !i32)
  %4 : !i32 = onnx.conv(%2 : !i32, %3 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = onnx.split(%5 : !i32, %4 : !i32)
  %7 : !i32 = arith.constant() ["value" = 2 : !i32]
  %8 : !i32 = onnx.split(%7 : !i32, %4 : !i32)
}
"""  

def test_two_conv_same_left_input_relu():
    # example from lines 13-14
    """
    Source: (relu (conv2d 1 1 0 0 ?input_7 ?input_10)), (conv2d 1 1 0 2 ?input_7 ?input_11)
    Target:
        (split_0 (split 1 (conv2d 1 1 0 2 ?input_7 (concat 0 4 ?input_10 ?input_11))))
        (split_1 (split 1 (conv2d 1 1 0 2 ?input_7 (concat 0 4 ?input_10 ?input_11))))
    """

    before = \
"""module() {
%0 : !i32 = onnx.Constant() ["value" = 10 : !i32]       // input7
%1 : !i32 = onnx.Constant() ["value" = 11 : !i32]       // input10
%2 : !i32 = onnx.Constant() ["value" = 12 : !i32]       // input11
%3 : !i32 = onnx.conv(%0 : !i32, %1 : !i32)
%4 : !i32 = onnx.relu(%3 : !i32)
%5 : !i32 = onnx.conv(%0 : !i32, %2 : !i32)
}
"""

    after = \
"""module() {
  %0 : !i32 = onnx.Constant() ["value" = 10 : !i32]
  %1 : !i32 = onnx.Constant() ["value" = 11 : !i32]
  %2 : !i32 = onnx.Constant() ["value" = 12 : !i32]
  %3 : !i32 = onnx.concat(%1 : !i32, %2 : !i32)
  %4 : !i32 = onnx.conv(%2 : !i32, %3 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = onnx.split(%5 : !i32, %4 : !i32)
  %7 : !i32 = arith.constant() ["value" = 2 : !i32]
  %8 : !i32 = onnx.split(%7 : !i32, %4 : !i32)
}
"""  

if __name__ == "__main__":
    test_two_matmul_same_left_input()