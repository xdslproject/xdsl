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
import pretty_errors
import difflib

###
#
#   This is not a test. It is a file for prototyping and experimentation. To be removed in a non draft PR
#
###


def rewriting_with_immutability_experiments():
# rewrite is a commutation of 2 unrelated additions. Both of them are root operations in the rewrite
# (at least unrelated in the sense that we do not match them using the op that uses both of them)
    before = \
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
    expected = \
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

        @dataclass(frozen=True)
        class PartialCommute(Strategy):
            fstAdd: IOp
            # normal commute rewrite
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

        def impl(self, op: IOp) -> RewriteResult:
            match op:
                case IOp(op_type=arith.Addi,
                        operands=[ISSAValue(), ISSAValue()]):
                    return self.PartialCommute(op)
                case _:
                    return failure(self)


    @dataclass(frozen=True)
    class DoubleMatMulSameFstInput_(Strategy):

        @dataclass(frozen=True)
        class Partial(Strategy):
            fstMM: IOp
            fstMMInput0: ISSAValue
            fstMMInput1: ISSAValue

            # normal commute rewrite
            def impl(self, op: IOp) -> RewriteResult:
                match op:
                    case IOp(op_type=tensat.MatMul,
                            operands=[input0, input1]) if op != self.fstMM and input0 == self.fstMMInput0:
                        #Shared stuff:
                        axis = new_cst(1)
                        axis2 = new_cst(2)
                        concat = new_op(tensat.Concat, operands=[input1, self.fstMMInput1], result_types=[i32]) # i.e. input_2, input_3 in figure
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

        def impl(self, op: IOp) -> Strategy:
            match op:
                case IOp(op_type=tensat.MatMul,
                        operands=[input0, input1]):
                    return self.Partial(op, input0, input1)
                case _:
                    return failure(self)



    @dataclass(frozen=True)
    class DoubleMatMulSameFstInput(Strategy):
        fstMM: IOp

        @dataclass(frozen=True)
        class Partial(Strategy):
            def apply(self, op: IOp) -> RewriteResult:
                match op:
                    case IOp(op_type=tensat.MatMul,
                            operands=[ISSAValue(), ISSAValue()]):
                        return id().apply(op)
                        # DoubleMatMulSameFstInput(op, input0, input1)
                    case _:
                        return failure(self)

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


    ctx = MLContext()
    builtin.Builtin(ctx)
    func.Func(ctx)
    arith.Arith(ctx)
    scf.Scf(ctx)
    affine.Affine(ctx)
    memref.MemRef(ctx)
    tensat.Tensat(ctx)

    parser = Parser(ctx, before_tensat_double_matmul)
    beforeM: Operation = parser.parse_op()
    immBeforeM: IOp = get_immutable_copy(beforeM)

    print("before:")
    printer = Printer()
    printer.print_op(beforeM)

    # DoubleCommute of two unrelated additions
    # rrImmM1 = topToBottom_hacked(DoubleCommute()).apply(immBeforeM)

    # Rewrite two matmuls which share the same first input
    # rrImmM1 = topToBottom_hacked(DoubleMatMulSameFstInput_()).apply(immBeforeM)

    # Rewrite two matmuls which share the same first input with better control
    rrImmM1 = multi_seq2(
        topToBottom_non_rebuilding(DoubleMatMulSameFstInput.Partial()), 
        lambda ops: topToBottom(DoubleMatMulSameFstInput(ops[0]))).apply(immBeforeM)


    print(rrImmM1)
    
    printer = Printer()
    printer.print_op(rrImmM1.result_op.get_mutable_copy())

if __name__ == "__main__":
    rewriting_with_immutability_experiments()
