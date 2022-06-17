from __future__ import annotations
from io import StringIO
from xdsl.dialects import memref
import xdsl.dialects.arith as arith
import xdsl.dialects.builtin as builtin
import xdsl.dialects.scf as scf
import xdsl.dialects.affine as affine
import xdsl.dialects.func as func
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
    # constant folding
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

# rewrite is a commutation of 2 unrelated additions. Both of them are root operations in the rewrite
    expected = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.constant() ["value" = 3 : !i32]
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%1 : !i32, %0 : !i32)
%5 : !i32 = arith.addi(%3 : !i32, %2 : !i32)
%6 : !i32 = arith.addi(%4 : !i32, %5 : !i32)
func.return(%6 : !i32)
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


    ctx = MLContext()
    builtin.Builtin(ctx)
    func.Func(ctx)
    arith.Arith(ctx)
    scf.Scf(ctx)
    affine.Affine(ctx)
    memref.MemRef(ctx)

    parser = Parser(ctx, before)
    beforeM: Operation = parser.parse_op()
    immBeforeM: IOp = get_immutable_copy(beforeM)

    print("before:")
    printer = Printer()
    printer.print_op(beforeM)


    # pretty_errors.configure(display_link        = True)
    rrImmM1 = topToBottom(DoubleCommute()).apply(immBeforeM)
    
    # rrImmM1 = topToBottom(DoubleCommute()).apply(immBeforeM)



    # rrImmM1 = backwards(debug() ^ LoopSplit(3)).apply(immBeforeM)

    # rrImmM1 = backwards(debug() ^ LoopUnroll()).apply(rrImmM1.result_op)

    # rrImmM1 = region(block(opsTopToBottom(region(block(opsTopToBottom(CommuteAdd())))))).apply(immBeforeM)

    # rrImmM1 = bottomToTop(debug() ^ CommuteAdd()).apply(immBeforeM)

    # rrImmM1 = region(block(opsTopToBottom(region(block(opsTopToBottom((id() ^ LoopSplit(3)))))))).apply(immBeforeM)
    
    print(rrImmM1)
    
    printer = Printer()
    printer.print_op(rrImmM1.result_op.get_mutable_copy())

    # rrImmM2 = region(block(opsTopToBottom(region(block(opsTopToBottom(debug() ^ LoopUnroll(), skips=1)))))).apply(rrImmM1.result_op)


    rrImmM2 = topToBottom(GarbageCollect()).apply(rrImmM1.result_op)

    # rrImmM2 = skip(backwards() ,isa(affine.For)) ^ LoopUnroll().apply(rrImmM1.result_op)
    printer = Printer()
    printer.print_op(rrImmM2.result_op.get_mutable_copy())

    # rrImmM1 = topdown(debug() ^ fail()).apply(immBeforeM)
    # rrImmM1 = operand_rec(LoopSplit(3)).apply(immBeforeM)
    # print(rrImmM1)
    # assert rrImmM1.isSuccess()

    # printer = Printer()
    # printer.print_op(rrImmM1.result_op.get_mutable_copy())

    # rrImmM2 = operand_rec(debug() ^ skip(isa(affine.For)) ^ LoopUnroll()).apply(rrImmM1.result_op)
    # print(rrImmM2)
    # assert (rrImmM2.isSuccess()
    #         and isinstance(rrImmM2.result_op, IOp))

    # file = StringIO("")
    # printer = Printer(stream=file)
    # printer.print_op(rrImmM2.result_op.get_mutable_copy())

    # For debugging: printing the actual output
    # print("after:")
    # print(file.getvalue().strip())

    # checkDiff = False
    # if checkDiff:
    #     diff = difflib.Differ().compare(file.getvalue().splitlines(True),
    #                                     expected.splitlines(True))
    #     print(''.join(diff))
    #     assert file.getvalue().strip() == expected.strip()


if __name__ == "__main__":
    rewriting_with_immutability_experiments()
