from __future__ import annotations
from io import StringIO
from optparse import Option
from pprint import pprint

from xdsl.dialects.affine import Affine
from xdsl.dialects.builtin import *
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
from xdsl.printer import Printer
from xdsl.dialects.std import *
from xdsl.dialects.std import Return as stdReturn
from xdsl.dialects.arith import *
from xdsl.dialects.rise.rise import *
from xdsl.dialects.rise.riseBuilder import RiseBuilder
from xdsl.elevate import *
from xdsl.immutable_ir import *

import difflib

###
#
#   In this experiment we use the factored out elevate version of elevate.py
#
###


def rewriting_with_immutability_experiments():
    # constant folding
        before = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 1 : !i32]
%1 : !i32 = arith.constant() ["value" = 2 : !i32]
%2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
%3 : !i32 = arith.constant() ["value" = 4 : !i32]
%4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
std.return(%4 : !i32)
}
"""

    before = \
"""module() {
  builtin.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"]{
  ^0(%arg: !i32):
    %0 : !i32 = arith.constant() ["value" = 0 : !i32]
    %1 : !i32 = arith.constant() ["value" = 1 : !i32]
    %res : !i32 = arith.addi(%0 : !i32, %1 : !i32) 
    std.return(%res : !i32)
  }
}
"""
    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 7 : !i32]
  std.return(%0 : !i32)
}
"""

    @dataclass
    class FoldConstantAdd(Strategy):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (isa(addOp := op, Addi)) and (isa(
                c1 := addOp.operands[0].get_op(), Constant)) and (isa(
                    c2 := addOp.operands[1].get_op(), Constant)):

                assert (isinstance((c1Attr := c1.get_attribute("value")).typ,
                                   IntegerType))
                assert (isinstance((c2Attr := c2.get_attribute("value")).typ,
                                   IntegerType))

                return success(
                    ImmutableOperation.create_new(
                        Constant,
                        result_types=[c1Attr.typ],
                        attributes={
                            "value":
                            IntegerAttr.from_params(
                                c1Attr.value.data + c2Attr.value.data,
                                c1Attr.typ)
                        }))
            else:
                return failure("FoldConstantAdd")

    @dataclass
    class CommuteAdd(Strategy):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (isa(addOp := op, Addi)):
                print("match!")
                return success(
                    ImmutableOperation.create_new(
                        Addi,
                        immutable_operands=[
                            addOp.operands[1], addOp.operands[0]
                        ],
                        result_types=[IntegerType.from_width(32)]))
            else:
                return failure("CommuteAdd")

    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)
    rise = Rise(ctx)
    rise_dsl = RiseBuilder(ctx)
    affine = Affine(ctx)

    parser = Parser(ctx, before)
    beforeM: ModuleOp = parser.parse_op()
    immBeforeM: ImmutableOperation = get_immutable_copy(beforeM)

    # test = topdown(seq(debug(), fail())).apply(immBeforeM)
    # print("before:")
    # printer = Printer()
    # printer.print_op(beforeM)

    rrImmM1 = topdown(FoldConstantAdd()).apply(immBeforeM)
    assert (rrImmM1.isSuccess()
            and isinstance(rrImmM1.result[-1], ImmutableOperation))

    # printer = Printer()
    # printer.print_op(rrImmM1.result[-1]._op)

    rrImmM2 = topdown(FoldConstantAdd()).apply(rrImmM1.result[-1])
    assert (rrImmM2.isSuccess()
            and isinstance(rrImmM2.result[-1], ImmutableOperation))

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(rrImmM2.result[-1]._op)

    # For debugging: printing the actual output
    # print("after:")
    # print(file.getvalue().strip())

    checkDiff = True
    if checkDiff:
        diff = difflib.Differ().compare(file.getvalue().splitlines(True),
                                        expected.splitlines(True))
        print(''.join(diff))
        assert file.getvalue().strip() == expected.strip()


if __name__ == "__main__":
    rewriting_with_immutability_experiments()
