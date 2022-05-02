from __future__ import annotations
from io import StringIO
from optparse import Option
from pprint import pprint

from xdsl.dialects.affine import Affine
from xdsl.dialects.builtin import *
from xdsl.dialects.scf import *
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriteWalker, PatternRewriter, RewritePattern
from xdsl.printer import Printer
from xdsl.dialects.func import *
from xdsl.dialects.func import Return as stdReturn
from xdsl.dialects.arith import *
from xdsl.elevate import *
from xdsl.immutable_ir import *

import difflib

###
#
#   This is a file for prototyping and experimentation. To be removed in a non draft PR
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
func.return(%4 : !i32)
}
"""

    block_args_before = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0: !i32, %1: !i32):
    %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%2 : !i32)
  }
}
"""

    # In current xdsl I have no way to get to the function from the call
    not_possible = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0(%0: !i32, %1: !i32):
    %3 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
    func.return(%3 : !i32)
  }
  %4 : !i32 = arith.constant() ["value" = 0 : !i32]
  %5 : !i32 = arith.constant() ["value" = 1 : !i32]
  %6 : !i32 = func.call(%4 : !i32, %5 : !i32) ["callee" = @test] 
}
"""

    before_scf_if = \
"""module() {
  func.func() ["sym_name" = "test", "type" = !fun<[!i32,!i32], [!i32]>, "sym_visibility" = "private"] {
  ^0():
    %0 : !i1 = arith.constant() ["value" = 1 : !i1]
    %1 : !i32 = scf.if(%0 : !i1) {
      %2 : !i32 = arith.constant() ["value" = 0 : !i32]
      scf.yield(%2 : !i32)
    }
    func.return(%1 : !i32)
  }
}
"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 7 : !i32]
  func.return(%0 : !i32)
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
                return success(
                    ImmutableOperation.create_new(
                        Addi,
                        operands=[addOp.operands[1], addOp.operands[0]],
                        result_types=[IntegerType.from_width(32)]))
            else:
                return failure("CommuteAdd")

    @dataclass
    class ChangeConstant(Strategy):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (isa(op, Constant)):
                return success(
                    ImmutableOperation.create_new(
                        Constant,
                        result_types=[IntegerType.from_width(32)],
                        attributes={
                            "value":
                            IntegerAttr.from_params(42,
                                                    IntegerType.from_width(32))
                        }))
            else:
                return failure("ChangeConstant")

    @dataclass
    class InlineIf(Strategy):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if isa(if_op := op, If) and isinstance(
                if_op.operands[0], ImmutableOpResult) and isa(
                    condition := if_op.operands[0].get_op(),
                    Constant) and (condition.get_attribute("value").value.data
                                   == 1) and isa(
                                       yield_op := if_op.region.block.ops[-1],
                                       Yield) and isinstance(
                                           yield_op.operands[0],
                                           ImmutableOpResult):
                op_to_inline = yield_op.operands[0].get_op
                print("match!!!")

                # TODO: rebuild the operation that is returned via yield
                # maybe build a helper to really just rebuild the op. Here I have to still give all operands

                return success(
                    ImmutableOperation.create_new(
                        op_to_inline._op.__class__, op_to_inline.operands,
                        [result.typ for result in op_to_inline.results],
                        op_to_inline.get_attributes_copy,
                        op_to_inline.successors, op_to_inline.regions))
            else:
                return failure("InlineIf")

    @dataclass
    class AddZero(Strategy):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if len(op.results) > 0 and isinstance(op.result.typ, IntegerType):
                b = ImmutableIRBuiler()
                new_ir = b.op(Addi, [
                    op,
                    b.op(Constant,
                         attributes={
                             "value": IntegerAttr.from_int_and_width(0, 32)
                         },
                         result_types=[op.result.typ])
                ],
                              result_types=[op.result.typ])

                return success(b)
            return failure("AddZero failure")

    ctx = MLContext()
    builtin = Builtin(ctx)
    func = Func(ctx)
    arith = Arith(ctx)
    scf = Scf(ctx)
    affine = Affine(ctx)

    parser = Parser(ctx, before)
    beforeM: Operation = parser.parse_op()
    immBeforeM: ImmutableOperation = get_immutable_copy(beforeM)

    print("before:")
    printer = Printer()
    printer.print_op(beforeM)
    # test = topdown(seq(debug(), fail())).apply(immBeforeM)

    print("mutable_copy:")
    printer = Printer()
    printer.print_op(immBeforeM.get_mutable_copy())

    rrImmM1 = topdown(AddZero()).apply(immBeforeM)
    assert (rrImmM1.isSuccess()
            and isinstance(rrImmM1.result[-1], ImmutableOperation))

    printer = Printer()
    printer.print_op(rrImmM1.result[-1].get_mutable_copy())

    # rrImmM2 = topdown(FoldConstantAdd()).apply(rrImmM1.result[-1])
    # assert (rrImmM2.isSuccess()
    #         and isinstance(rrImmM2.result[-1], ImmutableOperation))

    # file = StringIO("")
    # printer = Printer(stream=file)
    # printer.print_op(rrImmM2.result[-1].get_mutable_copy())

    # For debugging: printing the actual output
    # print("after:")
    # print(file.getvalue().strip())

    checkDiff = False
    if checkDiff:
        diff = difflib.Differ().compare(file.getvalue().splitlines(True),
                                        expected.splitlines(True))
        print(''.join(diff))
        assert file.getvalue().strip() == expected.strip()


if __name__ == "__main__":
    rewriting_with_immutability_experiments()
