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

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 7 : !i32]
  std.return(%0 : !i32)
}
"""

    expected_without_garbage_collection = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  %2 : !i32 = arith.constant() ["value" = 3 : !i32]
  %3 : !i32 = arith.constant() ["value" = 4 : !i32]
  %4 : !i32 = arith.constant() ["value" = 7 : !i32]
  std.return(%4 : !i32)
}
"""

    @dataclass
    class ImmutableRewrite:

        def apply(self, op: ImmutableOperation) -> RewriteResult:
            clonedOp = op._op.clone()
            clonedImmutableOp = get_immutable_copy(clonedOp)

            result = self.impl(clonedImmutableOp)
            if isinstance(result.result, str):
                return result

            # collectGarbage = False
            # if collectGarbage:
            #     rewriter = PatternRewriter(result.result[0])
            #     garbageCandidates = []
            #     garbageCandidates.extend([
            #         operand.op for operand in op._op.operands
            #         if isinstance(operand, OpResult)
            #     ])

            #     # We do not want to mutate the existing IR, but it is nice for checking validity
            #     doReplacement = False
            #     if doReplacement:
            #         rewriter.replace_matched_op(
            #             [result._op for result in result.result])

            #     for op in garbageCandidates:
            #         if all(len(result.uses) == 0 for result in op.results):
            #             rewriter.erase_op(op)

            return result

        @abstractmethod
        def impl(self, op: ImmutableOperation) -> RewriteResult:
            ...

    # This rewrite matches a module containing an addOp
    @dataclass
    class FoldConstantAddInModule(ImmutableRewrite):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (isa(module := op, ModuleOp)):
                addOp = None

                for op in module.region.block.ops:
                    if isa(op, Addi):
                        addOp = op
                        break

                if isa(c1 := addOp.operands[0].op, Constant) and isa(
                        c2 := addOp.operands[1].op, Constant):

                    assert (isinstance(
                        c1.get_attribute("value").typ, IntegerType))
                    assert (isinstance(
                        c2.get_attribute("value").typ, IntegerType))

                    c1Val = c1.get_attribute("value")
                    c2Val = c2.get_attribute("value")

                    newConstant = Constant.from_int_constant(
                        c1Val.value.data + c2Val.value.data, c1Val.typ)

                    #TODO: we do mutating changes here
                    rewriter = PatternRewriter(addOp._op)
                    rewriter.replace_matched_op(newConstant)

                    return success(
                        [ImmutableOperation.from_op(module._op, {})])
            return failure("FoldConstantAddInModule")

    def get_immutable_copy(op: Operation) -> ImmutableOperation:
        return ImmutableOperation.from_op(op, {})

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

    rrImmM1 = FoldConstantAddInModule().apply(immBeforeM)
    assert (rrImmM1.isSuccess()
            and isinstance(rrImmM1.result[0], ImmutableOperation))

    rrImmM2 = FoldConstantAddInModule().apply(rrImmM1.result[0])
    assert (rrImmM2.isSuccess()
            and isinstance(rrImmM2.result[0], ImmutableOperation))

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(rrImmM2.result[0].get_mutable_copy())

    diff = list(difflib.Differ().compare(
        file.getvalue().splitlines(True),
        expected_without_garbage_collection.splitlines(True)))
    print(''.join(diff))

    # For debugging: printing the actual output
    # print(file.getvalue().strip())

    assert file.getvalue().strip(
    ) == expected_without_garbage_collection.strip()


if __name__ == "__main__":
    rewriting_with_immutability_experiments()