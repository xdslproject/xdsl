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
#   In this experiment we use the concepts of Elevate strategies to traverse the IR
#   and enable that the FoldConstants rewrite actually matches the addOp and not the
#   module containing the addop.
#   We also clone the op we apply a Rewrite to before rewriting. Due to issues with this
#   it is difficult to define the topDown traversal. To combat this the one traversal is
#   defined to not only try to apply to the operands but also to the operations in the region
#   of an op
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
            print("cloning op:" + op.name)
            # This does not always work as wanted, bcs we do not have access to the operands of clonedOp anymore
            # Maybe implement a custom clone() for ImmutableOperation?

            clonedOp = op.get_mutable_copy()
            clonedImmutableOp = get_immutable_copy(clonedOp)

            return self.impl(clonedImmutableOp)

        @abstractmethod
        def impl(self, op: ImmutableOperation) -> RewriteResult:
            ...

    @dataclass
    class FoldConstantAdd(ImmutableRewrite):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (isa(addOp := op, Addi)) and (isa(
                    c1 := addOp.operands[0].op, Constant)) and (isa(
                        c2 := addOp.operands[1].op, Constant)):

                assert (isinstance(c1.get_attribute("value").typ, IntegerType))
                assert (isinstance(c2.get_attribute("value").typ, IntegerType))

                c1Val = c1.get_attribute("value")
                c2Val = c2.get_attribute("value")
                return success([
                    ImmutableOperation.from_op(
                        Constant.from_int_constant(
                            c1Val.value.data + c2Val.value.data, c1Val.typ),
                        {})
                ])
            else:
                return failure("FoldConstantAdd")

    @dataclass
    class ChangeConstantVal(ImmutableRewrite):
        value: int

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            assert (isinstance(op, ImmutableOperation))
            if isa(op, Constant):
                return success([
                    ImmutableOperation.from_op(
                        Constant.from_int_constant(
                            self.value,
                            op.get_attribute("value").typ))
                ])
            return failure("ChangeConstantVal")

    @dataclass
    class id(ImmutableRewrite):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            assert (isinstance(op, ImmutableOperation))
            print("id")
            return success([op])

    @dataclass
    class fail(ImmutableRewrite):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            assert (isinstance(op, ImmutableOperation))
            return failure("fail Strategy")

    @dataclass
    class debug(ImmutableRewrite):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            printer = Printer()
            printer.print_op(op._op)
            return success([op])

    @dataclass
    class seq(ImmutableRewrite):
        s1: ImmutableRewrite
        s2: ImmutableRewrite

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            rr = self.s1.apply(op)
            return rr.flatMapSuccess_imm(self.s2)

    @dataclass
    class leftChoice(ImmutableRewrite):
        s1: ImmutableRewrite
        s2: ImmutableRewrite

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            return self.s1.apply(op).flatMapFailure(lambda: self.s2.apply(op))

    @dataclass
    class try_(ImmutableRewrite):
        s: ImmutableRewrite

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            return leftChoice(self.s, id()).apply(op)

    @dataclass
    class one(ImmutableRewrite):
        """
        Try to apply s to one the operands of op or to the next operation in the same block.
        """
        s: ImmutableRewrite

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            # TODO: handle properly for regions and blocks
            if isa(module := op, ModuleOp):
                for nestedOp in module.region.block.ops:
                    rr = self.s.apply(nestedOp)
                    if rr.isSuccess():
                        #TODO: here we escape from Imm
                        nestedOp.replace_with(rr.result)
                        return success(
                            [ImmutableOperation.from_op(module._op)])
            for operand in op.operands:
                if (isinstance(operand, ImmutableOpResult)):
                    rr = self.s.apply(operand.op)
                    if rr.isSuccess():
                        #TODO: here we escape from Imm
                        operand.op.replace_with(rr.result)
                        return success([ImmutableOperation.from_op(op._op)]
                                       | rr.result)
            # This was for also applying this to the operation after this op in the same block
            # Does currently not work because of the cloning. We do not have access to sibling ops here
            # if block := op.parentBlock is not None:
            #     indexInParent = block.ops.index(op)
            #     if len(block.ops) < indexInParent + 1:
            #         rr = self.s.apply(block.ops[indexInParent + 1])
            return failure("one traversal failure")

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

    # illustrating the copy problem
    # rrImmM1 = one(seq(debug(), fail())).apply(immBeforeM)

    rrImmM1 = one(FoldConstantAdd()).apply(immBeforeM)
    assert (rrImmM1.isSuccess()
            and isinstance(rrImmM1.result[0], ImmutableOperation))

    rrImmM2 = one(FoldConstantAdd()).apply(rrImmM1.result[0])
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
    print(file.getvalue().strip())

    assert file.getvalue().strip(
    ) == expected_without_garbage_collection.strip()


if __name__ == "__main__":
    rewriting_with_immutability_experiments()