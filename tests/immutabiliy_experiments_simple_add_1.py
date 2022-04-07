from __future__ import annotations
from io import StringIO
from mimetypes import init
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
std.return(%2 : !i32)
}
"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 3 : !i32]
  std.return(%0 : !i32)
}
"""

    @dataclass
    class ImmutableRewrite:

        def apply(self, op: ImmutableOperation) -> RewriteResult:
            result = self.impl(op)
            if isinstance(result.result, str):
                return result

            rewriter = PatternRewriter(op._op)

            collectGarbage = False
            if collectGarbage:
                garbageCandidates = []
                garbageCandidates.extend([
                    operand.op for operand in op._op.operands
                    if isinstance(operand, OpResult)
                ])

                # We do not want to mutate the existing IR, but it is nice for checking validity
                rewriter.replace_matched_op(
                    [result._op for result in result.result])

                for op in garbageCandidates:
                    if all(len(result.uses) == 0 for result in op.results):
                        rewriter.erase_op(op)

            return result

        @abstractmethod
        def impl(self, op: ImmutableOperation) -> RewriteResult:
            ...

    ###
    # TODO: restructure rewrite to match the whole module and also create a completely new module.
    # This way I don't have to care about traversals right now.
    @dataclass
    class FoldConstantAdd(ImmutableRewrite):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (addOp := op.is_op(Addi)) and (
                    c1 := addOp.operands[0].op.is_op(Constant)) and (
                        c2 := addOp.operands[1].op.is_op(Constant)):

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
    class FoldConstantAddInModule(ImmutableRewrite):

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (module := op.is_op(ModuleOp)):
                addOp = None
                newModule = module._op.clone()
                newImmutableModule = get_immutable_copy(newModule)

                for op in newImmutableModule.region.block.ops:
                    if op.is_op(Addi):
                        addOp = op

                print(addOp.operands[0].op == addOp.operands[1].op)
                if addOp.operands[0].op.is_op(Constant) and (
                        addOp.operands[1].op.is_op(Constant)):

                    c1 = addOp.operands[0].op
                    c2 = addOp.operands[1].op
                    print(c1 == c2)
                    assert (isinstance(
                        c1.get_attribute("value").typ, IntegerType))
                    assert (isinstance(
                        c2.get_attribute("value").typ, IntegerType))

                    c1Val = c1.get_attribute("value")
                    c2Val = c2.get_attribute("value")

                    # newModule = ModuleOp.from_region_or_ops([
                    #     op.clone_without_regions()
                    #     for op in module._op.body.ops
                    # ])

                    printer = Printer()
                    printer.print_op(newModule)

                    #TODO: clean up
                    rewriter = PatternRewriter(addOp._op)
                    newConstant = Constant.from_int_constant(
                        c1Val.value.data + c2Val.value.data, c1Val.typ)
                    rewriter.replace_matched_op(newConstant)

                    printer.print_op(newModule)

                    return success(
                        [ImmutableOperation.from_op(newConstant, {})])
                    # return success([
                    #     ImmutableOperation.from_op(
                    #         Constant.from_int_constant(
                    #             c1Val.value.data + c2Val.value.data,
                    #             c1Val.typ), {})
                    # ])
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
    mutableModule: ModuleOp = parser.parse_op()

    immutableModule: ImmutableOperation = get_immutable_copy(mutableModule)
    immutableModule.walk(
        lambda op: print("i bims, 1 " + op.name)
        if isinstance(op, ImmutableOperation) else print("mutable"))

    # immutableModule.walk(
    #     lambda op:
    #     {print(FoldConstantAdd().apply(op)) if op.is_op(Addi) else None})

    print(FoldConstantAddInModule().apply(immutableModule))

    # maybeANewImmutableModule: RewriteResult = \
    #     FoldConstantAdd().apply(immutableModule)

    # assert(maybeANewImmutableModule is not Failure)
    # maybeANewImmutableModule2: RewriteResult[ImmutableModule] = newRewrite.apply(maybeANewImmutableModule)
    # module2: Module = make_mutable_copy(immutableModule2: MutableModule)

    # module1 = copy(originalModule)
    # maybeModule2 = someRewrite.apply(module1)
    # if True:
    #     module2 = copy(maybeModule2)
    #     maybeModul3 = someRewrite2.apply(module2)

    # PatternRewriteWalker(FoldConstantAdd()).rewrite_module(mutableModule)

    # TODO: get a mutable module out of the immutable module

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(mutableModule)
    diff = list(difflib.Differ().compare(file.getvalue().splitlines(True),
                                         expected.splitlines(True)))
    print(''.join(diff))

    print(file.getvalue().strip())

    assert file.getvalue().strip() == expected.strip()

    # Now do rewriting

    # Do easiest thing first. Do a complete clone of everything


if __name__ == "__main__":
    rewriting_with_immutability_experiments()