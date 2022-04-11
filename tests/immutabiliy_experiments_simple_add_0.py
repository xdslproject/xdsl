from __future__ import annotations
from io import StringIO
from mimetypes import init
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

import difflib

###
#
#   In this experiment we try a rough first interface for an Immutable view on ops
#   This has now been replaced by xdsl.immutable_ir.py
#
###


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
    class ImmutableOpView:
        name: str
        _op: Operation

        @classmethod
        def from_op(cls, op: Operation) -> ImmutableOpView:
            assert isinstance(op, Operation)
            return ImmutableOpView(op.name, op)

        def get_attribute(self, name: str) -> Attribute:
            return self._op.attributes[name]

    @dataclass
    class ImmutableRewrite:

        def match_and_rewrite(self, op: Operation,
                              rewriter: PatternRewriter) -> Operation:
            matchedOps = self.match(op)
            if matchedOps is None:
                return

            immutableOps: List[ImmutableOpView] = []
            for matchedOp in matchedOps:
                immutableOps.append(ImmutableOpView.from_op(matchedOp))

            newOps = self.rewrite(*immutableOps)
            if newOps is None:
                return

            rewriter.replace_matched_op(newOps)

            # collect garbage (non recursively currently)
            matchedOps.remove(op)
            for matchedOp in matchedOps:
                if all(len(result.uses) == 0 for result in matchedOp.results):
                    rewriter.erase_op(matchedOp)
            return newOps[0]

        @abstractmethod
        def match(self, op: Operation) -> List[Operation]:
            ...

        @abstractmethod
        def rewrite(self, *ops: ImmutableOpView) -> List[Operation]:
            ...

    @dataclass
    class FoldConstantAdd(ImmutableRewrite):

        def match(self, op: Operation) -> list[Operation]:
            if isinstance(op, Addi) and isinstance(op.input1.op,
                                                   Constant) and isinstance(
                                                       op.input2.op, Constant):
                assert (isinstance(op.input1.op.value.typ, IntegerType))
                assert (isinstance(op.input2.op.value.typ, IntegerType))
                return [op, op.input1.op, op.input2.op]

        def rewrite(self, addOp: ImmutableOpView, c1: ImmutableOpView,
                    c2: ImmutableOpView) -> List[Operation]:
            c1Val = c1.get_attribute("value")
            c2Val = c2.get_attribute("value")
            return [
                Constant.from_int_constant(c1Val.value.data + c2Val.value.data,
                                           c1Val.typ)
            ]

    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)
    rise = Rise(ctx)
    rise_dsl = RiseBuilder(ctx)
    affine = Affine(ctx)

    parser = Parser(ctx, before)
    moduleBefore: ModuleOp = parser.parse_op()

    rewrite = FoldConstantAdd()
    # moduleBefore.walk(lambda op: {rewrite.match_and_rewrite(op)})

    PatternRewriteWalker(FoldConstantAdd()).rewrite_module(moduleBefore)

    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(moduleBefore)
    diff = list(difflib.Differ().compare(file.getvalue().splitlines(True),
                                         expected.splitlines(True)))
    print(''.join(diff))

    print(file.getvalue().strip())

    assert file.getvalue().strip() == expected.strip()


if __name__ == "__main__":
    rewriting_with_immutability_experiments()