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

    # @dataclass
    # class ImmutableSSAValueView:
    #     _value: SSAValue

    @dataclass
    class ImmutableOpResultView:
        op: ImmutableOperation
        result_index: int

    @dataclass
    class RewriteResult:
        result: Union[str, List[Operation]]

        @staticmethod
        def success(ops: List[Operation]) -> RewriteResult:
            return RewriteResult(ops)

        @staticmethod
        def failure(message: str) -> RewriteResult:
            return RewriteResult(message)

        def __str__(self) -> str:
            if isinstance(self.result, str):
                return "Failure(" + self.result + ")"
            elif isinstance(self.result, List):
                return "Success, " + str(len(self.result)) + " new ops"
            else:
                assert False

    @dataclass
    class ImmutableRegion:
        blocks: FrozenList[ImmutableBlock]

        @staticmethod
        def from_immutable_operation_list(
                ops: List[ImmutableOperation]) -> ImmutableRegion:
            block = ImmutableBlock.from_immutable_ops(ops)
            return ImmutableRegion([block])

        @staticmethod
        def from_operation_list(ops: List[Operation]) -> ImmutableRegion:
            block = ImmutableBlock.from_ops(ops)
            return ImmutableRegion([block])

        @staticmethod
        def from_immutable_block_list(
                blocks: List[ImmutableBlock]) -> ImmutableRegion:
            return ImmutableRegion(blocks)

        @staticmethod
        def from_block_list(blocks: List[Block]) -> ImmutableRegion:
            immutable_blocks = [
                ImmutableBlock.from_block(block) for block in blocks
            ]
            return ImmutableRegion(immutable_blocks)

        def walk(self, fun: Callable[[Operation], None]) -> None:
            for block in self.blocks:
                block.walk(fun)

    @dataclass
    class ImmutableBlock:
        args: FrozenList[BlockArgument]
        ops: FrozenList[ImmutableOperation]
        # parent: Optional[ImmutableRegion] = field(default=None, init=False)

        @staticmethod
        def from_block(block: Block) -> ImmutableBlock:
            context: dict[Operation, ImmutableOperation] = {}
            immutableOps = [
                ImmutableOperation.from_op(op, context) for op in block.ops
            ]
            for key in context:
                print(key.name + ", " + context[key].name)

            return ImmutableBlock(block._args, immutableOps)

        @staticmethod
        def from_immutable_ops(
                ops: List[ImmutableOperation]) -> ImmutableBlock:
            return ImmutableBlock([], ops)

        @staticmethod
        def from__ops(ops: List[Operation]) -> ImmutableBlock:
            context: dict[Operation, ImmutableOperation] = {}
            immutable_ops = [
                ImmutableOperation.from_op(op, context) for op in ops
            ]
            return ImmutableBlock.from_immutable_ops(immutable_ops)

        def walk(self, fun: Callable[[Operation], None]) -> None:
            for op in self.ops:
                op.walk(fun)

    @dataclass
    class ImmutableOperation:
        name: str
        _op: Operation
        operands: FrozenList[ImmutableOpResultView]  # could also be BlockArg
        regions: FrozenList[ImmutableRegion]

        @staticmethod
        def from_op(
            op: Operation,
            context: dict[Operation,
                          ImmutableOperation]) -> ImmutableOperation:
            assert isinstance(op, Operation)

            operands: List[ImmutableOpResultView] = []
            for operand in op.operands:
                assert (isinstance(operand, OpResult))
                operands.append(
                    ImmutableOpResultView(context[operand.op],
                                          operand.result_index))

            regions: List[ImmutableRegion] = []
            for region in op.regions:
                regions.append(ImmutableRegion.from_block_list(region.blocks))

            immutableOp = ImmutableOperation("immutable." + op.name, op,
                                             operands, regions)

            context[op] = immutableOp
            return immutableOp

        def is_op(self, SomeOpClass):
            if isinstance(self._op, SomeOpClass):
                return self
            else:
                return None

        def get_attribute(self, name: str) -> Attribute:
            return self._op.attributes[name]

        def walk(self, fun: Callable[[Operation], None]) -> None:
            fun(self)
            for region in self.regions:
                region.walk(fun)

    @dataclass
    class ImmutableRewrite:

        @abstractmethod
        def apply(self, op: ImmutableOperation) -> RewriteResult:
            ...

        @abstractmethod
        def impl(self, op: ImmutableOperation) -> RewriteResult:
            ...

    ###
    # TODO: restructure rewrite to match the whole module and also create a completely new module.
    # This way I don't have to care about traversals right now.
    @dataclass
    class FoldConstantAdd(ImmutableRewrite):

        def apply(self, op: ImmutableOperation) -> RewriteResult:
            result = self.impl(op)

            rewriter = PatternRewriter(op._op)

            garbageCandidates = []
            # only works when all operands are OpResults
            garbageCandidates.extend(operand.op for operand in op._op.operands)

            # We do not want to mutate the existing IR, but it is nice for checking validity
            rewriter.replace_matched_op(
                [result._op for result in result.result])

            for op in garbageCandidates:
                if all(len(result.uses) == 0 for result in op.results):
                    rewriter.erase_op(op)

            return result

        def impl(self, op: ImmutableOperation) -> RewriteResult:
            if (addOp := op.is_op(Addi)) and (
                    c1 := addOp.operands[0].op.is_op(Constant)) and (
                        c2 := addOp.operands[1].op.is_op(Constant)):

                assert (isinstance(c1.get_attribute("value").typ, IntegerType))
                assert (isinstance(c2.get_attribute("value").typ, IntegerType))

                c1Val = c1.get_attribute("value")
                c2Val = c2.get_attribute("value")
                return RewriteResult.success([
                    ImmutableOperation.from_op(
                        Constant.from_int_constant(
                            c1Val.value.data + c2Val.value.data, c1Val.typ),
                        {})
                ])
            else:
                return RewriteResult.failure("FoldConstantAdd")

    def get_immutable_copy(op: Operation) -> ImmutableOperation:
        # if isinstance(op, ModuleOp):
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

    immutableModule.walk(
        lambda op:
        {print(FoldConstantAdd().apply(op)) if op.is_op(Addi) else None})

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