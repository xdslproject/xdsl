from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union, Tuple
from xdsl.dialects.builtin import ModuleOp
from xdsl.immutable_ir import *
from xdsl.ir import Operation, OpResult, Region, Block, BlockArgument, Attribute
from xdsl.rewriter import Rewriter
from xdsl.pattern_rewriter import *
from xdsl.printer import Printer


@dataclass
class RewriteResult:
    result: Union[str, List[ImmutableOperation]]

    def flatMapSuccess(self, s: Strategy) -> RewriteResult:
        if (not isinstance(self.result, List)):
            return self
        return s.apply(self.result[0])

    def flatMapFailure(self, f: Callable) -> RewriteResult:
        if (not isinstance(self.result, List)):
            return f()
        return self

    def __str__(self) -> str:
        if isinstance(self.result, str):
            return "Failure(" + self.result + ")"
        elif isinstance(self.result, List):
            return "Success, " + str(len(self.result)) + " new ops"
        else:
            assert False

    def isSuccess(self):
        return isinstance(self.result, List)


def success(ops: List[ImmutableOperation]) -> RewriteResult:
    assert isinstance(ops, List)
    assert all([isinstance(op, ImmutableOperation) for op in ops])
    return RewriteResult(ops)


def failure(errorMsg: str) -> RewriteResult:
    assert isinstance(errorMsg, str)
    return RewriteResult(errorMsg)


@dataclass
class Strategy:

    def apply(self, op: ImmutableOperation) -> RewriteResult:
        assert isinstance(op, ImmutableOperation)
        return self.impl(op)

    @abstractmethod
    def impl(self, op: ImmutableOperation) -> RewriteResult:
        ...


@dataclass
class id(Strategy):

    def impl(self, op: ImmutableOperation) -> RewriteResult:
        return success([op])


@dataclass
class fail(Strategy):

    def impl(self, op: ImmutableOperation) -> RewriteResult:
        return failure("fail Strategy")


@dataclass
class debug(Strategy):

    def impl(self, op: ImmutableOperation) -> RewriteResult:
        printer = Printer()
        printer.print_op(op._op)
        return success([op])


@dataclass
class seq(Strategy):
    s1: Strategy
    s2: Strategy

    def impl(self, op: ImmutableOperation) -> RewriteResult:
        rr = self.s1.apply(op)
        return rr.flatMapSuccess(self.s2)


@dataclass
class leftChoice(Strategy):
    s1: Strategy
    s2: Strategy

    def impl(self, op: ImmutableOperation) -> RewriteResult:
        return self.s1.apply(op).flatMapFailure(lambda: self.s2.apply(op))


@dataclass
class try_(Strategy):
    s: Strategy

    def impl(self, op: ImmutableOperation) -> RewriteResult:
        return leftChoice(self.s, id()).apply(op)


@dataclass
class one(Strategy):
    """
    Try to apply s to one the operands of op or to the first op in its region
    or to the next operation in the same block.
    """
    s: Strategy


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

    def impl(self, op: ImmutableOperation) -> RewriteResult:

        # TODO: not recursively currently
        def getUsedOperationsRecursively(
                op: ImmutableOperation,
                exceptions: List[int] = None) -> List[ImmutableOperation]:
            if exceptions is None:
                exceptions = []
            usedOps = []
            for idx, operand in enumerate(op.operands):
                if idx in exceptions:
                    continue
                if isinstance(operand, ImmutableOpResult):
                    usedOps.append(
                        ImmutableOperation.from_op(
                            operand.op.get_mutable_copy()))
            return usedOps

        for idx, operand in enumerate(op.operands):
            # Try to apply to the operands of this op
            if (isinstance(operand, ImmutableOpResult)):
                rr = self.s.apply(operand.op)
                if rr.isSuccess():
                    # recreate op
                    tmp = getUsedOperationsRecursively(op, [idx])
                    # newOp = op._op.clone()
                    # newOp.replace_operand(idx, rr.result[-1]._op.results[0])
                    # newImm = ImmutableOperation.from_op(newOp)

                    newOps = ImmutableOperation.create_new(
                        op._op.__class__, op.operands[:idx - 1] +
                        [rr.result[-1]] + op.operands[idx + 1:], op._op)
                    newOps[-1]._op.replace_operand(
                        idx, rr.result[-1]._op.results[0])

                    return success(tmp + rr.result + newOps)
        if len(op.regions) > 0 and len(op.region.block.ops) > 0:
            # Try to apply to last operation in the region of this op
            rr = self.s.apply(op.region.block.ops[-1])
            if rr.isSuccess():
                newOp = op._op.clone_without_regions()
                newOp.regions.clear()
                newRegion: Region = Region.from_operation_list(
                    [op._op for op in rr.result])

                newOp.add_region(newRegion)
                return success([ImmutableOperation.from_op(newOp)])

        return failure("one traversal failure")


@dataclass
class topdown(Strategy):
    """
    Topdown traversal
    """
    s: Strategy

    def impl(self, op: ImmutableOperation) -> RewriteResult:
        return leftChoice(self.s, one(topdown(self.s))).apply(op)


# old Strategy for mutable rewriting
# class Strategy(RewritePattern):

#     @abstractmethod
#     def impl(self, op: Operation) -> RewriteResult:
#         ...

#     def __call__(self, op: Operation,
#                  rewriter: PatternRewriter) -> RewriteResult:
#         return self.impl(op)

#     def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
#         """Keeping the original interface"""
#         result = self.impl(op)
#         operands = list()

#         if result is not None:
#             if not isinstance(result.result, str):
#                 rewriter.replace_matched_op(result.result)
#         else:
#             return

#         def addOperandsRecursively(op: Operation):
#             operands.extend(op.operands)
#             for operand in op.operands:
#                 if isinstance(operand, OpResult):
#                     addOperandsRecursively(operand.op)

#         #TODO: check trait
#         addOperandsRecursively(op)

#         # cleanup
#         eraseList = []
#         changed = True
#         while (changed):
#             changed = False
#             for value in operands:
#                 if len(value.uses) == 0:
#                     eraseList.append(value)

#             for value in eraseList:
#                 if value.op.parent is not None:
#                     print("erasing op:" + value.op.name)
#                     rewriter.erase_op(value.op)
#                     operands.remove(value)
#                     changed = True

# @dataclass
# class one(Strategy):
#     s: Strategy

#     def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
#         if isinstance(op, ModuleOp):
#             module: ModuleOp = op
#             return self.s(module.ops[0], rewriter)
#         for operand in reversed(op.operands):
#             if (isinstance(operand, OpResult)):
#                 rr = self.s(operand.op, rewriter)
#                 if isinstance(rr.result, Operation):
#                     return rr
#         return failure("one traversal failure")
