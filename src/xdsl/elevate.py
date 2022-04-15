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

    def impl(self, op: ImmutableOperation) -> RewriteResult:
        for idx, operand in enumerate(op.operands):
            # Try to apply to the operands of this op
            if (isinstance(operand, ImmutableOpResult)):
                rr = self.s.apply(operand.op)
                if rr.isSuccess():
                    assert isinstance(rr.result, List)
                    # build the operands including the new operand
                    newOperands = list(op.operands[:idx]) + [
                        rr.result[-1].results[operand.result_index]
                    ] + list(op.operands[idx + 1:])

                    # Not handled yet:
                    #   - when the operand has regions
                    #   - when the op has successors

                    newOps = ImmutableOperation.create_new(
                        op_type=op._op.__class__,
                        immutable_operands=newOperands,
                        result_types=op.result_types,
                        attributes=op.get_attributes_copy(),
                        successors=op._op.successors)

                    return success(rr.result + newOps)
        for idx, region in enumerate(op.regions):
            # Try to apply to last operation in the last block in the regions of this op
            if len(region.blocks) == 0:
                continue

            rr = self.s.apply((matched_block := region.blocks[-1]).ops[-1])
            if rr.isSuccess():
                assert isinstance(rr.result, List)
                # build new operation with the new region
                new_regions = list(op.regions[:idx]) + [
                    ImmutableRegion.create_new(
                        [ImmutableBlock.create_new(rr.result, matched_block)])
                ] + list(op.regions[idx + 1:])

                newOp = ImmutableOperation.create_new(
                    op_type=op._op.__class__,
                    immutable_operands=list(op.operands),
                    result_types=op.result_types,
                    attributes=op.get_attributes_copy(),
                    successors=list(op.successors),
                    regions=new_regions)

                return success(newOp)

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
