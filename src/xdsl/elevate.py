from __future__ import annotations
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union, Tuple

from numpy import true_divide

from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, OpResult, Region, Block, BlockArgument, Attribute
from xdsl.rewriter import Rewriter
from xdsl.pattern_rewriter import *
from xdsl.printer import Printer


@dataclass
class RewriteResult:
    result: Union[str, List[Operation]]

    def flatMapSuccess(self, s: Strategy, rewriter: Rewriter) -> RewriteResult:
        if (not isinstance(self.result, List)):
            return self
        return s(self.result, rewriter)

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


def success(op: Operation) -> RewriteResult:
    return RewriteResult(op)


def failure(errorMsg: str) -> RewriteResult:
    return RewriteResult(errorMsg)


class Strategy(RewritePattern):

    @abstractmethod
    def impl(self, op: Operation) -> RewriteResult:
        ...

    def __call__(self, op: Operation,
                 rewriter: PatternRewriter) -> RewriteResult:
        return self.impl(op)

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        """Keeping the original interface"""
        result = self.impl(op)
        operands = list()

        if result is not None:
            if not isinstance(result.result, str):
                rewriter.replace_matched_op(result.result)
        else:
            return

        def addOperandsRecursively(op: Operation):
            operands.extend(op.operands)
            for operand in op.operands:
                if isinstance(operand, OpResult):
                    addOperandsRecursively(operand.op)

        #TODO: check trait
        addOperandsRecursively(op)

        # cleanup
        eraseList = []
        changed = True
        while (changed):
            changed = False
            for value in operands:
                if len(value.uses) == 0:
                    eraseList.append(value)

            for value in eraseList:
                if value.op.parent is not None:
                    print("erasing op:" + value.op.name)
                    rewriter.erase_op(value.op)
                    operands.remove(value)
                    changed = True


# TODO: reintegration of stuff that is returned from the rewriting
class id(Strategy):

    def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
        return success(op)


class fail(Strategy):

    def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
        return failure("fail Strategy applied")


@dataclass
class debug(Strategy):
    # debugMsg: str

    def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
        printer = Printer()
        printer.print_op(op)
        return success(op)


@dataclass
class seq(Strategy):
    s1: Strategy
    s2: Strategy

    def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
        rr = self.s1(op, rewriter)
        return rr.flatMapSuccess(self.s2, rewriter)


@dataclass
class leftChoice(Strategy):
    s1: Strategy
    s2: Strategy

    def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
        return self.s1(op,
                       rewriter).flatMapFailure(lambda: self.s2(op, rewriter))


@dataclass
class try_(Strategy):
    s: Strategy

    def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
        return leftChoice(self.s, id())(op, rewriter)


@dataclass
class one(Strategy):
    s: Strategy

    def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
        if isinstance(op, ModuleOp):
            module: ModuleOp = op
            return self.s(module.ops[0], rewriter)
        for operand in reversed(op.operands):
            if (isinstance(operand, OpResult)):
                rr = self.s(operand.op, rewriter)
                if isinstance(rr.result, Operation):
                    return rr
        return failure("one traversal failure")


@dataclass
class topDown(Strategy):
    s: Strategy

    def impl(self, op: Operation, rewriter: PatternRewriter) -> RewriteResult:
        return leftChoice(self.s, one(topDown(self.s)))(op, rewriter)