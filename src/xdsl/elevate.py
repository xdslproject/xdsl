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
    result: Union[str, List[IOp]]
    environment: Dict[IVal, IVal]

    def flatMapSuccess(self, s: Strategy) -> RewriteResult:
        # I think we lose the environment here
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


def success(
        arg: Union[IOp, Tuple[IOp, Dict[IVal, IVal]],
                   IBuilder]) -> RewriteResult:
    match arg:
        case IOp():
            op = arg
            environment = {}
        case (IOp(), dict()):
            assert isinstance(arg[1], Dict)
            op = arg[0]
            environment = arg[1]
        case IBuilder():
            assert arg.last_op_created is not None
            op = arg.last_op_created
            environment = arg.environment
        case _:
            raise Exception("success called with incompatible arguments")

    assert isinstance(op, IOp)
    ops: List[IOp] = [op]

    # Add all dependant operations
    def add_operands(op: IOp, ops: List[IOp]):
        for operand in op.operands:
            if isinstance(operand, IRes):
                if operand.op not in ops:
                    ops.insert(0, operand.op)
                    add_operands(operand.op, ops)

    add_operands(op, ops)
    return RewriteResult(ops, environment)


def failure(errorMsg: str) -> RewriteResult:
    assert isinstance(errorMsg, str)
    return RewriteResult(errorMsg, {})


@dataclass
class Strategy:

    def apply(self, op: IOp) -> RewriteResult:
        assert isinstance(op, IOp)
        return self.impl(op)

    @abstractmethod
    def impl(self, op: IOp) -> RewriteResult:
        ...


@dataclass
class id(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return success(op)


@dataclass
class fail(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return failure("fail Strategy")


@dataclass
class debug(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        # printer = Printer()
        # printer.print_op(op._op)
        print("debug:" + op.name)
        return success(op)


@dataclass
class seq(Strategy):
    s1: Strategy
    s2: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        rr = self.s1.apply(op)
        return rr.flatMapSuccess(self.s2)


@dataclass
class leftChoice(Strategy):
    s1: Strategy
    s2: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return self.s1.apply(op).flatMapFailure(lambda: self.s2.apply(op))


@dataclass
class try_(Strategy):
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return leftChoice(self.s, id()).apply(op)


@dataclass
class one(Strategy):
    """
    Try to apply s to one the operands of op or to the first op in its region
    or to the next operation in the same block.
    """
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        for idx, operand in enumerate(op.operands):
            # Try to apply to the operands of this op
            if (isinstance(operand, IRes)):
                rr = self.s.apply(operand.op)
                if rr.isSuccess():
                    assert isinstance(rr.result, List)
                    # build the operands including the new operand
                    new_operands = list(op.operands[:idx]) + [
                        rr.result[-1].results[operand.result_index]
                    ] + list(op.operands[idx + 1:])

                    # Not handled yet:
                    #   - when the operand has regions
                    #   - when the op has successors
                    b = IBuilder()
                    b.environment |= rr.environment

                    new_op = b.op(op_type=op.op_type,
                                  operands=list(new_operands),
                                  result_types=op.result_types,
                                  attributes=op.get_attributes_copy(),
                                  successors=list(op.successors))

                    return success(b)
        for idx, region in enumerate(op.regions):
            # Try to apply to last operation in the last block in the regions of this op
            if len(region.blocks) == 0:
                continue

            rr = self.s.apply((matched_block := region.blocks[-1]).ops[-1])
            if rr.isSuccess():
                assert isinstance(rr.result, List)

                new_regions = list(op.regions[:idx]) + [
                    IRegion([
                        IBlock(list(matched_block.arg_types), rr.result,
                               rr.environment, matched_block)
                    ])
                ] + list(op.regions[idx + 1:])

                b = IBuilder()
                b.environment |= rr.environment
                new_op = b.op(op_type=op.op_type,
                              operands=list(op.operands),
                              result_types=op.result_types,
                              attributes=op.attributes,
                              successors=list(op.successors),
                              regions=new_regions)

                return success(b)

        return failure("one traversal failure")


@dataclass
class topdown(Strategy):
    """
    Topdown traversal
    """
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return leftChoice(self.s, one(topdown(self.s))).apply(op)
