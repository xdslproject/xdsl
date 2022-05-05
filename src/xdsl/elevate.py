from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Callable
from xdsl.immutable_ir import *
from xdsl.pattern_rewriter import *


@dataclass
class RewriteResult:
    result: Union[Strategy, List[IOp]]
    env: Dict[IVal, IVal]

    def flatMapSuccess(self, s: Strategy) -> RewriteResult:
        # I think we lose the environment here
        if (not isinstance(self.result, List)):
            return self
        return s.apply(self.result[0])

    def flatMapFailure(self, f: Callable[[], RewriteResult]) -> RewriteResult:
        if (not isinstance(self.result, List)):
            return f()
        return self

    def __str__(self) -> str:
        if isinstance(self.result, Strategy):
            return "Failure(" + str(self.result) + ")"
        return "Success, " + str(len(self.result)) + " new ops"

    def isSuccess(self):
        return isinstance(self.result, List)

    @property
    def result_op(self) -> IOp:
        assert self.isSuccess()
        assert not isinstance(self.result, Strategy)
        return self.result[-1]


def success(arg: IOp | RewrittenIOp) -> RewriteResult:
    match arg:
        case IOp():
            op = arg
            env = {}
        case RewrittenIOp():
            op = arg.op
            env = arg.env
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
    return RewriteResult(ops, env)


def failure(failed_strategy: Strategy) -> RewriteResult:
    assert isinstance(failed_strategy, Strategy)
    return RewriteResult(failed_strategy, {})


@dataclass
class Strategy:

    def apply(self, op: IOp) -> RewriteResult:
        assert isinstance(op, IOp)
        return self.impl(op)

    @abstractmethod
    def impl(self, op: IOp) -> RewriteResult:
        ...

    def __str__(self) -> str:
        name: str = self.__class__.__name__ + "("
        for strategy in vars(self).values():
            name += str(strategy)
        return name + ")"


@dataclass
class id(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return success(op)


@dataclass
class fail(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return failure(self)


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

                    result = new_op(op_type=op.op_type,
                                    operands=list(new_operands),
                                    result_types=op.result_types,
                                    attributes=op.get_attributes_copy(),
                                    successors=list(op.successors),
                                    regions=op.regions,
                                    env=rr.env)

                    return success(result)
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
                               rr.env, matched_block)
                    ])
                ] + list(op.regions[idx + 1:])

                result = new_op(op_type=op.op_type,
                                operands=list(op.operands),
                                result_types=op.result_types,
                                attributes=op.attributes,
                                successors=list(op.successors),
                                regions=new_regions,
                                env=rr.env)

                return success(result)
        return failure(self)


@dataclass
class topdown(Strategy):
    """
    Topdown traversal
    """
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return leftChoice(self.s, one(topdown(self.s))).apply(op)
