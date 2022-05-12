from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Callable, NamedTuple
from xdsl.immutable_ir import *
from xdsl.pattern_rewriter import *


class IOpReplacement(NamedTuple):
    matched_op: IOp
    replacement: List[IOp]


@dataclass
class RewriteResult:
    result: Union[Strategy, List[IOpReplacement]]

    def flatMapSuccess(self, s: Strategy) -> RewriteResult:
        if (not isinstance(self.result, List)):
            return self
        rr = s.apply(self.result_op)
        self += rr
        return self

    def flatMapFailure(self, f: Callable[[], RewriteResult]) -> RewriteResult:
        if (not isinstance(self.result, List)):
            return f()
        return self

    def __str__(self) -> str:
        if isinstance(self.result, Strategy):
            return "Failure(" + str(self.result) + ")"
        return "Success, " + str(len(self.result)) + " new ops"

    def isSuccess(self) -> bool:
        return isinstance(self.result, List)

    def __iadd__(self, other: RewriteResult):
        if self.isSuccess() and other.isSuccess():
            assert isinstance(self.result, List) and isinstance(
                other.result, List)
            self.result += other.result
            return self
        raise Exception("invalid concatenation of RewriteResults")

    @property
    def result_op(self) -> IOp:
        assert self.isSuccess()
        assert not isinstance(self.result, Strategy)
        return self.result[-1].replacement[-1]


def success(arg: IOp | List[IOp] | RewriteResult,
            matched_op: IOp) -> RewriteResult:
    match arg:
        case IOp():
            ops = [arg]
        case [*_]:
            ops = arg
        case _:
            raise Exception("success called with incompatible arguments")

    # TRACING DISABLED
    # tracing def use relations:
    # Add all dependant operations to `ops`
    def trace_operands_recursive(operands: IList[ISSAValue], ops: List[IOp]):
        for operand in operands:
            if isinstance(operand, IResult):
                if operand.op not in ops:
                    ops.insert(0, operand.op)
                    trace_operands_recursive(operand.op.operands, ops)

    # trace_operands_recursive(ops[-1].operands, ops)

    return RewriteResult([IOpReplacement(matched_op, ops)])


def failure(failed_strategy: Strategy) -> RewriteResult:
    assert isinstance(failed_strategy, Strategy)
    return RewriteResult(failed_strategy)


@dataclass
class Strategy:

    def apply(self, op: IOp) -> RewriteResult:
        assert isinstance(op, IOp)

        rr = self.impl(op)
        if rr.isSuccess():
            assert isinstance(rr.result, List)

            # If matched op is referred to in replacement IR add it to the replacement
            matched_op_used = False
            for result_op in (replacement := rr.result[-1].replacement):

                def uses_matched_op(result_op: IOp):
                    for result in op.results:
                        if result in result_op.operands:
                            nonlocal matched_op_used
                            matched_op_used = True

                result_op.walk(uses_matched_op)

            if matched_op_used and op not in replacement:
                replacement.insert(0, op)

        return rr

    @abstractmethod
    def impl(self, op: IOp) -> RewriteResult:
        ...

    def __str__(self) -> str:
        values = [str(value) for value in vars(self).values()]
        return f'{self.__class__.__name__}({",".join(values)})'


@dataclass
class id(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return success(op, op)


@dataclass
class fail(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return failure(self)


@dataclass
class debug(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        print("debug:" + op.name)
        return success(op, op)


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
            if (isinstance(operand, IResult)):
                rr = self.s.apply(operand.op)
                if rr.isSuccess():
                    assert isinstance(rr.result, List)
                    # build the operands including the new operand
                    new_operands: List[ISSAValue] = op.operands[:idx] + [
                        rr.result_op.results[operand.result_index]
                    ] + op.operands[idx + 1:]

                    result = new_op(op_type=op.op_type,
                                    operands=list(new_operands),
                                    result_types=op.result_types,
                                    attributes=op.get_attributes_copy(),
                                    successors=list(op.successors),
                                    regions=op.regions)

                    rr += success(result, op)
                    return rr
        for idx, region in enumerate(op.regions):
            # Try to apply to last operation in the last block in the regions of this op
            if len(region.blocks) == 0:
                continue

            rr = self.s.apply((matched_block := region.blocks[-1]).ops[-1])
            if rr.isSuccess():
                assert isinstance(rr.result, List)
                # applying the replacements in rr to the original ops of the matched block
                nested_ops: List[IOp] = matched_block.ops
                for (matched_op, replacement_ops) in rr.result:
                    if matched_op in nested_ops:
                        matched_op_index = nested_ops.index(matched_op)
                        nested_ops[matched_op_index:matched_op_index +
                                   1] = replacement_ops

                new_regions = op.regions[:idx] + [
                    IRegion([IBlock.from_iblock(nested_ops, matched_block)])
                ] + op.regions[idx + 1:]

                result = new_op(op_type=op.op_type,
                                operands=list(op.operands),
                                result_types=op.result_types,
                                attributes=op.attributes,
                                successors=list(op.successors),
                                regions=new_regions)
                rr += success(result, op)
                return rr
        return failure(self)


@dataclass
class topdown(Strategy):
    """
    Topdown traversal
    """
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return leftChoice(self.s, one(topdown(self.s))).apply(op)