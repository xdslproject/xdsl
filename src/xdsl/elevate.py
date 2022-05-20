from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Callable
from xdsl.immutable_ir import *
from xdsl.pattern_rewriter import *


@dataclass
class IOpReplacement:
    matched_op: IOp
    replacement_ops: List[IOp]


@dataclass
class RewriteResult:
    _result: Union[Strategy, List[IOpReplacement]]

    def flatMapSuccess(self, s: Strategy) -> RewriteResult:
        if (not self.isSuccess()):
            return self
        rr = s.apply(self.result_op)
        if (not rr.isSuccess()):
            return rr
        self += rr
        return self

    def flatMapFailure(self, f: Callable[[], RewriteResult]) -> RewriteResult:
        if (not self.isSuccess()):
            return f()
        return self

    def __str__(self) -> str:
        if not self.isSuccess():
            return "Failure(" + str(self.failed_strategy) + ")"
        return "Success, " + str(len(self.replacements)) + " replacements"

    def isSuccess(self) -> bool:
        return isinstance(self._result, List)

    def __iadd__(self, other: RewriteResult):
        if self.isSuccess() and other.isSuccess():
            assert self.isSuccess() and other.isSuccess()
            assert isinstance(self._result, List) and isinstance(
                other._result, List)
            self._result += other._result
            return self
        raise Exception("invalid concatenation of RewriteResults")

    @property
    def failed_strategy(self) -> Strategy:
        assert not self.isSuccess()
        assert isinstance(self._result, Strategy)
        return self._result

    @property
    def replacements(self) -> List[IOpReplacement]:
        assert self.isSuccess()
        assert isinstance(self._result, List)
        return self._result

    @property
    def result_op(self) -> IOp:
        assert self.isSuccess()
        assert not isinstance(self._result, Strategy)
        return self.replacements[-1].replacement_ops[-1]


def success(arg: IOp | List[IOp] | RewriteResult) -> RewriteResult:
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

    # matched op will be set by the Strategy itself in `apply`
    return RewriteResult([IOpReplacement(None, ops)])  # type: ignore


def failure(failed_strategy: Strategy) -> RewriteResult:
    assert isinstance(failed_strategy, Strategy)
    return RewriteResult(failed_strategy)


@dataclass
class Strategy:

    def apply(self, op: IOp) -> RewriteResult:
        assert isinstance(op, IOp)

        rr = self.impl(op)

        if rr.isSuccess():
            rr.replacements[-1].matched_op = op

            # If matched op is referred to in replacement IR add it to the replacement
            matched_op_used = False
            for result_op in (replacement :=
                              rr.replacements[-1].replacement_ops):

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
        return success(op)


@dataclass
class fail(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return failure(self)


@dataclass
class debug(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
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
class one(Strategy):  # TODO: think about name
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
                    assert len(rr.result_op.results) > operand.result_index
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

                    rr += success(result)
                    return rr
        for idx, region in enumerate(op.regions):
            # Try to apply to last operation in the last block in the regions of this op
            if len(region.blocks) == 0:
                continue

            rr = self.s.apply((matched_block := region.blocks[-1]).ops[-1])
            if rr.isSuccess():
                assert isinstance(rr.replacements, List)
                # applying the replacements in rr to the original ops of the matched block
                nested_ops: List[IOp] = list(matched_block.ops)

                completed_replacements: List[IOpReplacement] = []
                for replacement in rr.replacements:
                    if replacement.matched_op in nested_ops:
                        i = nested_ops.index(replacement.matched_op)
                        nested_ops[i:i + 1] = replacement.replacement_ops
                        completed_replacements.append(replacement)
                    else:
                        # print("op not in nested ops")
                        # print(replacement.matched_op.name)
                        # print("nested:(")
                        # for nested_op in nested_ops:
                        #     print(nested_op.name)
                        # print(") of block of op:")
                        # print(op.name)
                        # pass
                        raise Exception(
                            "replacement out of scope, could not be applied")

                for replacement in completed_replacements:
                    rr.replacements.remove(replacement)

                new_regions = op.regions[:idx] + [
                    IRegion([IBlock.from_iblock(nested_ops, matched_block)])
                ] + op.regions[idx + 1:]

                result = new_op(op_type=op.op_type,
                                operands=list(op.operands),
                                result_types=op.result_types,
                                attributes=op.attributes,
                                successors=list(op.successors),
                                regions=new_regions)
                rr += success(result)
                return rr
        return failure(self)


@dataclass
class topdown(Strategy):  # TODO: think about name
    """
    Topdown traversal
    """
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return leftChoice(self.s, one(topdown(self.s))).apply(op)