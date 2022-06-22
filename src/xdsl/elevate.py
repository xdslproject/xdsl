from __future__ import annotations
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import List, Callable, MutableSequence
from xdsl.immutable_ir import *
from xdsl.pattern_rewriter import *


@dataclass
class IOpReplacement:
    matched_op: IOp
    replacement_ops: List[IOp]


@dataclass
class RewriteResult:
    _result: Union[Strategy, Sequence[IOpReplacement]]

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


def success(arg: IOp | Sequence[IOp], matched_op: Optional[IOp] = None) -> RewriteResult:
    match arg:
        case IOp():
            ops = [arg]
        case [*_]:
            # remove duplicates - this way we can choose to enforce a specific
            # order of the ops by explicitly ordering them in the call to success
            ops = list(OrderedDict.fromkeys(arg))
        case _:
            raise Exception("success called with incompatible arguments")

    # matched op will be set by the Strategy itself in `apply`
    return RewriteResult([IOpReplacement(matched_op, ops)])  # type: ignore


def failure(failed_strategy: Strategy) -> RewriteResult:
    assert isinstance(failed_strategy, Strategy)
    return RewriteResult(failed_strategy)


@dataclass(frozen=True)
class Strategy:

    def apply(self, op: IOp) -> RewriteResult:
        assert isinstance(op, IOp)

        rr = self.impl(op)

        if isinstance(rr, Strategy):
            return rr

        if rr.isSuccess():
            for replacement in rr.replacements:
                if replacement.matched_op is None:
                    replacement.matched_op = op

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

    # Overloading ^ operator for sequential composition of Strategies
    def __xor__(self: Strategy, other: Strategy):
        return seq(self, other)


@dataclass(frozen=True)
class id(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return success(op)


@dataclass(frozen=True)
class fail(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        return failure(self)


@dataclass(frozen=True)
class debug(Strategy):

    def impl(self, op: IOp) -> RewriteResult:
        print("debug:" + op.name)
        return success(op)


@dataclass(frozen=True)
class seq(Strategy):
    s1: Strategy
    s2: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        rr = self.s1.apply(op)
        return rr.flatMapSuccess(self.s2)


@dataclass(frozen=True)
class leftChoice(Strategy):
    s1: Strategy
    s2: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return self.s1.apply(op).flatMapFailure(lambda: self.s2.apply(op))


@dataclass(frozen=True)
class try_(Strategy):
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return leftChoice(self.s, id()).apply(op)


########################################################################
######################    Traversal Strategies    ######################
########################################################################


@dataclass(frozen=True)
class backwards_step(Strategy):
    """
    Try to apply s to one the operands of op or to the last op in its region
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
                        # We never want to materialize rewrite.id operations
                        replacement.replacement_ops = [
                            op for op in replacement.replacement_ops
                            if op.op_type != RewriteId
                        ]

                        i = nested_ops.index(replacement.matched_op)
                        nested_ops[i:i + 1] = replacement.replacement_ops
                        completed_replacements.append(replacement)
                    else:
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


@dataclass(frozen=True)
class backwards(Strategy):
    """
    backwards traversal - Try to apply the Strategy `s` to the `op` we are matching on. 
    If unsuccessful try to apply to the operands of `op`. Proceeds recursively to the 
    operands of operands.
    """
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return leftChoice(self.s, backwards_step(backwards(self.s))).apply(op)


@dataclass(frozen=True)
class outermost(Strategy):
    """
    Outermost traversal
    """
    predicate: Strategy
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:
        return backwards(seq(self.predicate, self.s)).apply(op)


@dataclass(frozen=True)
class RegionsTraversal(Strategy, ABC):
    """
    Traversal which handles application of a BlockTraversal to a specific IRegion
    and builds a replacement for the matched op with a new IRegion afterwards.  
    """
    block_trav: BlocksTraversal


@dataclass(frozen=True)
class BlocksTraversal(ABC):
    """
    Special kind of Strategy which is applied to an IRegion and returns a new IRegion
    on success. Only composes with RegionTraversal and OpTraversal
    """
    op_trav: OpsTraversal

    def apply(self, region: IRegion) -> Optional[IRegion]:
        return self.impl(region)

    @abstractmethod
    def impl(self, region: IRegion) -> Optional[IRegion]:
        ...


@dataclass(frozen=True)
class OpsTraversal(ABC):
    """
    Special kind of Strategy which is applied to an IBlock and returns a new IBlock
    on success. Only composes with BlockTraversal
    """
    s: Strategy

    def apply(self, block: IBlock) -> Optional[IBlock]:
        return self.impl(block)

    @abstractmethod
    def impl(self, block: IBlock) -> Optional[IBlock]:
        ...


@dataclass(frozen=True)
class regionN(RegionsTraversal):
    """
    Descend into the region with index `n` to apply a Strategy inside. 
    The Strategy and where exactly it is to be applied is determined by
    `block_trav`. Afterwards a replacement for the matched op is built with
    an updated region.
    """
    block_trav: BlocksTraversal
    n: int

    def impl(self, op: IOp) -> RewriteResult:
        if len(op.regions) <= self.n:
            return failure(self)
        new_region = self.block_trav.apply(op.regions[self.n])
        if isinstance(new_s := new_region, Strategy):
            return new_s
        if new_region is None:
            return failure(self)
        regions: List[IRegion] = op.regions[:self.n] + [
            new_region
        ] + op.regions[self.n + 1:]

        result = new_op(op_type=op.op_type,
                        operands=list(op.operands),
                        result_types=op.result_types,
                        attributes=op.attributes,
                        successors=list(op.successors),
                        regions=regions)
        return success(result)


@dataclass(frozen=True)
class firstRegion(regionN):
    """
    regionN with n = 0
    """
    block_trav: BlocksTraversal
    n: int = 0


@dataclass(frozen=True)
class regionsTopToBottom(RegionsTraversal):
    """
    Descend successively into all regions, in top to bottom order to apply a Strategy inside. 
    The Strategy and where exactly it is to be applied is determined by
    `block_trav`. Afterwards a replacement for the matched op is built with
    an updated region.
    """
    block_trav: BlocksTraversal

    def impl(self, op: IOp) -> RewriteResult:
        for region_idx in range(0, len(op.regions)):
            rr = regionN(self.block_trav, region_idx).apply(op)
            if isinstance(new_s := rr, Strategy):
                return new_s
            if rr.isSuccess():
                return rr
        return failure(self)


@dataclass(frozen=True)
class regionsBottomToTop(RegionsTraversal):
    """
    Descend successively into all regions, in bottom to top order to apply a Strategy inside. 
    The Strategy and where exactly it is to be applied is determined by
    `block_trav`. Afterwards a replacement for the matched op is built with
    an updated region.
    """
    block_trav: BlocksTraversal

    def impl(self, op: IOp) -> RewriteResult:
        for region_idx in reversed(range(0, len(op.regions))):
            rr = regionN(self.block_trav, region_idx).apply(op)
            if rr.isSuccess():
                return rr
        return failure(self)


@dataclass(frozen=True)
class blockN(BlocksTraversal):
    """
    Descend into the block with index `n` to apply a Strategy inside. 
    The Strategy and where exactly it is to be applied is determined by
    `op_trav`. Afterwards a replacement for the matched region is built with
    an updated block.
    """
    op_trav: OpsTraversal
    n: int

    def impl(self, region: IRegion) -> Optional[IRegion]:
        if len(region.blocks) <= self.n:
            return None
        new_block = self.op_trav.apply(region.blocks[self.n])
        if isinstance(new_s := new_block, Strategy):
            return new_s
        if new_block is None:
            return None
        blocks = region.blocks[:self.n] + [new_block
                                           ] + region.blocks[self.n + 1:]
        return IRegion(blocks)


@dataclass(frozen=True)
class firstBlock(blockN):
    """
    blockN with n = 0
    """
    op_trav: OpsTraversal
    n: int = 0


@dataclass(frozen=True)
class blocksTopToBottom(BlocksTraversal):
    """
    Descend successively into all blocks, in top to bottom order 
    to apply a Strategy inside. 
    The Strategy and to which ops exactly it is to be applied is determined by
    `op_trav`. Afterwards a replacement for the matched region is built with
    an updated block.
    """
    op_trav: OpsTraversal

    def impl(self, region: IRegion) -> Optional[IRegion]:
        for block_idx in range(0, len(region.blocks)):
            new_block = blockN(self.op_trav, block_idx).apply(region)
            if new_block is None:
                continue
            return new_block
        return None


@dataclass(frozen=True)
class blocksBottomToTop(BlocksTraversal):
    """
    Descend successively into all blocks, in bottom to top order 
    to apply a Strategy inside. 
    The Strategy and to which ops exactly it is to be applied is determined by
    `op_trav`. Afterwards a replacement for the matched region is built with
    an updated block.
    """
    op_trav: OpsTraversal

    def impl(self, region: IRegion) -> Optional[IRegion]:
        for block_idx in range(0, len(region.blocks)):
            new_block = blockN(self.op_trav, block_idx).apply(region)
            if new_block is None:
                continue
            return new_block
        return None


@dataclass(frozen=True)
class blocksCFG(BlocksTraversal):
    """
    Walk the blocks in a region in the order specified by their 
    control flow graph (depth first) to apply a Strategy inside. 
    The Strategy and to which ops exactly it is to be applied is determined by
    `op_trav`. Afterwards a replacement for the matched region is built with
    an updated block. 
    """
    op_trav: OpsTraversal
    # Entry block is always block#0
    cur_idx: int = 0

    def impl(self, region: IRegion) -> Optional[IRegion]:
        # try to apply to this block
        new_block = blockN(self.op_trav, self.cur_idx).apply(region)
        if new_block is not None:
            return new_block
        # try to apply to successors:
        if len(region.blocks[self.cur_idx].ops) > 0 and len(
            successors := region.blocks[self.cur_idx].ops[-1].successors) > 0:
            for successor in successors:
                if successor in region.blocks:
                    new_block = blocksCFG(
                        self.op_trav,
                        region.blocks.index(successor)).apply(region)
                    if new_block is not None:
                        return new_block
                else:
                    raise Exception("invalid successor")

        return None


@dataclass(frozen=True)
class opN(OpsTraversal):
    """
    Apply a strategy `s` to the op with index `n` in an IBlock. Afterwards builds
    an new IBlock from the result. 
    """
    s: Strategy
    n: int

    def impl(self, block: IBlock) -> Optional[IBlock]:
        if len(block.ops) <= self.n:
            return None
        rr = self.s.apply(block.ops[self.n])
        if isinstance(new_s := rr, Strategy):
            return new_s

        if rr.isSuccess():
            nested_ops: List[IOp] = list(block.ops)

            completed_replacements: List[IOpReplacement] = []
            for repl_idx, replacement in enumerate(rr.replacements):
                if replacement.matched_op in nested_ops:
                    # walk all operations of the new block and
                    # add replacements for the uses of the op we are replacing
                    if len(replacement.replacement_ops) > 0:
                        add_replacements: Callable[[IOp], None] = partial(
                            self._add_replacements_for_uses_of_matched_op,
                            rr.replacements, repl_idx)
                        for op in nested_ops:
                            op.walk(add_replacements)

                        # We never want to materialize rewrite.id operations
                        replacement.replacement_ops = [
                            op for op in replacement.replacement_ops
                            if op.op_type != RewriteId
                        ]

                    # Actually replacing the matched op with replacement ops
                    i = nested_ops.index(replacement.matched_op)
                    nested_ops[i:i + 1] = replacement.replacement_ops

                    completed_replacements.append(replacement)
                else:
                    raise Exception(
                        "replacement out of scope, could not be applied")

            for replacement in completed_replacements:
                rr.replacements.remove(replacement)

            return IBlock.from_iblock(nested_ops, block)
        return None

    @staticmethod
    def _add_replacements_for_uses_of_matched_op(
            replacements: MutableSequence[IOpReplacement], repl_idx: int,
            user_op: IOp):
        replacement = replacements[repl_idx]
        for idx, matched_op_value in enumerate(replacement.matched_op.results):
            if matched_op_value in user_op.operands:
                replacements.insert(
                    repl_idx + 1,
                    IOpReplacement(
                        user_op,
                        from_op(
                            user_op,
                            env={
                                matched_op_value:
                                (replacement.replacement_ops[-1].results[idx]
                                 if not replacement.replacement_ops[-1].op_type
                                 == RewriteId else
                                 replacement.replacement_ops[-1].operands[0])
                            })))
                # TODO: think about whether we have to do this in the replacement ops as well
                if user_op in (matched_op_list := [
                    repl.matched_op for repl in replacements[repl_idx + 2:]
                ]):
                    index = matched_op_list.index(user_op)
                    if not replacement.replacement_ops[-1].op_type == RewriteId:
                        replacements[
                            repl_idx + 1 +
                            index].matched_op = replacement.replacement_ops[-1]
                    else:
                        print(
                            "ERROR in the rewriting! Removing a replacement because its match went stale."
                        )
                        replacements[repl_idx + 1 + index:repl_idx + 1 +
                                     index + 1] = []


@dataclass(frozen=True)
class firstOp(opN):
    """
    opN with n = 0
    """
    s: Strategy
    n: int = 0


@dataclass(frozen=True)
class opsTopToBottom(OpsTraversal):
    """
    Try to apply a strategy `s` to all ops in an IBlock starting at the top. 
    After successful application builds an new IBlock from the result. 
    """
    s: Strategy
    start_index: int = 0
    skips: int = 0

    def impl(self, block: IBlock) -> Optional[IBlock]:
        for op_idx in range(0, len(block.ops)):
            if op_idx < self.start_index:
                continue
            new_block: Optional[IBlock] = opN(self.s, op_idx).apply(block)
            if new_block is not None:
                if self.skips > 0 and op_idx < len(block.ops):
                    return opsTopToBottom(self.s, op_idx + 1,
                                          self.skips - 1).apply(block)
                return new_block
        return None

@dataclass(frozen=True)
class allOpsTopToBottom(OpsTraversal):
    """
    """
    s: Strategy
    start_index: int = 0
    skips: int = 0

    # This could be implemented way more efficiently, but this is much more concise
    def impl(self, block: IBlock) -> Optional[IBlock]:
        if self.start_index == len(block.ops):
            # We visited all ops successfully previously
            return block
        new_block = opN(self.s, self.start_index).apply(block)
        if new_block is None:
            return None
        # To not miss any ops when the block is modified, we advance by ops_added +1.
        # i.e. we skip newly created ops and avoid skipping an op if the matched op was deleted
        ops_added = len(new_block.ops) - len(block.ops)
        return allOpsTopToBottom(self.s, start_index=self.start_index+1+ops_added).apply(new_block)

@dataclass(frozen=True)
class opsBottomToTop(OpsTraversal):
    """
    Try to apply a strategy `s` to all ops in an IBlock starting at the bottom. 
    After successful application builds an new IBlock from the result. 
    """
    s: Strategy
    start_index: int = -1
    skips: int = 0

    def impl(self, block: IBlock) -> Optional[IBlock]:

        for op_idx in reversed(range(0, len(block.ops))):
            if self.start_index != -1 and op_idx > self.start_index:
                continue
            new_block: Optional[IBlock] = opN(self.s, op_idx).apply(block)
            if new_block is not None:
                if self.skips > 0 and op_idx > 0:
                    return opsBottomToTop(self.s, op_idx - 1,
                                          self.skips - 1).apply(block)
                return new_block
        return None


@dataclass(frozen=True)
class topToBottom(Strategy):
    """
    topToBottom traversal - Try to apply a strategy `s` to `op` itself and all
    ops in nested regions from top to bottom. Terminates after successful application.
    """
    s: Strategy
    skips: int = 0

    def impl(self, op: IOp) -> RewriteResult:
        if (rr := self.s.apply(op)).isSuccess():
            return rr
        rr = regionsTopToBottom(
            blocksTopToBottom(
                opsTopToBottom(topToBottom(self.s, skips=self.skips),
                               skips=self.skips))).apply(op)
        if rr.isSuccess():
            return rr

        return failure(self)

@dataclass(frozen=True)
class topToBottom_hacked(Strategy):
    """
    topToBottom traversal - Try to apply a strategy `s` to `op` itself and all
    ops in nested regions from top to bottom. Terminates after successful application.
    """
    s: Strategy
    skips: int = 0


    def impl(self, op: IOp) -> RewriteResult:

        rr = self.s.apply(op)
        if isinstance(new_s := rr, Strategy):
            return new_s
        if rr.isSuccess():
            return rr
        rr = regionsTopToBottom(
            blocksTopToBottom(
                opsTopToBottom(topToBottom_hacked(self.s, skips=self.skips),
                               skips=self.skips))).apply(op)
        if isinstance(new_s := rr, Strategy):
            return topToBottom_hacked(new_s).apply(op)
        if rr.isSuccess():
            return rr

        return failure(self)


@dataclass(frozen=True)
class topToBottom_non_rebuilding(Strategy):
    """
    Traverses the IR top to bottom but does not support modification.
    Only returns the RewriteResult of s without handling replacements.
    This is useful for partial Strategies, which just have to find an op
    to forward it to another Strategy
    """
    s: Strategy

    def impl(self, op: IOp) -> RewriteResult:

        result: Optional[RewriteResult] = None 

        def apply(op: IOp) -> bool:
            nonlocal result
            if (rr := self.s.apply(op)).isSuccess():
                result = rr
                return False
            # advance
            return True

        op.walk_abortable(apply)
        if result is not None:
            return result

        return failure(self)



@dataclass(frozen=True)
class topToBottomMulti(Strategy):
    """
    topToBottom traversal - Try to apply a strategy `s` to `op` itself and all
    ops in nested regions from top to bottom. Terminates after successful application.
    """
    s: callable([IOp], [Optional[Strategy]])
    
    skips: int = 0

    def impl(self, op: IOp) -> RewriteResult:

        # Traversal expects a multi pattern

        rr = self.s.apply(op)
        if isinstance(new_s := rr, Strategy):

            return new_s
        if rr.isSuccess():
            return rr
        rr = regionsTopToBottom(
            blocksTopToBottom(
                opsTopToBottom(topToBottom(self.s, skips=self.skips),
                               skips=self.skips))).apply(op)
        if isinstance(new_s := rr, Strategy):
            return topToBottom(new_s).apply(op)
        if rr.isSuccess():
            return rr

        return failure(self)

@dataclass(frozen=True)
class PartialStrategy:
    @abstractmethod
    def apply(self, op: IOp) -> Optional[List[IOp | ISSAValue]]:
        ...

@dataclass(frozen=True)
class multi_seq(Strategy):
    """
    Applies a PartialStrategy `s0` and initializes `s1` with the result
    of that application. Afterwards applies the initialized `s1` (s1_complete) 
    """
    s0: PartialStrategy
    s1: Type[Strategy]
    
    # Question: not obvious how to do traversals with this approach. 
    def impl(self, op: IOp) -> RewriteResult:
        args: Optional[List[IOp | ISSAValue]] = self.s0.apply(op)
        if args is None:
            return failure(self)

        s1_complete = self.s1(*args)
        return s1_complete.apply(op)


@dataclass(frozen=True)
class multi_seq2(Strategy):
    """
    Applies s0 to op and uses the replacement_ops in the RewriteResult to initialize
    s1. Then applies the initialized s1 to op.
    """
    s0: Strategy
    s1: Callable[[List[IOp]], Strategy]
    
    def impl(self, op: IOp) -> RewriteResult:
        rr = self.s0.apply(op)
        if not rr.isSuccess():
            return failure(self)
        
        s1_complete = self.s1(rr.replacements[-1].replacement_ops)
        return s1_complete.apply(op)

@dataclass(frozen=True)
class bottomToTop(Strategy):
    """
    bottomToTop traversal - Try to apply a strategy `s` to all
    ops in nested regions from bottom to top and the op itself.
    Terminates after successful application.
    """
    s: Strategy
    skips: int = 0

    def impl(self, op: IOp) -> RewriteResult:
        rr = regionsBottomToTop(
            blocksBottomToTop(
                opsBottomToTop(bottomToTop(self.s, skips=self.skips),
                               skips=self.skips))).apply(op)
        if rr.isSuccess():
            return rr
        if (rr := self.s.apply(op)).isSuccess():
            return rr

        return failure(self)


########################################################################
######################    Predicate Strategies    ######################
########################################################################


@dataclass(frozen=True)
class isa(Strategy):
    """
    Predicate Strategy checking whether on op is of a specific op_type
    """
    op_type: Type[Operation]

    def impl(self, op: IOp) -> RewriteResult:
        if op.op_type == self.op_type:
            return success(op)
        return failure(self)


@dataclass(frozen=True)
class has_attributes(Strategy):
    """
    Predicate Strategy checking whether on op has a set of attributes
    """
    attributes: List[Attribute]

    def impl(self, op: IOp) -> RewriteResult:
        for attr in self.attributes:
            if attr not in op.attributes:
                return failure(self)
        return success(op)
