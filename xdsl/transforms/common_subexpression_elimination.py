from dataclasses import dataclass, field
from typing import TypeVar

from xdsl.dialects.builtin import ModuleOp, UnregisteredOp
from xdsl.ir import Block, MLContext, Operation, Region, Use
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.traits import IsolatedFromAbove, IsTerminator, is_side_effect_free
from xdsl.transforms.dead_code_elimination import is_trivially_dead


@dataclass
class OperationInfo:
    """
    Boilerplate helper to use in KnownOps cache.

    This is to compare operations on their name, attributes, properties, results,
    operands, and matching region structure.
    """

    op: Operation

    @property
    def name(self):
        return (
            self.op.op_name.data
            if isinstance(self.op, UnregisteredOp)
            else self.op.name
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                sum(hash(i) for i in self.op.attributes.items()),
                sum(hash(i) for i in self.op.properties.items()),
                hash(tuple(i.type for i in self.op.results)),
                hash(self.op.operands),
            )
        )

    def __eq__(self, other: object):
        return (
            isinstance(other, OperationInfo)
            and hash(self) == hash(other)
            and self.name == other.name
            and self.op.attributes == other.op.attributes
            and self.op.properties == other.op.properties
            and self.op.operands == other.op.operands
            and len(self.op.results) == len(other.op.results)
            and all(r.type == o.type for r, o in zip(self.op.results, other.op.results))
            and len(self.op.regions) == len(other.op.regions)
            and all(
                s.is_structurally_equivalent(o)
                for s, o in zip(self.op.regions, other.op.regions)
            )
        )


_D = TypeVar("_D")


class KnownOps:
    """
    Cache dictionary for known operations used in CSE.
    It quacks like a dict[Operation, Operation], but uses OperationInfo of an Opetration
    as the actual key.
    """

    _known_ops: dict[OperationInfo, Operation]

    def __init__(self, known_ops: "KnownOps | None" = None):
        if known_ops is None:
            self._known_ops = {}
        else:
            self._known_ops = dict(known_ops._known_ops)

    def __getitem__(self, k: Operation):
        return self._known_ops[OperationInfo(k)]

    def __setitem__(self, k: Operation, v: Operation):
        self._known_ops[OperationInfo(k)] = v

    def __contains__(self, k: Operation):
        return OperationInfo(k) in self._known_ops

    def get(self, k: Operation, default: _D = None) -> Operation | _D:
        return self._known_ops.get(OperationInfo(k), default)

    def pop(self, k: Operation):
        return self._known_ops.pop(OperationInfo(k))


class CSEDriver:
    """
    Boilerplate class to handle and carry the state for CSE.
    """

    _rewriter: Rewriter
    _to_erase: set[Operation] = field(default_factory=set)
    _known_ops: KnownOps = KnownOps()

    def __init__(self):
        self._rewriter = Rewriter()
        self._to_erase = set()
        self._known_ops = KnownOps()

    def _mark_erasure(self, op: Operation):
        self._to_erase.add(op)

    def _commit_erasures(self):
        for o in self._to_erase:
            if o.parent is not None:
                self._rewriter.erase_op(o)

    def _simplify_operation(self, op: Operation):
        """
        Simplify a single operation: replace it by a corresponding known operation in
        scope, if any.
        Also just delete dead operations.
        """
        # Don't simplify terminators.
        if op.has_trait(IsTerminator):
            return

        # If the operation is already trivially dead just add it to the erase list.
        if is_trivially_dead(op):
            self._mark_erasure(op)
            return

        # Don't simplify operations with regions that have multiple blocks.
        # MLIR doesn't either at time of writing :)
        if any(len(region.blocks) > 1 for region in op.regions):
            return

        # Here, MLIR says something like "if the operation has side effects"
        # Using more generics analytics; and has fancier analysis for that case,
        # where it might simplify some side-effecting operations still.
        # Doesmn't mean we can't just simplify what we can with our simpler model :)
        if not is_side_effect_free(op):
            return

        # This operation rings a bell!
        if existing := self._known_ops.get(op):

            # Just replace results
            def wasVisited(use: Use):
                return use.operation not in self._known_ops

            for o, n in zip(op.results, existing.results, strict=True):
                if all(wasVisited(u) for u in o.uses):
                    o.replace_by(n)

            # If no uses remain, we can mark this operation for erasure
            if all(not r.uses for r in op.results):
                self._mark_erasure(op)

            return

        # First time seeing this one, noting it down!
        self._known_ops[op] = op

    def _simplify_block(self, block: Block):
        for op in block.ops:

            if op.regions:
                might_be_isolated = isinstance(op, UnregisteredOp) or (
                    op.get_trait(IsolatedFromAbove) is not None
                )
                # If we can't be sure the op isn't isolated, we assume it is for safety
                if might_be_isolated:
                    # Then save the current scope for later, but continue inside with a
                    # blank slate
                    old_scope = self._known_ops
                    self._known_ops = KnownOps()
                    for region in op.regions:
                        self._simplify_region(region)
                    self._known_ops = old_scope
                else:
                    for region in op.regions:
                        self._simplify_region(region)

            self._simplify_operation(op)

    def _simplify_region(self, region: Region):
        if not region.blocks:
            return

        if len(region.blocks) == 1:

            old_scope = self._known_ops
            self._known_ops = KnownOps(self._known_ops)

            self._simplify_block(region.block)

            self._known_ops = old_scope

    def simplify(self, thing: Operation | Block | Region):
        match thing:
            case Operation():
                for region in thing.regions:
                    self._simplify_region(region)
            case Block():
                self._simplify_block(thing)
            case Region():
                self._simplify_region(thing)
        self._commit_erasures()


def cse(thing: Operation | Block | Region):
    CSEDriver().simplify(thing)


class CommonSubexpressionElimination(ModulePass):
    name = "cse"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        cse(op)
