from dataclasses import dataclass, field
from typing import TypeVar

from xdsl.dialects.builtin import ModuleOp, UnregisteredOp
from xdsl.ir import Block, MLContext, Operation, Region, Use
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.traits import IsolatedFromAbove, IsTerminator, Pure
from xdsl.transforms.dead_code_elimination import is_trivially_dead


@dataclass
class OperationInfo:
    """
    Boilerplate helper to use in KnownOps cache.

    This is to compare operations on their name, attributes, properties, results,
    operands, and matching region structure.
    """

    op: Operation

    def __hash__(self):
        if isinstance(self.op, UnregisteredOp):
            name = self.op.op_name.data
        else:
            name = self.op.name
        return hash(
            (
                name,
                sum(hash(i) for i in self.op.attributes.items()),
                sum(hash(i) for i in self.op.properties.items()),
                hash(tuple(i.type for i in self.op.results)),
                hash(tuple(i for i in self.op.operands)),
            )
        )

    def __eq__(self, other: object):
        if not isinstance(other, OperationInfo):
            return False
        if hash(self) != hash(other):
            return False
        sregions = self.op.regions
        oregions = other.op.regions
        if len(sregions) != len(oregions):
            return False
        return all(s.is_structurally_equivalent(o) for s, o in zip(sregions, oregions))


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


@dataclass
class CSEDriver:
    """
    Boilerplate class to handle and carry the state for CSE.
    """

    rewriter: Rewriter
    to_erase: set[Operation] = field(default_factory=set)
    known_ops: KnownOps = KnownOps()

    def simplify_operation(self, op: Operation):
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
            self.to_erase.add(op)
            return

        # Don't simplify operations with regions that have multiple blocks.
        # MLIR doesn't either at time of writing :)
        if any(len(region.blocks) > 1 for region in op.regions):
            return

        # Here, MLIR says something like "if the operation has side effects"
        # Using more generics analytics; and has fancier analysis for that case,
        # where it might simplify some side-effecting operations still.
        # Doesmn't mean we can't just simplify what we can with our simpler model :)
        if not op.has_trait(Pure):
            return

        # This operation rings a bell!
        if existing := self.known_ops.get(op):

            # Just replace results
            def wasVisited(use: Use):
                return self.known_ops.get(use.operation) is None

            for o, n in zip(op.results, existing.results, strict=True):
                if all(wasVisited(u) for u in o.uses):
                    o.replace_by(n)

            # If no uses remain, we can mark this operation for erasure
            if all(not r.uses for r in op.results):
                self.to_erase.add(op)

            return

        # First time seeing this one, noting it down!
        self.known_ops[op] = op

    def simplify_block(self, block: Block):
        for op in block.ops:

            if op.regions:
                might_be_isolated = isinstance(op, UnregisteredOp) or (
                    op.get_trait(IsolatedFromAbove) is not None
                )
                # If we can't be sure the op isn't isolated, we assume it is for safety
                if might_be_isolated:
                    # Then save the current scope for later, but continue inside with a
                    # blank slate
                    old_scope = self.known_ops
                    self.known_ops = KnownOps()
                    for region in op.regions:
                        self.simplify_region(region)
                    self.known_ops = old_scope
                else:
                    for region in op.regions:
                        self.simplify_region(region)

            self.simplify_operation(op)

    def simplify_region(self, region: Region):
        if not region.blocks:
            return

        if len(region.blocks) == 1:

            old_scope = self.known_ops
            self.known_ops = KnownOps(self.known_ops)

            self.simplify_block(region.block)

            self.known_ops = old_scope


class CommonSubexpressionElimination(ModulePass):
    name = "cse"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        rewriter = Rewriter()
        driver = CSEDriver(rewriter)
        for region in op.regions:
            driver.simplify_region(region)
        for o in driver.to_erase:
            if o.parent is not None:
                rewriter.erase_op(o)
