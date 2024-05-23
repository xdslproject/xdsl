from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self, TypeVar

from xdsl.dialects.builtin import ModuleOp, UnregisteredOp
from xdsl.ir import Attribute, Block, MLContext, Operation, Region, SSAValue, Use
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter, RewritePattern
from xdsl.rewriter import Rewriter
from xdsl.traits import IsolatedFromAbove, IsTerminator, Pure


class RemoveUnusedOperations(RewritePattern):
    """
    Removes operations annotated with the `Pure` trait, where results have no uses.
    """

    def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
        # Check that operation is side-effect-free
        if not op.has_trait(Pure):
            return

        # Check whether any of the results are used
        results = op.results
        for result in results:
            if len(result.uses):
                # At least one of the results is used
                return

        rewriter.erase_op(op)


@dataclass
class OperationInfo:

    name: str
    attributes: Mapping[str, Attribute]
    properties: Mapping[str, Attribute]
    result_types: Sequence[Attribute]
    operands: Sequence[SSAValue]

    @staticmethod
    def from_op(op: Operation):
        if isinstance(op, UnregisteredOp):
            name = op.op_name.data
        else:
            name = op.name
        info = OperationInfo(
            str(name),
            dict(op.attributes),
            dict(op.properties),
            [r.type for r in op.results],
            list(op.operands),
        )
        return info

    def __hash__(self):
        return (
            hash(self.name)
            + sum(hash(i) for i in self.attributes.items())
            + sum(hash(i) for i in self.properties.items())
            + sum(hash(i) for i in self.result_types)
            + sum(hash(i) for i in self.operands)
        )


_D = TypeVar("_D")


class KnownOps:
    _known_ops: dict[OperationInfo, Operation]

    def __init__(self, known_ops: Self | None = None):
        if known_ops is None:
            self._known_ops = {}
        else:
            self._known_ops = dict(known_ops._known_ops)

    def __getitem__(self, k: Operation):
        return self._known_ops[OperationInfo.from_op(k)]

    def __setitem__(self, k: Operation, v: Operation):
        self._known_ops[OperationInfo.from_op(k)] = v

    def __contains__(self, k: Operation):
        return OperationInfo.from_op(k) in self._known_ops

    def get(self, k: Operation, default: _D = None) -> Operation | _D:
        return self._known_ops.get(OperationInfo.from_op(k), default)

    def pop(self, k: Operation):
        return self._known_ops.pop(OperationInfo.from_op(k))


@dataclass
class CSEDriver:

    rewriter: Rewriter
    to_erase: set[Operation] = field(default_factory=set)
    known_ops: KnownOps = KnownOps()

    def simplify_operation(self, op: Operation):
        if op.has_trait(IsTerminator):
            return

        # Here MLIR does early check if the op can already be DCE'd away
        # Not necessary, probably more efficient

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
            rewriter.erase_op(o)
