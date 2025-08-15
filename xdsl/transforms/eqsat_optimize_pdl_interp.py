from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.ir import Block, SSAValue, Use
from xdsl.passes import ModulePass
from xdsl.traits import SymbolTable


@dataclass(frozen=True)
class EqsatOptimizePDLInterp(ModulePass):
    name = "eqsat-optimize-pdl-interp"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        matcher = SymbolTable.lookup_symbol(op, "matcher")
        assert isinstance(matcher, pdl_interp.FuncOp)
        assert matcher is not None, "matcher function not found"

        mo = MatcherOptimizer(matcher)
        mo.optimize()


@dataclass
class MatcherOptimizer:
    matcher: pdl_interp.FuncOp

    finalize_blocks: set[Block] = field(default_factory=set[Block])

    def optimize(self):
        for op in self.matcher.walk(reverse=True):
            if isinstance(op, pdl_interp.AreEqualOp):
                self.optimize_equality_constraints(op)

    def optimize_equality_constraints(
        self,
        are_equal_op: pdl_interp.AreEqualOp,
    ) -> bool:
        if not self.is_finalize_block(are_equal_op.false_dest):
            # If the false destination is not a finalize block, we cannot optimize this AreEqualOp.
            return False

        assert (are_equal_block := are_equal_op.parent_block())
        blocks_to_move = [are_equal_block]

        # We can always pick only one of the predecessors to move up.
        # Forked paths will never define values that are used after paths
        # are joined again, since there are no block arguments/phi nodes.
        # The final destination of the blocks will always be along a
        # non-split path at the post-dominator frontier:
        assert (incoming_edge := next(iter(are_equal_block.uses), None)) is not None
        incoming_edges: list[Use] = [incoming_edge]

        outgoing_edges = [are_equal_op._successor_uses[0]]  # pyright: ignore[reportPrivateUsage]

        block_dependencies: set[SSAValue] = {
            result for op in are_equal_block.ops for result in op.operands
        }
        insertion_edge = None
        while True:
            should_move_block = False
            pred = incoming_edge.operation.parent_block()
            assert pred is not None
            outgoing_edge = incoming_edge
            incoming_edge = next(iter(pred.uses), None)
            if incoming_edge is None:
                insertion_edge = outgoing_edge
                break
            for succ in outgoing_edge.operation.successors:
                if succ == outgoing_edge.operation.successors[outgoing_edge.index]:
                    continue
                if not self.is_finalize_block(succ):
                    # there is a successor that is not the block on the path to the AreEqualOp,
                    # this means we cannot move blocks past this point.
                    return False
            for op in pred.ops:
                is_gdo = isinstance(op, pdl_interp.GetDefiningOpOp)
                for result in op.results:
                    if result in block_dependencies:
                        if is_gdo:
                            # The result of the GDO is used. We can stop iterating and
                            # will insert the moved blocks after this one.
                            insertion_edge = outgoing_edge
                            break
                        should_move_block = True  # block defines a value that is used by the blocks to move.
                        continue
            if insertion_edge is not None:
                break
            terminator = pred.last_op
            assert terminator is not None, "Expected each block to have a terminator"
            if any(
                self.is_finalize_block(succ) for succ in terminator.successors
            ) and any(operand in block_dependencies for operand in terminator.operands):
                # The block contains a check on one of the values that will be moved,
                # this means we need to move this block as well.
                if not isinstance(terminator, pdl_interp.AreEqualOp):
                    # We make an exception for AreEqualOp: this allows blocks checking equality to be reordered.
                    should_move_block = True
            if not should_move_block:
                continue
            blocks_to_move.append(pred)
            outgoing_edges.append(outgoing_edge)
            incoming_edges.append(incoming_edge)
            for op in pred.ops:
                block_dependencies.update(op.operands)

        assert insertion_edge is not None

        #  initial    move  2      move  4
        #  ┌─────┐    ┌─────┐      ┌─────┐
        #  │0    │    │0    │      │0    │    * blocks_to_move
        #  └──┬──┘    └──┬──┘      └──┬──┘    < insertion_edge
        #     │<     ┌───┘┌───┐       │
        #  ┌──▼──┐   │┌───▼─┐ │    ┌──▼──┐
        #  │1    │   ││1    │ │<   │2  * │    (While blocks are visually reordered
        #  └──┬──┘   │└──┬──┘ │    └──┬──┘    in the diagram, they are not moved in
        #     │      └──┐└───┐│   ┌───┘┌──┐   the block linked list structure. Only
        #  ┌──▼──┐    ┌─▼───┐││   │┌───▼─┐│   the successor edges are updated.)
        #  │2  * │    │2  * │││   ││1    ││
        #  └──┬──┘    └──┬──┘││   │└──┬──┘│
        #     │          └───┼┘   │   │   │<
        #  ┌──▼──┐       ┌───┘    │┌──▼──┐│
        #  │3    │    ┌──▼──┐     ││3    ││
        #  └──┬──┘    │3    │     │└──┬──┘│
        #     │       └──┬──┘     │   ▼   │
        #  ┌──▼──┐       │        └───┐   │
        #  │4  * │    ┌──▼──┐      ┌──▼──┐│
        #  └──┬──┘    │4  * │      │4  * ││
        #     │       └──┬──┘      └──┬──┘│
        #     ▼          ▼            └───┘

        for block, incoming, outgoing in zip(
            reversed(blocks_to_move), reversed(incoming_edges), reversed(outgoing_edges)
        ):
            assert outgoing.operation.parent_block() is block
            if incoming.operation is insertion_edge.operation:
                insertion_edge = outgoing
                continue
            insertion_dest = insertion_edge.operation.successors[insertion_edge.index]
            insertion_edge.operation.successors[insertion_edge.index] = block
            outgoing_dest = outgoing.operation.successors[outgoing.index]
            outgoing.operation.successors[outgoing.index] = insertion_dest
            if incoming.operation != insertion_edge.operation:
                incoming.operation.successors[incoming.index] = outgoing_dest
            insertion_edge = outgoing.operation._successor_uses[outgoing.index]  # pyright: ignore[reportPrivateUsage]

        return True

    def is_finalize_block(self, block: Block) -> bool:
        if block in self.finalize_blocks:
            return True
        if len(block.ops) == 1 and isinstance(block.last_op, pdl_interp.FinalizeOp):
            self.finalize_blocks.add(block)
            return True
        return False
