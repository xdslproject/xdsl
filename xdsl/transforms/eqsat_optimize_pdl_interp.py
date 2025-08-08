from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field

from xdsl.context import Context
from xdsl.dialects import builtin, pdl_interp
from xdsl.ir import Block, Region, SSAValue, Use
from xdsl.irdl.dominance import DominanceInfo
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.traits import SymbolTable


class FilteredDominanceInfo(DominanceInfo):
    """
    A DominanceInfo subclass that filters out finalize blocks from post-dominance computation.
    This is used to ignore edges that lead to finalize blocks when computing post-dominance.
    """

    def __init__(
        self,
        region: Region,
        is_finalize_block: Callable[[Block], bool],
        compute_postdominance: bool = False,
    ):
        self.is_finalize_block = is_finalize_block
        super().__init__(region, compute_postdominance)

    def _get_flow_predecessors(self, block: Block) -> Sequence[Block]:
        """Override to filter out finalize blocks from successors when computing post-dominance."""
        if self._is_postdominance:
            # Filter out successors that are finalize blocks
            if block.last_op:
                filtered_successors = [
                    succ
                    for succ in block.last_op.successors
                    if not self.is_finalize_block(succ)
                ]
                return filtered_successors
            else:
                return []
        else:
            # For regular dominance, use the original behavior
            return block.predecessors()


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
        changed = True
        postdominance = None
        for op in self.matcher.walk(reverse=True):
            if isinstance(op, pdl_interp.GetDefiningOpOp):
                # changed = self.optimize_name_constraints(op)
                ...
            elif isinstance(op, pdl_interp.AreEqualOp):
                if changed:
                    postdominance = FilteredDominanceInfo(
                        self.matcher.body,
                        self.is_finalize_block,
                        compute_postdominance=True,
                    )
                    changed = False
                assert postdominance is not None
                changed = self.optimize_equality_constraints(op, postdominance)

    def optimize_name_constraints(
        self,
        gdo: pdl_interp.GetDefiningOpOp,
    ) -> bool:
        assert (gdo_block := gdo.parent_block())
        if not (
            isinstance(gdo_block.last_op, pdl_interp.IsNotNullOp)
            and self.is_finalize_block(gdo_block.last_op.false_dest)
        ):
            return False

        check_blocks = set[Block]()
        names = set[builtin.StringAttr]()

        for use in gdo.input_op.uses:
            check_op = use.operation
            if isinstance(check_op, pdl_interp.SwitchOperationNameOp):
                finalizes = self.is_finalize_block(check_op.default_dest), f"{check_op}"
                names.update(check_op.case_values)
            elif isinstance(check_op, pdl_interp.CheckOperationNameOp):
                finalizes = self.is_finalize_block(check_op.false_dest)
                names.add(check_op.operation_name)
            else:
                continue
            if finalizes:
                # when this `check_op` is reached, the pattern will either fail to match or
                # the operation will be constrained to a name contained in `names`.
                assert (check_block := check_op.parent_block())
                check_blocks.add(check_block)

        worklist: list[Block] = list(gdo_block.last_op.successors)
        additional_gdo_on_path = False
        while worklist:
            block = worklist.pop()
            for op in block.ops:
                if isinstance(op, pdl_interp.GetDefiningOpOp):
                    additional_gdo_on_path = True
            if block in check_blocks:
                continue
            terminator = block.last_op
            assert terminator, "Expected each block to have a terminator"
            if isinstance(terminator, pdl_interp.RecordMatchOp):
                # There is a rewrite that is triggered without ever constraining the operation's name.
                # This means we cannot move a aggregate name check right after the GDO.
                return False
            worklist.extend(
                terminator.successors
            )  # This terminates since CFG is acyclic.

        # To prevent inserting too many new checks, we only insert if there is actual danger
        # of exponential blowup due to the presence of multiple GDOs on the path.
        if not additional_gdo_on_path:
            return False

        self.insert_name_constraint(
            names,
            gdo,
        )

        return True

    def optimize_equality_constraints(
        self,
        are_equal_op: pdl_interp.AreEqualOp,
        postdominance: DominanceInfo,
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

        block_dependencies: set[SSAValue] = set()
        for op in are_equal_block.ops:
            for result in op.operands:
                block_dependencies.add(result)

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
            if not postdominance.postdominates(are_equal_block, pred):
                insertion_edge = outgoing_edge
                break
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
                should_move_block = True
            if not should_move_block:
                continue
            blocks_to_move.append(pred)
            outgoing_edges.append(outgoing_edge)
            incoming_edges.append(incoming_edge)

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
            # print(f"moving block with terminator: {block.last_op}")
            # print(f"insertion_edge operation: {insertion_edge.operation}")
            # print(f"incoming operation: {incoming.operation}")
            # print(f"outgoing operation: {outgoing.operation}")
            if incoming.operation is insertion_edge.operation:
                # print("ordering is already correct, skipping")
                # print()
                insertion_edge = outgoing
                continue
            insertion_dest = insertion_edge.operation.successors[insertion_edge.index]
            insertion_edge.operation.successors[insertion_edge.index] = block
            outgoing_dest = outgoing.operation.successors[outgoing.index]
            outgoing.operation.successors[outgoing.index] = insertion_dest
            if incoming.operation != insertion_edge.operation:
                incoming.operation.successors[incoming.index] = outgoing_dest
            insertion_edge = outgoing.operation._successor_uses[outgoing.index]  # pyright: ignore[reportPrivateUsage]
            # print()

        return True

    def is_finalize_block(self, block: Block) -> bool:
        if block in self.finalize_blocks:
            return True
        if len(block.ops) == 1 and isinstance(block.last_op, pdl_interp.FinalizeOp):
            self.finalize_blocks.add(block)
            return True
        return False

    def insert_name_constraint(
        self,
        valid_names: Collection[builtin.StringAttr],
        gdo: pdl_interp.GetDefiningOpOp,
    ):
        gdo_block = gdo.parent_block()
        # at this point, we also know that the false_dest of the terminator is a finalize block
        assert gdo_block
        assert isinstance(terminator := gdo_block.last_op, pdl_interp.IsNotNullOp)

        continue_dest = terminator.true_dest
        if len(valid_names) == 1:
            new_check_op = pdl_interp.CheckOperationNameOp(
                next(iter(valid_names)),
                gdo.input_op,
                trueDest=continue_dest,
                falseDest=terminator.false_dest,
            )
        else:
            new_check_op = pdl_interp.SwitchOperationNameOp(
                valid_names,
                gdo.input_op,
                default_dest=terminator.false_dest,
                cases=[continue_dest for _ in range(len(valid_names))],
            )
        new_block = Block((new_check_op,))
        self.matcher.body.insert_block_after(new_block, gdo_block)

        rewriter = Rewriter()
        rewriter.replace_op(
            terminator,
            pdl_interp.IsNotNullOp(
                terminator.value, trueDest=new_block, falseDest=terminator.false_dest
            ),
        )
