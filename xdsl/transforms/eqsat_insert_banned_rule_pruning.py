from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, eqsat, pdl_interp
from xdsl.ir import Block
from xdsl.passes import ModulePass
from xdsl.traits import IsTerminator


@dataclass(frozen=True)
class EqsatInsertBannedRulePruningPass(ModulePass):
    name = "eqsat-insert-banned-rule-pruning"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        matcher = None
        for cur in op.walk():
            if isinstance(cur, pdl_interp.FuncOp):
                if cur.sym_name.data == "matcher":
                    matcher = cur
                    break
        assert matcher is not None, "matcher function not found"

        insert_pruning(reachable=reachable_rules(matcher), op=matcher)


@dataclass(frozen=True)
class Edge:
    to: Block
    to_index: int
    fro: Block


def insert_pruning(
    reachable: dict[Block, set[builtin.SymbolRefAttr]],
    op: pdl_interp.FuncOp,
):
    queue: list[Edge] = []
    for fro in op.body.blocks:
        reachable_from = reachable[fro]
        for to_index, to in enumerate(successors(fro)):
            reachable_to = reachable[to]
            # If the collection of reachable rules changes when following the edge,
            # we need to insert a pruning operation.
            if len(reachable_from) != len(reachable_to):
                queue.append(Edge(to=to, to_index=to_index, fro=fro))

    finalize_block = Block((pdl_interp.FinalizeOp(),))

    assert op.body.first_block, "Function body must have at least one block."
    assert op.body.first_block.first_op, (
        "Function body must have at least one operation."
    )
    if not reachable[op.body.first_block]:
        return  # there are no reachable rules, so we don't need to insert anything

    # split the first block such that the arguments are kept intact.
    (b := op.body.first_block).split_before(op.body.first_block.first_op)
    assert b.next_block
    b.add_op(
        eqsat.CheckAllUnreachableOp(
            unreachable_dest=finalize_block,
            reachable_dest=b.next_block,
        )
    )
    op.body.insert_block_after(finalize_block, op.body.first_block)

    for edge in queue:
        if not (reachable[edge.to]):
            continue
        newly_unreachable_rules = list(
            reachable[edge.fro].difference(reachable[edge.to])
        )
        b = Block(
            (
                eqsat.MarkUnreachableOp(newly_unreachable_rules),
                eqsat.CheckAllUnreachableOp(
                    unreachable_dest=finalize_block,
                    reachable_dest=edge.to,
                ),
            )
        )

        op.body.insert_block_after(b, edge.fro)
        assert edge.fro.last_op
        edge.fro.last_op.successors[edge.to_index] = b


def reachable_rules(
    op: pdl_interp.FuncOp,
) -> dict[Block, set[builtin.SymbolRefAttr]]:
    reachable: dict[Block, set[builtin.SymbolRefAttr]] = {}

    for block in reversed(toposort(op.body.blocks)):
        if rule := rule_in_block(block):
            reachable[block] = {rule}
        else:
            reachable[block] = set()
        for successor in successors(block):
            if successor not in reachable:
                continue
            # Add all rules from the successor to the current block
            reachable[block].update(reachable[successor])

    return reachable


def toposort(blocks: Sequence[Block]):
    # Kahn's algorithm: https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm

    sorted: list[Block] = []  # L in Kahn's algorithm
    if not blocks:
        return sorted

    all_blocks: set[Block] = set(blocks)

    # Calculate incoming edge counts for each block
    incoming_count: dict[Block, int] = {block: 0 for block in all_blocks}
    for block in all_blocks:
        for successor in successors(block):
            if successor in all_blocks:  # Only count edges within this region
                incoming_count[successor] += 1

    # Start from the first block:
    S = {next(iter(blocks))}

    # Main loop of Kahn's algorithm
    while S:
        n = S.pop()
        sorted.append(n)

        # For each successor of n, simulate removing the edge
        for m in successors(n):
            if m in all_blocks:  # Only process blocks within this region
                incoming_count[m] -= 1
                if incoming_count[m] == 0:
                    S.add(m)

    # Check for cycles - if we haven't processed all blocks, there's a cycle
    if len(sorted) != len(all_blocks):
        raise ValueError("Graph has at least one cycle")

    return sorted


def successors(block: Block) -> list[Block]:
    terminator = block.last_op
    assert terminator is not None, "Block must contain operations."
    assert terminator.has_trait(IsTerminator), (
        "Expected the last operation of the block to be a terminator."
    )

    return list(terminator.successors)


def rule_in_block(block: Block) -> builtin.SymbolRefAttr | None:
    terminator = block.last_op
    assert terminator is not None, "Block must contain operations."
    assert terminator.has_trait(IsTerminator), (
        "Expected the last operation of the block to be a terminator."
    )
    if isinstance(terminator, pdl_interp.RecordMatchOp):
        return terminator.rewriter
    return None
