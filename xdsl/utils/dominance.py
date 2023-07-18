from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, cast

from xdsl.ir import Block, Region
from xdsl.utils import traversal


def _intersect(
    block1: Block, block2: Block, doms: Dict[Block, Block], order: List[Block]
) -> Block:
    """
    Helper method performing fast intersection in the immediate dominators tree to find
    the common immediate dominator for block1 and block2.

    This relies on 2 guarantees:
    - Having the order set in reverse postorder (RPO) guarantees the immediate
      predecessor of a block has a lower index than itself.
      The entry block has the lowest RPO index (i.e., 0).

    - The in-progress constructed dominator tree (doms) is guaranteed to contain the
      most up-to-date information since it is also built following RPO, which means all
      predecessors of a block are already in the tree.
      The entry block is processed first.

    Thus, using two "fingers" to freely move in the tree means that we can trace our way
    towards its root one while one index is larger than the other.
    This either stops at the entry block (root of tree) or earlier if another common
    closest immediate dominator is found.
    """
    index = {block: idx for idx, block in enumerate(order)}

    finger1: Block = block1
    finger2: Block = block2

    while index[finger1] is not index[finger2]:
        while index[finger1] > index[finger2]:
            finger1 = doms[finger1]
        while index[finger2] > index[finger1]:
            finger2 = doms[finger2]

    return finger1


def dominance_tree(region: Region) -> Dict[Block, Block | None]:
    """
    Creates the dominance tree for an SSACFG region.
    This is represented as a map of each block to its immediate dominator (IDom).

    Unreachable blocks are not included in the results.
    The entry block is always included, if it exists, since it dominates all reachable
    blocks in a region.

    Finding the dominator set for a block N involves traversing the tree up to the entry
    block, adding the immediate dominators found and including itself in set at the end
    to satisfy proper dominance.
    The entry block has no immediate dominator an

    This is described by:

        Dom(N) = {N}∪IDom(N)∪IDom(IDom(N)))∪...∪{entry}

    where IDom is the mapping returned by this function representing the dominance tree.

    This is based on the "Engineered Algorithm" from the paper:
    "A Simple, Fast Dominance Algorithm" by Cooper, Harvey and Kennedy
    https://hdl.handle.net/1911/96345
    """
    if not region.blocks:
        return {}

    entry = region.blocks[0]

    if len(region.blocks) == 1:
        return {entry: None}

    order = traversal.postorder(region)
    order.reverse()

    preds = traversal.predecessors(region)

    doms: Dict[Block, Block] = defaultdict(None)

    # temporarily add entry as its own dominator (i.e., proper dominance)
    doms[entry] = entry

    changed: bool = True
    while changed:
        changed = False
        for block in order[1:]:  # skip entry, guaranteed to be first in RPO
            block_preds = list(preds[block])

            # pick the first processed predecessor
            # where processed means that we have already calculated its idom
            new_idom = None
            for pred in block_preds:
                if pred in doms:
                    new_idom = pred
                    break

            # RPO traversal guarantees that we have already processed a predecessor
            assert new_idom

            block_preds.remove(new_idom)
            for pred in block_preds:
                if pred in doms:
                    new_idom = _intersect(pred, new_idom, doms, order)

            # update idom if not already calculated or produced a different value
            if block not in doms or doms[block] is not new_idom:
                doms[block] = new_idom
                changed = True

    # entry block has no immediate dominator
    idoms = cast(Dict[Block, Block | None], doms)
    idoms[entry] = None

    return idoms
