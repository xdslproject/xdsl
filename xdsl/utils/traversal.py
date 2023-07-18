from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set

from xdsl.ir import Block, Region


def predecessors(region: Region) -> Dict[Block, Set[Block]]:
    """
    Returns a dictionary of blocks in an SSACFG region and their predecessors.
    If a block has no predecessors, it is not included in the dictionary.
    This also excludes the entry block if it is not a successor of a contained operation.
    """

    if not region.blocks:
        return {}

    preds: Dict[Block, Set[Block]] = defaultdict(set)

    for block in region.blocks:
        if block.last_op:
            for succ in block.last_op.successors:
                preds[succ].add(block)

    return preds


def postorder(region: Region) -> List[Block]:
    """
    Returns the list of reachable blocks of an SSACFG region when visited in postorder.
    The entry block is always included, if it exists.
    """
    if not region.blocks:
        return []

    if len(region.blocks) == 1:
        return [region.block]

    visited: Set[Block] = set()
    order: List[Block] = []

    # use DFS traversal to gather up blocks after visiting their successors
    def dfs(block: Block):
        visited.add(block)
        if block.last_op:
            for succ in block.last_op.successors:
                if succ not in visited:
                    dfs(succ)
        order.append(block)

    dfs(region.blocks[0])

    return order
