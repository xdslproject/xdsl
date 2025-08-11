from xdsl.ir import Block, Region


class DominanceInfo:
    """
    Computes and exposes the dominance relation amongst blocks of a region.

    See external [documentation](https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332).
    """

    _dominance: dict[Block, set[Block]]

    def __init__(self, region: Region):
        """
        Compute (improper) dominance.

        See external [documentation](https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332).
        """

        self._dominance = {}

        # No block, no work
        if not (region.blocks):
            return

        # Build the preceding relationship
        pred: dict[Block, set[Block]] = {}
        for b in region.blocks:
            pred[b] = set()
        for b in region.blocks:
            if b.last_op is not None:
                for s in b.last_op.successors:
                    pred[s].add(b)

        # Get entry and other blocks
        entry, *blocks = region.blocks

        # The entry block is only dominated by itself
        self._dominance[entry] = {entry}

        # Instantiate other blocks dominators to all blocks
        for b in blocks:
            self._dominance[b] = set(region.blocks)

        # Iteratively filter out dominators until it converges
        changed = True
        while changed:
            changed = False
            for b in blocks:
                old = self._dominance[b].copy()
                self._dominance[b] = {b} | (
                    set[Block].intersection(*(self._dominance[p] for p in pred[b]))
                    if pred[b]
                    else set()
                )
                if old != self._dominance[b]:
                    changed = True

    def strictly_dominates(self, a: Block, b: Block) -> bool:
        """
        Return if `a` *strictly* dominates `b`.
        i.e., if it dominates `b` and is not `b`.
        """
        if a is b:
            return False
        return self.dominates(a, b)

    def dominates(self, a: Block, b: Block) -> bool:
        """
        Return if `a` dominates `b`.
        """
        return a in self._dominance[b]


def _strictly_dominates_block(a: Block, b: Block) -> bool:
    """
    Returns true if block `a` strictly dominates block `b`(i.e., if it dominates `b` and
    is not `b`.), assuming they are in the same region.
    """
    if a is b:
        return False
    if a.parent is None:
        raise ValueError("Block `a` has no parent region")
    if a.parent is not b.parent:
        raise ValueError("Blocks `a` and `b` are not in the same region")

    return DominanceInfo(a.parent).strictly_dominates(a, b)


# This function could be deemed useless for now, but it's intended to be
# overloaded with Values and Operations
def strictly_dominates(a: Block, b: Block) -> bool:
    """
    Returns true if block `a` strictly dominates block `b`, assuming they are in the
    same region.
    """
    return _strictly_dominates_block(a, b)
