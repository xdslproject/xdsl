from xdsl.ir import Block, Region


class DominanceInfo:
    """
    Computes and exposes the dominance relation amongst blocks of a region.

    https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332
    """

    _dominfo: dict[Block, set[Block]]

    def __init__(self, region: Region):
        """
        Compute (improper) dominance.

        https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332
        """

        self._dominfo = {}

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
        self._dominfo[entry] = {entry}

        # Instantiate other blocks dominators to all blocks
        for b in blocks:
            self._dominfo[b] = set(region.blocks)

        # Iteratively filter out dominators until it converges
        changed = True
        while changed:
            changed = False
            for b in blocks:
                oldie = self._dominfo[b].copy()
                self._dominfo[b] = {b} | (
                    set[Block].intersection(*(self._dominfo[p] for p in pred[b]))
                    if pred[b]
                    else set()
                )
                if oldie != self._dominfo[b]:
                    changed = True

    def properly_dominates(self, a: Block, b: Block) -> bool:
        """
        Return if `a` *properly* ("strictly") dominates `b`.
        i.e., if it dominates `b` and is not `b`.
        """
        if a is b:
            return False
        return self.dominates(a, b)

    def dominates(self, a: Block, b: Block) -> bool:
        """
        Return if `a` dominates `b`.
        """
        return a in self._dominfo[b]


def _properly_dominates_block(a: Block, b: Block) -> bool:
    """
    Returns true if block `a` properly dominates block `b`.
    """
    if a is b:
        return False
    if a.parent is None:
        raise ValueError("Block `a` has no parent region")
    region = a.parent
    bb = b
    while bb.parent is not a.parent:
        parent = bb.parent_block()
        if parent is None:
            return False
        if parent is a:
            return True
        bb = parent

    return DominanceInfo(region).properly_dominates(a, b)


def properly_dominates(a: Block, b: Block) -> bool:
    return _properly_dominates_block(a, b)
