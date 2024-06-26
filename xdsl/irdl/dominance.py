from xdsl.ir import Block, Region


class DominanceInfo:
    _dominfo: dict[Block, set[Block]]

    def __init__(self, region: Region):
        self._dominfo = {}
        if not (region.blocks):
            return

        pred: dict[Block, set[Block]] = {}

        for b in region.blocks:
            pred[b] = set()

        for b in region.blocks:
            if b.last_op is not None:
                for s in b.last_op.successors:
                    pred[s].add(b)

        entry, *blocks = region.blocks

        self._dominfo[entry] = {entry}

        for b in blocks:
            self._dominfo[b] = set(region.blocks)
        changed = True
        while changed:
            changed = False
            for b in blocks:
                oldie = self._dominfo[b].copy()
                self._dominfo[b] = {b} | set[Block].intersection(
                    *(self._dominfo[p] for p in pred[b])
                )
                if oldie != self._dominfo[b]:
                    changed = True

    def properly_dominates(self, a: Block, b: Block) -> bool:
        if a is b:
            return False
        return self.dominates(a, b)

    def dominates(self, a: Block, b: Block) -> bool:
        return a in self._dominfo[b]


def _region_properly_dominates_block(a: Block, b: Block) -> bool:
    """
    Returns true if block `a` properly dominates block `b`, *assuming they are in the same region*
    """
    if a.parent is None:
        raise ValueError("Block `a` has no parent region")
    if b.parent is not a.parent:
        raise ValueError("Blocks `a` and `b` are not in the same region")
    if a is b:
        return False
    if a.last_op is None:
        return False
    return DominanceInfo(a.parent).properly_dominates(a, b)


def properly_dominates(a: Block, b: Block) -> bool:
    return _region_properly_dominates_block(a, b)
