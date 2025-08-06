from collections.abc import Sequence

from xdsl.ir import Block, Region


class DominanceInfo:
    """
    Computes and exposes the dominance relation amongst blocks of a region.

    See external [documentation](https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332).
    """

    _dominance: dict[Block, set[Block]]
    _is_postdominance: bool

    def __init__(self, region: Region, compute_postdominance: bool = False):
        """
        Compute (improper) dominance or post-dominance.

        See external [documentation](https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332).

        Args:
            region: The region to analyze
            compute_postdominance: If True, compute post-dominance instead of dominance
        """
        self._is_postdominance = compute_postdominance
        self._dominance = {}

        # No block, no work
        if not region.blocks:
            return

        if compute_postdominance:
            self._compute_with_reversed_flow(region)
        else:
            self._compute_with_forward_flow(region)

    def _get_roots_and_others(self, region: Region) -> tuple[list[Block], list[Block]]:
        """Get root blocks and other blocks based on flow direction."""
        if self._is_postdominance:
            # For post-dominance: exits are roots
            exits = [
                b for b in region.blocks if not (b.last_op and b.last_op.successors)
            ]
            if not exits:
                exits = [region.blocks[-1]]
            return exits, [b for b in region.blocks if b not in exits]
        else:
            # For dominance: entry is root
            entry, *others = region.blocks
            return [entry], others

    def _get_flow_predecessors(self, block: Block) -> Sequence[Block]:
        """Get flow predecessors based on direction (predecessors or successors)."""
        if self._is_postdominance:
            return block.last_op.successors if block.last_op else []
        else:
            return block.predecessors()

    def _compute_dominance_relation(self, region: Region):
        """Compute dominance/post-dominance using unified algorithm."""
        roots, others = self._get_roots_and_others(region)

        # Roots are dominated only by themselves
        for root in roots:
            self._dominance[root] = {root}

        # Others start dominated by all blocks
        for b in others:
            self._dominance[b] = set(region.blocks)

        # Iteratively refine
        changed = True
        while changed:
            changed = False
            for b in others:
                old = self._dominance[b].copy()
                flow_preds = self._get_flow_predecessors(b)
                self._dominance[b] = {b} | (
                    set[Block].intersection(*(self._dominance[p] for p in flow_preds))
                    if flow_preds
                    else set()
                )
                if old != self._dominance[b]:
                    changed = True

    def _compute_with_forward_flow(self, region: Region):
        """Compute dominance using forward flow (predecessors)."""
        self._compute_dominance_relation(region)

    def _compute_with_reversed_flow(self, region: Region):
        """Compute post-dominance using reversed flow (successors)."""
        self._compute_dominance_relation(region)

    def strictly_dominates(self, a: Block, b: Block) -> bool:
        """Return if `a` *strictly* dominates `b`."""
        if self._is_postdominance:
            raise ValueError(
                "Use strictly_postdominates() when initialized with compute_postdominance=True."
            )
        return a is not b and a in self._dominance[b]

    def dominates(self, a: Block, b: Block) -> bool:
        """Return if `a` dominates `b`."""
        if self._is_postdominance:
            raise ValueError(
                "Use postdominates() when initialized with compute_postdominance=True."
            )
        return a in self._dominance[b]

    def strictly_postdominates(self, a: Block, b: Block) -> bool:
        """Return if `a` *strictly* post-dominates `b`."""
        if not self._is_postdominance:
            raise ValueError(
                "Use strictly_dominates() when initialized with compute_postdominance=False."
            )
        return a is not b and a in self._dominance[b]

    def postdominates(self, a: Block, b: Block) -> bool:
        """Return if `a` post-dominates `b`."""
        if not self._is_postdominance:
            raise ValueError(
                "Use dominates() when initialized with compute_postdominance=False."
            )
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


def _strictly_postdominates_block(a: Block, b: Block) -> bool:
    """
    Returns true if block `a` strictly post-dominates block `b`(i.e., if it post-dominates `b` and
    is not `b`.), assuming they are in the same region.
    """
    if a is b:
        return False
    if a.parent is None:
        raise ValueError("Block `a` has no parent region")
    if a.parent is not b.parent:
        raise ValueError("Blocks `a` and `b` are not in the same region")

    return DominanceInfo(a.parent, compute_postdominance=True).strictly_postdominates(
        a, b
    )


# This function could be deemed useless for now, but it's intended to be
# overloaded with Values and Operations
def strictly_dominates(a: Block, b: Block) -> bool:
    """
    Returns true if block `a` strictly dominates block `b`, assuming they are in the
    same region.
    """
    return _strictly_dominates_block(a, b)


def strictly_postdominates(a: Block, b: Block) -> bool:
    """
    Returns true if block `a` strictly post-dominates block `b`, assuming they are in the
    same region.
    """
    return _strictly_postdominates_block(a, b)
