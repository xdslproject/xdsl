"""
Collection of helpers for dealing with `Region`s, similar to `RegionUtils.cpp` in MLIR.
"""

from xdsl.ir import Region


def iter_used_values_defined_above(region: Region, limit: Region | None = None):
    """
    An iterator over the values used within `region` defined outside of `limit`.
    If `limit` is `None`, it is set to `region`.
    Values may be repeated.
    """
    if limit is not None:
        assert limit.is_ancestor(region), (
            "expected isolation limit to be an ancestor of the given region"
        )
    else:
        limit = region

    # Collect proper ancestors of `limit` upfront to avoid traversing the region
    # tree for every value.
    proper_ancestors = set[Region]()
    ancestor = limit.parent_region()
    while ancestor is not None:
        proper_ancestors.add(ancestor)
        ancestor = ancestor.parent_region()

    for op in region.walk():
        for operand in op.operands:
            if operand.owner.parent_region() in proper_ancestors:
                yield operand


def used_values_defined_above(region: Region, limit: Region | None = None):
    """
    The set of values used within `region` defined outside of `limit`.
    If `limit` is `None`, it is set to `region`.
    """
    return set(iter_used_values_defined_above(region, limit))
