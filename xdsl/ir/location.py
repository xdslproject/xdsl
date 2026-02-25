"""
Location utility functions.

This module provides utilities for working with location attributes,
including walking, finding, and fusing locations.
"""

from __future__ import annotations

from collections.abc import Callable

from typing_extensions import TypeVar

from xdsl.dialects.builtin import (
    CallSiteLoc,
    FusedLoc,
    LocationAttr,
    NameLoc,
    OpaqueLoc,
    UnknownLoc,
)

T = TypeVar("T", bound=LocationAttr)


def walk_locations(
    loc: LocationAttr,
    fn: Callable[[LocationAttr], None],
) -> None:
    """
    Recursively walk all locations nested under the given location.

    This is a pre-order traversal that visits the parent location before
    its children.

    Example:
        >>> def print_loc(loc): print(loc)
        >>> walk_locations(fused_loc, print_loc)

    Args:
        loc: The location to walk.
        fn: A function to call for each location.
    """
    fn(loc)

    # Recurse into child locations based on location type
    if isinstance(loc, FusedLoc):
        for child_loc in loc.locations.data:
            walk_locations(child_loc, fn)
    elif isinstance(loc, NameLoc):
        walk_locations(loc.child_loc, fn)
    elif isinstance(loc, CallSiteLoc):
        walk_locations(loc.callee, fn)
        walk_locations(loc.caller, fn)
    elif isinstance(loc, OpaqueLoc):
        walk_locations(loc.fallback_location, fn)
    # UnknownLoc and FileLineColLoc/FileLineColRange have no children


def find_location_of_type(
    loc: LocationAttr,
    loc_type: type[T],
) -> T | None:
    """
    Find the first nested location of the given type.

    Returns None if no location of the specified type is found.

    Example:
        >>> file_loc = find_location_of_type(fused_loc, FileLineColLoc)

    Args:
        loc: The location to search.
        loc_type: The type of location to find.

    Returns:
        The first location of the specified type, or None if not found.
    """
    result: T | None = None

    def finder(current: LocationAttr) -> None:
        nonlocal result
        if result is None and isinstance(current, loc_type):
            result = current

    walk_locations(loc, finder)
    return result


def fuse_locations(
    locations: list[LocationAttr],
    metadata: LocationAttr | None = None,
) -> LocationAttr:
    """
    Fuse multiple locations into a single FusedLoc.

    This function applies the same fusion logic as MLIR C++:
    1. Unwrap nested FusedLoc with same metadata
    2. Remove UnknownLoc
    3. Simplify if single location without metadata

    Args:
        locations: List of locations to fuse.
        metadata: Optional metadata attribute for the fusion.

    Returns:
        A fused location, or simplified result.
    """
    return FusedLoc.create(locations, metadata)


def strip_locations(loc: LocationAttr) -> UnknownLoc:
    """
    Replace all nested locations with UnknownLoc.

    This is useful for implementing the -strip-debuginfo pass.

    Args:
        loc: The location to strip.

    Returns:
        UnknownLoc.
    """
    return UnknownLoc()


def locations_equal(loc1: LocationAttr, loc2: LocationAttr) -> bool:
    """
    Check if two locations are structurally equal.

    This compares the full structure of nested locations.

    Args:
        loc1: First location to compare.
        loc2: Second location to compare.

    Returns:
        True if the locations are structurally equal.
    """
    # Simple implementation - compare string representations
    # For more precise comparison, we could compare the structure directly
    return str(loc1) == str(loc2)


def collect_locations(loc: LocationAttr) -> list[LocationAttr]:
    """
    Collect all nested locations into a list.

    Example:
        >>> locs = collect_locations(fused_loc)
        >>> print(len(locs))  # Number of nested locations

    Args:
        loc: The location to collect from.

    Returns:
        A list of all nested locations (including the root).
    """
    result: list[LocationAttr] = []

    def collector(current: LocationAttr) -> None:
        result.append(current)

    walk_locations(loc, collector)
    return result


def has_location_type(loc: LocationAttr, loc_type: type[T]) -> bool:
    """
    Check if a location contains a nested location of the given type.

    Example:
        >>> has_location_type(fused_loc, FileLineColLoc)
        True

    Args:
        loc: The location to check.
        loc_type: The type of location to look for.

    Returns:
        True if the location contains a nested location of the given type.
    """
    return find_location_of_type(loc, loc_type) is not None
