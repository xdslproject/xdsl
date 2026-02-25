"""
Tests for Location utility functions.
"""

from xdsl.dialects.builtin import (
    CallSiteLoc,
    FileLineColLoc,
    FusedLoc,
    NameLoc,
    OpaqueLoc,
    StringAttr,
    UnknownLoc,
)
from xdsl.ir.location import (
    collect_locations,
    find_location_of_type,
    fuse_locations,
    has_location_type,
    locations_equal,
    strip_locations,
    walk_locations,
)


def test_walk_unknown_loc():
    """Test walking UnknownLoc."""
    visited = []
    loc = UnknownLoc()
    walk_locations(loc, lambda x: visited.append(type(x).__name__))
    assert visited == ["UnknownLoc"]


def test_walk_file_line_col_loc():
    """Test walking FileLineColLoc."""
    visited = []
    loc = FileLineColLoc("test.cpp", 10, 8)
    walk_locations(loc, lambda x: visited.append(type(x).__name__))
    assert visited == ["FileLineColLoc"]


def test_walk_fused_loc():
    """Test walking FusedLoc."""
    visited = []
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    fused = FusedLoc.create([loc1, loc2])

    walk_locations(fused, lambda x: visited.append(type(x).__name__))

    assert visited[0] == "FusedLoc"
    assert "FileLineColLoc" in visited


def test_walk_name_loc():
    """Test walking NameLoc."""
    visited = []
    child = FileLineColLoc("test.cpp", 10, 8)
    named = NameLoc("CSE", child)

    walk_locations(named, lambda x: visited.append(type(x).__name__))

    assert visited == ["NameLoc", "FileLineColLoc"]


def test_walk_callsite_loc():
    """Test walking CallSiteLoc."""
    visited = []
    callee = UnknownLoc()
    caller = FileLineColLoc("main.cpp", 10, 8)
    callsite = CallSiteLoc(callee, caller)

    walk_locations(callsite, lambda x: visited.append(type(x).__name__))

    assert visited[0] == "CallSiteLoc"
    assert "UnknownLoc" in visited
    assert "FileLineColLoc" in visited


def test_walk_opaque_loc():
    """Test walking OpaqueLoc."""
    visited = []
    fallback = FileLineColLoc("test.cpp", 10, 8)
    opaque = OpaqueLoc.create(object(), fallback)

    walk_locations(opaque, lambda x: visited.append(type(x).__name__))

    assert visited == ["OpaqueLoc", "FileLineColLoc"]


def test_walk_nested_fused_loc():
    """Test walking deeply nested FusedLoc."""
    visited = []
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    loc3 = FileLineColLoc("c.cpp", 3, 3)

    inner_fused = FusedLoc.create([loc2, loc3])
    outer_fused = FusedLoc.create([loc1, inner_fused])

    walk_locations(outer_fused, lambda x: visited.append(type(x).__name__))

    assert visited.count("FileLineColLoc") == 3
    assert visited.count("FusedLoc") == 2


def test_find_location_of_type_simple():
    """Test finding location type in simple location."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    result = find_location_of_type(loc, FileLineColLoc)
    assert result == loc


def test_find_location_of_type_not_found():
    """Test finding location type that doesn't exist."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    result = find_location_of_type(loc, CallSiteLoc)
    assert result is None


def test_find_location_of_type_in_fused():
    """Test finding location type in FusedLoc."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    fused = FusedLoc.create([loc1, loc2])

    result = find_location_of_type(fused, FileLineColLoc)
    assert result == loc1


def test_find_location_of_type_in_name():
    """Test finding location type in NameLoc."""
    child = FileLineColLoc("test.cpp", 10, 8)
    named = NameLoc("CSE", child)

    result = find_location_of_type(named, FileLineColLoc)
    assert result == child


def test_find_location_of_type_in_callsite():
    """Test finding location type in CallSiteLoc."""
    callee = FileLineColLoc("callee.cpp", 1, 1)
    caller = FileLineColLoc("caller.cpp", 2, 2)
    callsite = CallSiteLoc(callee, caller)

    result = find_location_of_type(callsite, FileLineColLoc)
    assert result == callee


def test_fuse_locations_empty():
    """Test fusing empty list."""
    result = fuse_locations([])
    assert isinstance(result, UnknownLoc)


def test_fuse_locations_single():
    """Test fusing single location."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    result = fuse_locations([loc])
    assert result == loc


def test_fuse_locations_multiple():
    """Test fusing multiple locations."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)

    result = fuse_locations([loc1, loc2])
    assert isinstance(result, FusedLoc)
    assert len(result.locations) == 2


def test_fuse_locations_with_metadata():
    """Test fusing with metadata."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    metadata = StringAttr("CSE")

    result = fuse_locations([loc1, loc2], metadata)
    assert isinstance(result, FusedLoc)
    assert result.metadata == metadata


def test_fuse_locations_removes_unknown():
    """Test that fusing removes UnknownLoc."""
    unknown = UnknownLoc()
    loc = FileLineColLoc("test.cpp", 10, 8)

    result = fuse_locations([unknown, loc])
    assert result == loc


def test_fuse_locations_unwraps_nested():
    """Test that fusing unwraps nested FusedLoc with same metadata."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    loc3 = FileLineColLoc("c.cpp", 3, 3)

    metadata = StringAttr("CSE")
    nested = fuse_locations([loc2, loc3], metadata)

    result = fuse_locations([loc1, nested], metadata)
    assert isinstance(result, FusedLoc)
    assert len(result.locations) == 3


def test_strip_locations():
    """Test strip_locations function."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    result = strip_locations(loc)
    assert isinstance(result, UnknownLoc)

    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    fused = FusedLoc.create([loc1, loc2])

    result = strip_locations(fused)
    assert isinstance(result, UnknownLoc)


def test_locations_equal_same():
    """Test comparing same location."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    assert locations_equal(loc, loc)


def test_locations_equal_different_instances():
    """Test comparing different instances with same content."""
    loc1 = FileLineColLoc("test.cpp", 10, 8)
    loc2 = FileLineColLoc("test.cpp", 10, 8)
    assert locations_equal(loc1, loc2)


def test_locations_not_equal_different_content():
    """Test comparing locations with different content."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    assert not locations_equal(loc1, loc2)


def test_locations_not_equal_different_types():
    """Test comparing locations of different types."""
    loc1 = FileLineColLoc("test.cpp", 10, 8)
    loc2 = UnknownLoc()
    assert not locations_equal(loc1, loc2)


def test_collect_locations_simple():
    """Test collecting from simple location."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    result = collect_locations(loc)
    assert result == [loc]


def test_collect_locations_fused():
    """Test collecting from FusedLoc."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    fused = FusedLoc.create([loc1, loc2])

    result = collect_locations(fused)

    assert len(result) == 3
    assert fused in result


def test_collect_locations_nested():
    """Test collecting from deeply nested locations."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    loc3 = FileLineColLoc("c.cpp", 3, 3)

    inner_fused = FusedLoc.create([loc2, loc3])
    outer_fused = FusedLoc.create([loc1, inner_fused])

    result = collect_locations(outer_fused)

    assert len(result) == 5


def test_has_location_type_simple():
    """Test has_location_type with simple location."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    assert has_location_type(loc, FileLineColLoc)
    assert not has_location_type(loc, CallSiteLoc)


def test_has_location_type_nested():
    """Test has_location_type with nested location."""
    child = FileLineColLoc("test.cpp", 10, 8)
    named = NameLoc("CSE", child)

    assert has_location_type(named, FileLineColLoc)
    assert not has_location_type(named, CallSiteLoc)


def test_has_location_type_in_fused():
    """Test has_location_type with FusedLoc."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = CallSiteLoc(UnknownLoc(), FileLineColLoc("b.cpp", 2, 2))
    fused = FusedLoc.create([loc1, loc2])

    assert has_location_type(fused, FileLineColLoc)
    assert has_location_type(fused, CallSiteLoc)
    assert not has_location_type(fused, OpaqueLoc)
