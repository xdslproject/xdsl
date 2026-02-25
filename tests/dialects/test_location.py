"""
Tests for Location types.
"""

from io import StringIO

import pytest

from xdsl.context import Context
from xdsl.dialects.builtin import (
    CallSiteLoc,
    FileLineColLoc,
    FileLineColRange,
    FusedLoc,
    IntAttr,
    NameLoc,
    OpaqueLoc,
    StringAttr,
    UnknownLoc,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.utils.exceptions import ParseError


def test_unknown_loc():
    """Test UnknownLoc creation and printing."""
    loc = UnknownLoc()
    assert isinstance(loc, UnknownLoc)

    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(loc)
    assert stream.getvalue() == "loc(unknown)"


def test_file_line_col_loc():
    """Test FileLineColLoc creation and printing."""
    # Test creation with different argument types
    loc1 = FileLineColLoc("test.cpp", 10, 8)
    assert loc1.filename.data == "test.cpp"
    assert loc1.line.data == 10
    assert loc1.column.data == 8

    loc2 = FileLineColLoc(StringAttr("test.cpp"), IntAttr(10), IntAttr(8))
    assert loc2.filename.data == "test.cpp"
    assert loc2.line.data == 10
    assert loc2.column.data == 8

    # Test printing
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(loc1)
    assert stream.getvalue() == 'loc("test.cpp":10:8)'

    # Test equality
    assert loc1 == loc2
    assert loc1 != FileLineColLoc("test.cpp", 10, 9)


def test_file_line_col_range():
    """Test FileLineColRange creation and printing."""
    # Single point
    loc1 = FileLineColRange("test.cpp", 10, 8)
    assert loc1.filename.data == "test.cpp"
    assert loc1.start_line.data == 10
    assert loc1.start_column.data == 8
    assert loc1.end_line.data == 10
    assert loc1.end_column.data == 8

    # Range
    loc2 = FileLineColRange("test.cpp", 10, 8, 12, 18)
    assert loc2.end_line.data == 12
    assert loc2.end_column.data == 18

    # Test printing single point
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(loc1)
    assert stream.getvalue() == 'loc("test.cpp":10:8)'

    # Test printing range
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(loc2)
    assert stream.getvalue() == 'loc("test.cpp":10:8 to 12:18)'


def test_name_loc():
    """Test NameLoc creation and printing."""
    # Without child location
    loc1 = NameLoc("CSE")
    assert loc1.name_attr.data == "CSE"
    assert isinstance(loc1.child_loc, UnknownLoc)

    # With child location
    child = FileLineColLoc("test.cpp", 10, 8)
    loc2 = NameLoc("CSE", child)
    assert loc2.name_attr.data == "CSE"
    assert loc2.child_loc == child

    # Test printing without child
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(loc1)
    assert stream.getvalue() == 'loc("CSE")'

    # Test printing with child
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(loc2)
    assert stream.getvalue() == 'loc("CSE"(loc("test.cpp":10:8)))'


def test_callsite_loc():
    """Test CallSiteLoc creation and printing."""
    callee = UnknownLoc()
    caller = FileLineColLoc("main.cpp", 10, 8)
    loc = CallSiteLoc(callee, caller)

    assert loc.callee == callee
    assert loc.caller == caller

    # Test printing
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(loc)
    assert stream.getvalue() == 'loc(callsite(loc(unknown) at loc("main.cpp":10:8)))'


def test_fused_loc():
    """Test FusedLoc creation and printing."""
    from xdsl.dialects.builtin import NoneAttr

    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)

    # Without metadata
    fused1 = FusedLoc([loc1, loc2])
    assert len(fused1.locations) == 2
    assert isinstance(fused1.metadata, NoneAttr)

    # With metadata
    fused2 = FusedLoc([loc1, loc2], StringAttr("CSE"))
    assert len(fused2.locations) == 2
    assert fused2.metadata.data == "CSE"

    # Test printing without metadata
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(fused1)
    assert stream.getvalue() == 'loc(fused[loc("a.cpp":1:1), loc("b.cpp":2:2)])'

    # Test printing with metadata
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(fused2)
    assert stream.getvalue() == 'loc(fused<"CSE">[loc("a.cpp":1:1), loc("b.cpp":2:2)])'


def test_fused_loc_create_simplification():
    """Test FusedLoc.create() simplification logic."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)

    # Empty list should return UnknownLoc
    result1 = FusedLoc.create([])
    assert isinstance(result1, UnknownLoc)

    # Single location without metadata should be simplified
    result2 = FusedLoc.create([loc1])
    assert result2 == loc1

    # Multiple locations should remain FusedLoc
    result3 = FusedLoc.create([loc1, loc2])
    assert isinstance(result3, FusedLoc)
    assert len(result3.locations) == 2


def test_fused_loc_create_unwrap_nested():
    """Test FusedLoc.create() unwraps nested FusedLoc with same metadata."""
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)
    loc3 = FileLineColLoc("c.cpp", 3, 3)

    metadata = StringAttr("CSE")
    nested = FusedLoc.create([loc2, loc3], metadata)

    # Unwrap nested FusedLoc with same metadata
    result = FusedLoc.create([loc1, nested], metadata)
    assert isinstance(result, FusedLoc)
    assert len(result.locations) == 3


def test_fused_loc_create_remove_unknown():
    """Test FusedLoc.create() removes UnknownLoc."""
    unknown = UnknownLoc()
    loc1 = FileLineColLoc("a.cpp", 1, 1)

    # UnknownLoc should be removed
    result = FusedLoc.create([unknown, loc1])
    assert result == loc1


def test_opaque_loc():
    """Test OpaqueLoc creation and printing."""

    class MyData:
        value = 42

    data = MyData()
    fallback = FileLineColLoc("test.cpp", 10, 8)

    loc = OpaqueLoc.create(data, fallback)

    assert OpaqueLoc.get_underlying_location(loc) == data
    assert OpaqueLoc.get_underlying_location(loc).value == 42
    assert loc.fallback_location == fallback

    # Test with None fallback
    loc2 = OpaqueLoc.create(data)
    assert isinstance(loc2.fallback_location, UnknownLoc)

    # Test get_underlying_location with non-OpaqueLoc
    non_opaque = FileLineColLoc("test.cpp", 10, 8)
    assert OpaqueLoc.get_underlying_location(non_opaque) is None

    # Test printing (only prints fallback filename)
    stream = StringIO()
    printer = Printer(stream=stream)
    printer.print_location(loc)
    assert stream.getvalue() == '"test.cpp"'


def test_parse_location():
    """Test parsing all location types."""
    ctx = Context()

    # UnknownLoc
    attr = Parser(ctx, "loc(unknown)").parse_optional_location()
    assert attr == UnknownLoc()

    # FileLineColLoc
    attr = Parser(ctx, 'loc("one":2:3)').parse_optional_location()
    assert attr == FileLineColLoc(StringAttr("one"), IntAttr(2), IntAttr(3))

    # FileLineColRange
    attr = Parser(ctx, 'loc("test.cpp":10:8 to 12:18)').parse_optional_location()
    assert isinstance(attr, FileLineColRange)

    # NameLoc
    attr = Parser(ctx, 'loc("CSE"(loc("test.cpp":10:8)))').parse_optional_location()
    assert isinstance(attr, NameLoc)

    # CallSiteLoc
    attr = Parser(
        ctx, 'loc(callsite(loc(unknown) at loc("main.cpp":10:8)))'
    ).parse_optional_location()
    assert isinstance(attr, CallSiteLoc)

    # FusedLoc
    attr = Parser(
        ctx, 'loc(fused[loc("a.cpp":1:1), loc("b.cpp":2:2)])'
    ).parse_optional_location()
    assert isinstance(attr, FusedLoc)

    # FusedLoc with metadata
    attr = Parser(
        ctx, 'loc(fused<"CSE">[loc("a.cpp":1:1), loc("b.cpp":2:2)])'
    ).parse_optional_location()
    assert isinstance(attr, FusedLoc)

    # Invalid location syntax
    with pytest.raises(ParseError, match="Unexpected location syntax."):
        Parser(ctx, "loc(unexpected)").parse_optional_location()


def test_parse_location_with_metadata():
    """Test parsing location with various metadata types."""
    ctx = Context()

    # FusedLoc with string metadata
    attr = Parser(ctx, 'loc(fused<"CSE">[loc("a.cpp":1:1)])').parse_optional_location()
    assert isinstance(attr, FusedLoc)
    assert attr.metadata == StringAttr("CSE")
