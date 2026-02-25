"""
Tests for Location infrastructure in IR.
"""

from io import StringIO

import pytest

from xdsl.context import Context
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    FileLineColLoc,
    IntegerType,
    UnknownLoc,
)
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer


def test_operation_default_location():
    """Test that operations default to UnknownLoc."""
    op = ConstantOp.from_int_and_width(42, 32)
    assert isinstance(op.get_loc(), UnknownLoc)


def test_operation_set_location():
    """Test setting operation location."""
    op = ConstantOp.from_int_and_width(42, 32)
    loc = FileLineColLoc("test.cpp", 10, 8)
    op.set_loc(loc)
    assert op.get_loc() == loc


def test_operation_create_with_location():
    """Test creating operation with location."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    op = ConstantOp.create(
        result_types=[IntegerType(32)],
        operands=[],
        location=loc,
    )
    assert op.get_loc() == loc


def test_operation_create_without_location():
    """Test creating operation without location defaults to UnknownLoc."""
    op = ConstantOp.create(
        result_types=[IntegerType(32)],
        operands=[],
    )
    assert isinstance(op.get_loc(), UnknownLoc)


def test_block_argument_default_location():
    """Test that block arguments default to UnknownLoc."""
    block = Block()
    arg = block.add_arg(IntegerType(32))
    assert isinstance(arg.get_loc(), UnknownLoc)


def test_block_argument_set_location():
    """Test setting block argument location."""
    block = Block()
    arg = block.add_arg(IntegerType(32))
    loc = FileLineColLoc("test.cpp", 10, 8)
    arg.set_loc(loc)
    assert arg.get_loc() == loc


def test_block_add_arg_with_location():
    """Test adding block argument with location."""
    block = Block()
    loc = FileLineColLoc("test.cpp", 10, 8)
    arg = block.add_arg(IntegerType(32), location=loc)
    assert arg.get_loc() == loc


def test_block_add_args_with_locations():
    """Test adding multiple block arguments with locations."""
    block = Block()
    loc1 = FileLineColLoc("a.cpp", 1, 1)
    loc2 = FileLineColLoc("b.cpp", 2, 2)

    args = block.add_args(
        [IntegerType(32), IntegerType(64)],
        [loc1, loc2],
    )

    assert len(args) == 2
    assert args[0].get_loc() == loc1
    assert args[1].get_loc() == loc2


def test_block_add_args_location_count_mismatch():
    """Test error when location count doesn't match argument count."""
    block = Block()
    loc1 = FileLineColLoc("a.cpp", 1, 1)

    with pytest.raises(ValueError, match="Number of locations"):
        block.add_args(
            [IntegerType(32), IntegerType(64)],
            [loc1],
        )


def test_block_insert_arg_with_location():
    """Test inserting block argument with location."""
    block = Block()
    loc = FileLineColLoc("test.cpp", 10, 8)
    arg = block.insert_arg(IntegerType(32), 0, location=loc)
    assert arg.get_loc() == loc


def test_region_location_from_parent():
    """Test that region location is inherited from parent operation."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    op = ConstantOp.create(
        result_types=[IntegerType(32)],
        operands=[],
        location=loc,
    )

    region = Region()
    op.add_region(region)

    assert region.get_loc() == loc


def test_region_location_orphan():
    """Test that orphan region returns UnknownLoc."""
    region = Region()
    assert isinstance(region.get_loc(), UnknownLoc)


def test_parse_func_with_arg_locations():
    """Test parsing function with argument locations."""
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)

    parser = Parser(
        ctx,
        """
        builtin.module {
            func.func @test(%arg0: i32 loc("arg.mlir":1:1), %arg1: i64 loc("arg.mlir":2:2)) {
                func.return
            }
        }
    """,
    )
    module = parser.parse_module()

    func_op = list(module.ops)[0]
    entry_block = func_op.body.blocks.first

    assert entry_block.args[0].get_loc() == FileLineColLoc("arg.mlir", 1, 1)
    assert entry_block.args[1].get_loc() == FileLineColLoc("arg.mlir", 2, 2)


def test_parse_func_with_op_location():
    """Test parsing function with operation location."""
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)

    parser = Parser(
        ctx,
        """
        builtin.module {
            func.func @test(%arg0: i32) loc("func.mlir":10:8) {
                func.return
            }
        }
    """,
    )
    module = parser.parse_module()

    func_op = list(module.ops)[0]
    assert func_op.get_loc() == FileLineColLoc("func.mlir", 10, 8)


def test_print_func_with_locations():
    """Test printing function with locations."""
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import Func

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Func)

    parser = Parser(
        ctx,
        """
        builtin.module {
            func.func @test(%arg0: i32 loc("arg.mlir":1:1)) loc("func.mlir":10:8) {
                func.return
            }
        }
    """,
    )
    module = parser.parse_module()

    # Test that operation location is printed
    stream = StringIO()
    printer = Printer(stream=stream, print_generic_format=False, print_debuginfo=True)
    printer.print_op(module)

    printed = stream.getvalue()
    assert 'loc("func.mlir":10:8)' in printed


def test_location_preserved_through_operations():
    """Test that location is preserved through operation manipulations."""
    loc = FileLineColLoc("test.cpp", 10, 8)
    op = ConstantOp.create(
        result_types=[IntegerType(32)],
        operands=[],
        location=loc,
    )

    # Change location
    new_loc = FileLineColLoc("new.cpp", 20, 16)
    op.set_loc(new_loc)

    assert op.get_loc() == new_loc

    # Change back
    op.set_loc(loc)
    assert op.get_loc() == loc
