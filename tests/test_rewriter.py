from collections.abc import Callable
from io import StringIO

import pytest

from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.arith import AddiOp, Arith, ConstantOp
from xdsl.dialects.builtin import Builtin, ModuleOp, f32, f64, i32, i64
from xdsl.ir import Block, Region
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.rewriter import BlockInsertPoint, InsertPoint, Rewriter


def rewrite_and_compare(
    prog: str, expected_prog: str, transformation: Callable[[ModuleOp, Rewriter], None]
):
    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(test.Test)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    rewriter = Rewriter()
    transformation(module, rewriter)

    file = StringIO()
    printer = Printer(stream=file, print_generic_format=True)
    printer.print_op(module)

    assert file.getvalue().strip() == expected_prog.strip()


def test_operation_deletion():
    """Test rewrites where SSA values are deleted."""

    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
^bb0:
}) : () -> ()"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops.first
        assert constant_op is not None
        rewriter.erase_op(constant_op)

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement
def test_replace_op_one_op():
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 43 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops.first
        assert constant_op is not None
        new_constant_op = ConstantOp.from_int_and_width(43, i32)
        rewriter.replace_op(constant_op, new_constant_op)

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement with multiple ops
def test_replace_op_multiple_op():
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %2 = "arith.addi"(%1, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops.first
        assert constant_op is not None
        new_constant = ConstantOp.from_int_and_width(1, i32)
        new_add = AddiOp(new_constant, new_constant)

        rewriter.replace_op(constant_op, [new_constant, new_add])

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement with manually specified results
def test_replace_op_new_results():
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %2 = "arith.muli"(%1, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 2 : i32}> : () -> i32
  %1 = "arith.muli"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        ops_iter = iter(module.ops)
        next(ops_iter)
        add_op = next(ops_iter)
        assert isinstance(add_op, AddiOp)

        rewriter.replace_op(add_op, [], [add_op.lhs])

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_at_end():
    """Test the inlining of a block at end."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  "test.op"() ({
    %1 = "test.op"() : () -> !test.type<"int">
  }) : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  "test.op"() ({
  }) : () -> ()
  %1 = "test.op"() : () -> !test.type<"int">
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        ops_iter = iter(module.ops)
        next(ops_iter)
        test_op = next(ops_iter)
        module_block = module.regions[0].blocks[0]
        test_block = test_op.regions[0].blocks[0]

        rewriter.inline_block(test_block, InsertPoint.at_end(module_block))

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_before():
    """Test the inlining of a block before an operation."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  "test.op"() ({
    %1 = "test.op"() : () -> !test.type<"int">
  }) : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() : () -> !test.type<"int">
  "test.op"() ({
  }) : () -> ()
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        ops_iter = iter(module.ops)
        next(ops_iter)
        test_op = next(ops_iter)
        test_block = test_op.regions[0].blocks[0]

        rewriter.inline_block(test_block, InsertPoint.before(test_op))

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_after():
    """Test the inlining of a block after an operation."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  "test.op"() ({
    %1 = "test.op"() : () -> !test.type<"int">
  }) : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() : () -> !test.type<"int">
  "test.op"() ({
  }) : () -> ()
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        ops_iter = iter(module.ops)
        constant_op = next(ops_iter)
        test_op = next(ops_iter)
        test_block = test_op.regions[0].blocks[0]

        rewriter.inline_block(test_block, InsertPoint.after(constant_op))

    rewrite_and_compare(prog, expected, transformation)


def test_insert_block():
    """Test the insertion of a block in a region."""
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = true}> : () -> i1
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
^bb0:
^bb1:
  %0 = "arith.constant"() <{value = true}> : () -> i1
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        module.regions[0].insert_block(Block(), 0)

    rewrite_and_compare(prog, expected, transformation)


def test_insert_block2():
    """Test the insertion of a block in a region."""
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = true}> : () -> i1
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = true}> : () -> i1
^bb0:
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        module.regions[0].insert_block(Block(), 1)

    rewrite_and_compare(prog, expected, transformation)


def test_insert_block_before():
    """Test the insertion of a block before another block."""
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = true}> : () -> i1
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
^bb0:
^bb1:
  %0 = "arith.constant"() <{value = true}> : () -> i1
}) : () -> ()
"""

    def insert_empty_block_before(module: ModuleOp, rewriter: Rewriter) -> None:
        rewriter.insert_block(
            Block(), BlockInsertPoint.before(module.regions[0].blocks[0])
        )

    rewrite_and_compare(prog, expected, insert_empty_block_before)


def test_insert_block_after():
    """Test the insertion of a block after another block."""
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = true}> : () -> i1
}) : () -> ()


"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = true}> : () -> i1
^bb0:
}) : () -> ()
"""

    def insert_empty_block_after(module: ModuleOp, rewriter: Rewriter) -> None:
        rewriter.insert_block(
            Block(), BlockInsertPoint.after(module.regions[0].blocks[0])
        )

    rewrite_and_compare(prog, expected, insert_empty_block_after)


def test_insert_op_before():
    """Test the insertion of an operation before another operation."""
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 43 : i32}> : () -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 34 : i64}> : () -> i64
  %1 = "arith.constant"() <{value = 43 : i32}> : () -> i32
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant = ConstantOp.from_int_and_width(34, i64)
        first_op = module.regions[0].blocks[0].first_op
        assert first_op is not None
        rewriter.insert_op(constant, InsertPoint.before(first_op))

    rewrite_and_compare(prog, expected, transformation)


def test_insert_op_after():
    """Test the insertion of an operation after another operation."""
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 43 : i32}> : () -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 43 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 34 : i64}> : () -> i64
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant = ConstantOp.from_int_and_width(34, i64)
        first_op = module.regions[0].blocks[0].first_op
        assert first_op is not None
        rewriter.insert_op(constant, InsertPoint.after(first_op))

    rewrite_and_compare(prog, expected, transformation)


def test_preserve_naming_single_op():
    """Test the preservation of names of SSAValues"""
    prog = """\
"builtin.module"() ({
  %i = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%i, %i) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %i = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %0 = "arith.addi"(%i, %i) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops.first
        assert constant_op is not None
        new_constant = ConstantOp.from_int_and_width(1, i32)

        rewriter.replace_op(constant_op, [new_constant])

    rewrite_and_compare(prog, expected, transformation)


def test_preserve_naming_multiple_ops():
    """Test the preservation of names of SSAValues for transformations to multiple ops"""
    prog = """\
"builtin.module"() ({
  %i = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%i, %i) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %i = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %i_1 = "arith.addi"(%i, %i) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %0 = "arith.addi"(%i_1, %i_1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops.first
        assert constant_op is not None
        new_constant = ConstantOp.from_int_and_width(1, i32)
        new_add = AddiOp(new_constant, new_constant)

        rewriter.replace_op(constant_op, [new_constant, new_add])

    rewrite_and_compare(prog, expected, transformation)


def test_no_result_rewriter():
    """Test rewriter on ops without results"""
    prog = """\
"builtin.module"() ({
  "test.termop"() : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  "test.op"() : () -> ()
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        return_op = module.ops.first
        assert return_op is not None
        new_op = test.TestOp.create()

        rewriter.replace_op(return_op, [new_op])

    rewrite_and_compare(prog, expected, transformation)


# Test erase operation
def test_erase_op():
    prog = """\
"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "arith.addi"(%1, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()
"""

    def transformation_safe(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops.first
        assert constant_op is not None
        rewriter.erase_op(constant_op, safe_erase=True)

    def transformation_unsafe(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops.first
        assert constant_op is not None
        rewriter.erase_op(constant_op, safe_erase=False)

    rewrite_and_compare(prog, expected, transformation_unsafe)

    with pytest.raises(
        Exception,
        match="Attempting to delete SSA value that still has uses",
    ):
        rewrite_and_compare(prog, expected, transformation_safe)


def test_erase_orphan_op():
    """Test that we can erase an orphan operation."""
    module = ModuleOp([])
    rewriter = Rewriter()
    rewriter.erase_op(module)


def test_inline_region_before():
    """Test the insertion of a block in a region."""
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  %1 = "test.op"() : () -> i64
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  %1 = "test.op"() : () -> f32
^bb1:
  %2 = "test.op"() : () -> f64
^bb2:
  %3 = "test.op"() : () -> i64
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        region = Region(
            (
                Block((test.TestOp(result_types=(f32,)),)),
                Block((test.TestOp(result_types=(f64,)),)),
            )
        )
        rewriter.inline_region(region, BlockInsertPoint.before(module.body.blocks[1]))

    rewrite_and_compare(prog, expected, transformation)


def test_inline_region_after():
    """Test the insertion of a block in a region."""
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  %1 = "test.op"() : () -> i64
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  %1 = "test.op"() : () -> f32
^bb1:
  %2 = "test.op"() : () -> f64
^bb2:
  %3 = "test.op"() : () -> i64
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        region = Region(
            (
                Block((test.TestOp(result_types=(f32,)),)),
                Block((test.TestOp(result_types=(f64,)),)),
            )
        )
        rewriter.inline_region(region, BlockInsertPoint.after(module.body.blocks[0]))

    rewrite_and_compare(prog, expected, transformation)


def test_inline_region_at_start():
    """Test the insertion of a block in a region."""
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  %1 = "test.op"() : () -> i64
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> f32
^bb0:
  %1 = "test.op"() : () -> f64
^bb1:
  %2 = "test.op"() : () -> i32
^bb2:
  %3 = "test.op"() : () -> i64
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        region = Region(
            (
                Block((test.TestOp(result_types=(f32,)),)),
                Block((test.TestOp(result_types=(f64,)),)),
            )
        )
        rewriter.inline_region(region, BlockInsertPoint.at_start(module.body))

    rewrite_and_compare(prog, expected, transformation)


def test_inline_region_at_end():
    """Test the insertion of a block in a region."""
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  %1 = "test.op"() : () -> i64
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  %1 = "test.op"() : () -> i64
^bb1:
  %2 = "test.op"() : () -> f32
^bb2:
  %3 = "test.op"() : () -> f64
}) : () -> ()
"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        region = Region(
            (
                Block((test.TestOp(result_types=(f32,)),)),
                Block((test.TestOp(result_types=(f64,)),)),
            )
        )
        rewriter.inline_region(region, BlockInsertPoint.at_end(module.body))

    rewrite_and_compare(prog, expected, transformation)


def test_verify_inline_region():
    region = Region(Block())

    with pytest.raises(ValueError, match="Cannot move region into itself."):
        Rewriter.inline_region(region, BlockInsertPoint.before(region.block))

    with pytest.raises(ValueError, match="Cannot move region into itself."):
        Rewriter.inline_region(region, BlockInsertPoint.after(region.block))

    with pytest.raises(ValueError, match="Cannot move region into itself."):
        Rewriter.inline_region(region, BlockInsertPoint.at_start(region))

    with pytest.raises(ValueError, match="Cannot move region into itself."):
        Rewriter.inline_region(region, BlockInsertPoint.at_end(region))
