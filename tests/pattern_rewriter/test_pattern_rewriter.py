import re
from collections.abc import Sequence
from io import StringIO

import pytest

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import test
from xdsl.dialects.arith import AddiOp, Arith, ConstantOp, MuliOp
from xdsl.dialects.builtin import (
    Builtin,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    UnitAttr,
    i32,
    i64,
)
from xdsl.ir import Block, Operation, SSAValue
from xdsl.irdl import BaseAttr
from xdsl.parser import Parser
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriterListener,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_constr_rewrite_pattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.printer import Printer
from xdsl.rewriter import BlockInsertPoint, InsertPoint


def rewrite_and_compare(
    prog: str,
    expected_prog: str,
    walker: PatternRewriteWalker,
    *,
    op_inserted: int = 0,
    op_removed: int = 0,
    op_modified: int = 0,
    op_replaced: int = 0,
    block_created: int = 0,
    expect_rewrite: bool = True,
):
    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(test.Test)

    num_op_inserted = 0
    num_op_removed = 0
    num_op_modified = 0
    num_op_replaced = 0
    num_block_created = 0

    def op_inserted_handler(op: Operation):
        nonlocal num_op_inserted
        num_op_inserted += 1

    def op_removed_handler(op: Operation):
        nonlocal num_op_removed
        num_op_removed += 1

    def op_modified_handler(op: Operation):
        nonlocal num_op_modified
        num_op_modified += 1

    def op_replaced_handler(op: Operation, values: Sequence[SSAValue | None]):
        nonlocal num_op_replaced
        num_op_replaced += 1

    def block_created_handler(block: Block):
        nonlocal num_block_created
        num_block_created += 1

    listener = PatternRewriterListener()
    listener.operation_insertion_handler = [op_inserted_handler]
    listener.operation_removal_handler = [op_removed_handler]
    listener.operation_modification_handler = [op_modified_handler]
    listener.operation_replacement_handler = [op_replaced_handler]
    listener.block_creation_handler = [block_created_handler]

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    walker.listener = listener
    did_rewrite = walker.rewrite_module(module)

    file = StringIO()
    printer = Printer(stream=file, print_generic_format=True)
    printer.print_op(module)

    assert file.getvalue().strip() == expected_prog.strip()

    assert num_op_inserted == op_inserted
    assert num_op_removed == op_removed
    assert num_op_modified == op_modified
    assert num_op_replaced == op_replaced
    assert num_block_created == block_created
    assert did_rewrite == expect_rewrite


def test_non_recursive_rewrite():
    """Test a simple non-recursive rewrite"""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 43 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    class RewriteConst(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(op, ConstantOp):
                new_constant = ConstantOp.from_int_and_width(43, i32)
                rewriter.replace_op(op, [new_constant])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(RewriteConst(), apply_recursively=False),
        op_inserted=1,
        op_removed=1,
        op_replaced=1,
        op_modified=2,
    )


def test_non_recursive_rewrite_reversed():
    """Test a simple non-recursive rewrite with reverse walk order."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 43 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    class RewriteConst(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(op, ConstantOp):
                new_constant = ConstantOp.from_int_and_width(43, i32)
                rewriter.replace_op(op, [new_constant])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(
            RewriteConst(), apply_recursively=False, walk_reverse=True
        ),
        op_inserted=1,
        op_removed=1,
        op_replaced=1,
        op_modified=2,
    )


def test_op_type_rewrite_pattern_method_decorator():
    """Test op_type_rewrite_pattern decorator on methods."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 43 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    class RewriteConst(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter):
            rewriter.replace_op(op, ConstantOp.from_int_and_width(43, i32))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(RewriteConst(), apply_recursively=False),
        op_inserted=1,
        op_removed=1,
        op_replaced=1,
        op_modified=2,
    )


def test_op_type_rewrite_pattern_union_type():
    """Test op_type_rewrite_pattern decorator on static functions."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %2 = "test"(%0, %1) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %2 = "test"(%0, %1) : (i32, i32) -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ConstantOp | AddiOp, rewriter: PatternRewriter):
            rewriter.replace_op(op, ConstantOp.from_int_and_width(42, i32))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=2,
        op_removed=2,
        op_replaced=2,
        op_modified=4,
    )


def test_recursive_rewriter():
    """Test recursive walks on operations created by rewrites."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %2 = "arith.addi"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %4 = "arith.addi"(%2, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %5 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %6 = "arith.addi"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %7 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %8 = "arith.addi"(%6, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter):
            if not isinstance(op_val := op.value, IntegerAttr):
                return
            val = op_val.value.data
            if val == 0 or val == 1:
                return
            constant_op = ConstantOp(IntegerAttr.from_int_and_width(val - 1, 32), i32)
            constant_one = ConstantOp(IntegerAttr.from_int_and_width(1, 32), i32)
            add_op = AddiOp(constant_op, constant_one)
            rewriter.replace_op(op, [constant_op, constant_one, add_op])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=True),
        op_inserted=12,
        op_removed=4,
        op_replaced=4,
        op_modified=3,
    )


def test_recursive_rewriter_reversed():
    """Test recursive walks on operations created by rewrites, in reverse walk order."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %2 = "arith.addi"(%0, %1) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %3 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %4 = "arith.addi"(%2, %3) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %5 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %6 = "arith.addi"(%4, %5) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  %7 = "arith.constant"() <{value = 1 : i32}> : () -> i32
  %8 = "arith.addi"(%6, %7) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter):
            if not isinstance(op_val := op.value, IntegerAttr):
                return
            val = op_val.value.data
            if val == 0 or val == 1:
                return
            constant_op = ConstantOp(IntegerAttr.from_int_and_width(val - 1, 32), i32)
            constant_one = ConstantOp(IntegerAttr.from_int_and_width(1, 32), i32)
            add_op = AddiOp(constant_op, constant_one)
            rewriter.replace_op(op, [constant_op, constant_one, add_op])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=True, walk_reverse=True),
        op_inserted=12,
        op_removed=4,
        op_replaced=4,
        op_modified=3,
    )


def test_greedy_rewrite_pattern_applier():
    """Test GreedyRewritePatternApplier."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 43 : i32}> : () -> i32
  %1 = "arith.muli"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    class ConstantRewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter):
            rewriter.replace_op(op, [ConstantOp.from_int_and_width(43, i32)])

    class AddiRewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: AddiOp, rewriter: PatternRewriter):
            rewriter.replace_op(op, [MuliOp(op.lhs, op.rhs)])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConstantRewrite(), AddiRewrite()]),
            apply_recursively=False,
        ),
        op_inserted=2,
        op_removed=2,
        op_replaced=2,
        op_modified=2,
    )


def test_insert_op_before_matched_op():
    """Test rewrites where operations are inserted before the matched operation."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 5 : i32}> : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, cst: ConstantOp, rewriter: PatternRewriter):
            new_cst = ConstantOp.from_int_and_width(42, i32)

            rewriter.insert_op(new_cst)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=1,
    )


def test_insert_op_at_start():
    """Test rewrites where operations are inserted with a given position."""

    prog = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    expected = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            new_cst = ConstantOp.from_int_and_width(42, i32)

            rewriter.insert_op(new_cst, InsertPoint.at_start(op.regions[0].blocks[0]))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=1,
    )


def test_insert_op_before():
    """Test rewrites where operations are inserted before a given operation."""

    prog = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    expected = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            new_cst = ConstantOp.from_int_and_width(42, i32)

            first_op = op.regions[0].block.ops.first
            assert first_op is not None
            rewriter.insert_op(new_cst, InsertPoint.before(first_op))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=1,
    )


def test_insert_op_after():
    """Test rewrites where operations are inserted after a given operation."""

    prog = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    expected = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
    %1 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            new_cst = ConstantOp.from_int_and_width(42, i32)

            first_op = op.regions[0].block.ops.first
            assert first_op is not None
            rewriter.insert_op(new_cst, InsertPoint.after(first_op))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=1,
    )


def test_insert_op_after_matched_op():
    """Test rewrites where operations are inserted after a given operation."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 42 : i32}> : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, cst: ConstantOp, rewriter: PatternRewriter):
            new_cst = ConstantOp.from_int_and_width(42, i32)

            rewriter.insert_op(new_cst, InsertPoint.after(cst))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=1,
    )


def test_insert_op_after_matched_op_reversed():
    """Test rewrites where operations are inserted after a given operation."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  %1 = "arith.constant"() <{value = 42 : i32}> : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, cst: ConstantOp, rewriter: PatternRewriter):
            new_cst = ConstantOp.from_int_and_width(42, i32)

            rewriter.insert_op(new_cst, InsertPoint.after(cst))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False, walk_reverse=True),
        op_inserted=1,
    )


def test_operation_deletion():
    """Test rewrites where SSA values are deleted."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
^bb0:
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter):
            rewriter.erase_op(op)

    rewrite_and_compare(prog, expected, PatternRewriteWalker(Rewrite()), op_removed=1)


def test_operation_deletion_reversed():
    """
    Test rewrites where SSA values are deleted.
    They have to be deleted in order for the rewrite to not fail.
    """

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
^bb0:
}) : () -> ()"""

    class EraseAll(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, ModuleOp):
                rewriter.erase_op(op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(EraseAll(), walk_reverse=True),
        op_removed=2,
    )


def test_operation_deletion_failure():
    """Test rewrites where SSA values are deleted with still uses."""

    ctx = Context()
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  %1 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ConstantOp, rewriter: PatternRewriter):
            rewriter.erase_op(op)

    parser = Parser(ctx, prog)
    module = parser.parse_module()
    walker = PatternRewriteWalker(Rewrite())

    # Check that the rewrite fails
    with pytest.raises(
        ValueError,
        match="Attempting to delete SSA value that still has uses of result of operation",
    ):
        walker.rewrite_module(module)


def test_delete_inner_op():
    """Test rewrites where an operation inside a region of the matched op is deleted."""

    prog = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    expected = """"builtin.module"() ({
  "test.op"() ({
  ^bb0:
  }) : () -> ()
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            first_op = op.regions[0].block.ops.first

            assert first_op is not None
            rewriter.erase_op(first_op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_removed=1,
    )


def test_replace_inner_op():
    """Test rewrites where an operation inside a region of the matched op is deleted."""

    prog = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 5 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    expected = """"builtin.module"() ({
  "test.op"() ({
    %0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  }) : () -> ()
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            first_op = op.regions[0].block.ops.first

            assert first_op is not None
            rewriter.replace_op(first_op, [ConstantOp.from_int_and_width(42, i32)])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=1,
        op_removed=1,
        op_replaced=1,
    )


def test_block_argument_type_change():
    """Test the modification of a block argument type."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() ({
  ^bb0(%1 : !test.type<"int">):
    %2 = "test.op"() : () -> !test.type<"int">
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() ({
  ^bb0(%1 : i64):
    %2 = "test.op"() : () -> !test.type<"int">
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                rewriter.replace_value_with_new_type(
                    matched_op.regs[0].blocks[0].args[0], i64
                )

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_modified=1,
    )


def test_block_argument_erasure():
    """Test the erasure of a block argument."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() ({
  ^bb0(%1 : !test.type<"int">):
    %2 = "test.op"() : () -> !test.type<"int">
  }, {
  ^bb0:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() ({
    %1 = "test.op"() : () -> !test.type<"int">
  }, {
  ^bb0:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                rewriter.erase_block_argument(matched_op.regs[0].blocks[0].args[0])

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=False)
    )


def test_block_argument_insertion():
    """Test the insertion of a block argument."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() ({
    %1 = "test.op"() : () -> !test.type<"int">
  }, {
  ^bb0:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() ({
  ^bb0(%1 : !test.type<"int">):
    %2 = "test.op"() : () -> !test.type<"int">
  }, {
  ^bb1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                rewriter.insert_block_argument(
                    matched_op.regs[0].blocks[0], 0, test.TestType("int")
                )

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_inline_block_before_matched_op():
    """Test the inlining of a block before the matched operation."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
  ^bb0:
    %2 = "test.op"() : () -> !test.type<"int">
  }, {
  ^bb1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() : () -> !test.type<"int">
  %2 = "test.op"() ({
  }, {
  ^bb0:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                rewriter.inline_block(
                    matched_op.regs[0].blocks[0], InsertPoint.before(matched_op)
                )

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_inline_block_before():
    """Test the inlining of a block before an operation."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %1 = "test.op"() ({
      %1 = "test.op"() : () -> !test.type<"int">
    }, {
    ^bb1:
    }) : () -> !test.type<"int">
  }, {
  ^bb2:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %2 = "test.op"() : () -> !test.type<"int">
    %3 = "test.op"() ({
    }, {
    ^bb0:
    }) : () -> !test.type<"int">
  }, {
  ^bb1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                first_op = matched_op.regs[0].blocks[0].first_op

                if isinstance(first_op, test.TestOp):
                    inner_block = first_op.regs[0].blocks[0]
                    rewriter.inline_block(inner_block, InsertPoint.before(first_op))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_inline_block_at_before_when_op_is_matched_op():
    """Test the inlining of a block before an operation, being the matched one."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
  ^bb0:
    %2 = "test.op"() : () -> !test.type<"int">
  }, {
  ^bb1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() : () -> !test.type<"int">
  %2 = "test.op"() ({
  }, {
  ^bb0:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                rewriter.inline_block(
                    matched_op.regs[0].blocks[0], InsertPoint.before(matched_op)
                )

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_inline_block_before_with_args():
    """Test the inlining of a block before an operation."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
  ^bb0(%arg0 : !test.type<"int">):
    %1 = "test.op"() ({
    ^bb1(%arg1 : !test.type<"int">):
      %1 = "test.op"(%arg1) : (!test.type<"int">) -> !test.type<"int">
    }, {
    ^bb2:
    }) : () -> !test.type<"int">
  }, {
  ^bb3:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
  ^bb0(%arg0 : !test.type<"int">):
    %2 = "test.op"(%arg0) : (!test.type<"int">) -> !test.type<"int">
    %3 = "test.op"() ({
    }, {
    ^bb1:
    }) : () -> !test.type<"int">
  }, {
  ^bb2:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                outer_block = matched_op.regs[0].blocks[0]
                first_op = outer_block.first_op

                if isinstance(first_op, test.TestOp):
                    inner_block = first_op.regs[0].blocks[0]
                    rewriter.inline_block(
                        inner_block, InsertPoint.before(first_op), outer_block.args
                    )

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_inline_block_after():
    """Test the inlining of a block after an operation."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %1 = "test.op"() ({
      %1 = "test.op"() : () -> !test.type<"int">
    }, {
      ^bb1:
    }) : () -> !test.type<"int">
  }, {
  ^bb2:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %2 = "test.op"() ({
    }, {
    ^bb0:
    }) : () -> !test.type<"int">
    %3 = "test.op"() : () -> !test.type<"int">
  }, {
  ^bb1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                first_op = matched_op.regs[0].blocks[0].first_op

                if first_op is not None and isinstance(first_op, test.TestOp):
                    if first_op.regs and first_op.regs[0].blocks:
                        inner_block = first_op.regs[0].blocks[0]
                        rewriter.inline_block(inner_block, InsertPoint.after(first_op))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_inline_block_after_matched():
    """Test the inlining of a block after an operation."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %1 = "test.op"() ({
      %1 = "test.op"() : () -> !test.type<"int">
    }, {
    ^bb1:
    }) : () -> !test.type<"int">
  }, {
  ^bb2:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %2 = "test.op"() ({
    }, {
    ^bb0:
    }) : () -> !test.type<"int">
  }, {
  ^bb1:
  }) : () -> !test.type<"int">
  %3 = "test.op"() : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                first_op = matched_op.regs[0].block.first_op

                if first_op is not None and isinstance(first_op, test.TestOp):
                    if first_op.regs and first_op.regs[0].blocks:
                        inner_block = first_op.regs[0].blocks[0]
                        rewriter.inline_block(
                            inner_block, InsertPoint.after(matched_op)
                        )

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_move_region_contents_to_new_regions():
    """Test moving a region outside of a region."""

    prog = """\
"builtin.module"() ({
  "test.op"() ({
    %0 = "test.op"() : () -> !test.type<"int">
    %1 = "test.op"() ({
    ^bb0:
      %2 = "test.op"() : () -> !test.type<"int">
    }) : () -> !test.type<"int">
  }) : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  "test.op"() ({
    %0 = "test.op"() : () -> !test.type<"int">
    %1 = "test.op"() ({
    }) : () -> !test.type<"int">
    %2 = "test.op"() ({
      %3 = "test.op"() : () -> !test.type<"int">
    }) : () -> !test.type<"int">
  }) : () -> ()
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            # Match the toplevel test.op
            if not isinstance(op.parent_op(), ModuleOp):
                return
            ops_iter = iter(op.regions[0].block.ops)

            _ = next(ops_iter)  # skip first op
            old_op = next(ops_iter)
            assert isinstance(old_op, test.TestOp)
            new_region = rewriter.move_region_contents_to_new_regions(old_op.regions[0])
            res_types = old_op.result_types
            new_op = test.TestOp.create(result_types=res_types, regions=[new_region])
            rewriter.insert_op(new_op, InsertPoint.after(old_op))

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=1,
    )


def test_insert_same_block():
    """Test rewriter on ops without results"""
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() {label = "a"} : () -> i32
  %1 = "test.op"() {label = "b"} : () -> i32
  %2 = "test.op"() {label = "c"} : () -> i32
  "func.return"() : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() {label = "alloc"} : () -> i32
  %1 = "test.op"() {label = "a"} : () -> i32
  "test.op"(%0) {label = "init"} : (i32) -> ()
  %2 = "test.op"() {label = "c"} : () -> i32
  "test.op"(%0) {label = "dealloc"} : (i32) -> ()
  "func.return"() : () -> ()
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if op.attributes["label"] != StringAttr("b"):
                return

            block = op.parent

            if block is None:
                return

            last_op = block.last_op
            assert last_op is not None

            alloc = test.TestOp.build(
                operands=((),),
                attributes={"label": StringAttr("alloc")},
                regions=((),),
                result_types=((i32,),),
            )
            init = test.TestOp.build(
                operands=(alloc.res,),
                attributes={"label": StringAttr("init")},
                regions=((),),
                result_types=((),),
            )
            dealloc = test.TestOp.build(
                operands=(alloc.res,),
                attributes={"label": StringAttr("dealloc")},
                regions=((),),
                result_types=((),),
            )

            # Allocate before first use
            rewriter.insert_op(alloc, InsertPoint.at_start(block))
            # Deallocate after last use
            rewriter.insert_op(dealloc, InsertPoint.before(last_op))
            # Init instead of creating, and replace result with allocated value
            rewriter.replace_op(op, init, alloc.res)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=3,
        op_removed=1,
        op_replaced=1,
    )


def test_inline_region_before():
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  "test.op"() ({
    ^bb1:
    %1 = "test.op"() : () -> f32
    ^bb2:
    %2 = "test.op"() : () -> f64
  }) {label = "а"} : () -> ()
  %2 = "test.op"() : () -> i64
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

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if op.attributes.get("label") != StringAttr("а"):
                return
            if op.parent is None:
                return

            rewriter.inline_region(op.regions[0], BlockInsertPoint.before(op.parent))
            rewriter.erase_op(op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=0,
        op_removed=1,
    )


def test_inline_region_after():
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
  "test.op"() ({
    ^bb1:
    %1 = "test.op"() : () -> f32
    ^bb2:
    %2 = "test.op"() : () -> f64
  }) {label = "а"} : () -> ()
^bb0:
  %2 = "test.op"() : () -> i64
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

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if op.attributes.get("label") != StringAttr("а"):
                return
            if op.parent is None:
                return

            rewriter.inline_region(op.regions[0], BlockInsertPoint.after(op.parent))
            rewriter.erase_op(op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=0,
        op_removed=1,
    )


def test_inline_region_at_start():
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  "test.op"() ({
    ^bb1:
    %1 = "test.op"() : () -> f32
    ^bb2:
    %2 = "test.op"() : () -> f64
  }) {label = "а"} : () -> ()
  %2 = "test.op"() : () -> i64
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

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if op.attributes.get("label") != StringAttr("а"):
                return
            parent_region = op.parent_region()
            if parent_region is None:
                return

            rewriter.inline_region(
                op.regions[0], BlockInsertPoint.at_start(parent_region)
            )
            rewriter.erase_op(op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=0,
        op_removed=1,
    )


def test_inline_region_at_end():
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> i32
^bb0:
  "test.op"() ({
    ^bb1:
    %1 = "test.op"() : () -> f32
    ^bb2:
    %2 = "test.op"() : () -> f64
  }) {label = "а"} : () -> ()
  %2 = "test.op"() : () -> i64
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

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if op.attributes.get("label") != StringAttr("а"):
                return
            parent_region = op.parent_region()
            if parent_region is None:
                return

            rewriter.inline_region(
                op.regions[0], BlockInsertPoint.at_end(parent_region)
            )
            rewriter.erase_op(op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=0,
        op_removed=1,
    )


def test_erased_ssavalue():
    prog = """\
builtin.module {
  "test.op"() ({
    %0 = "test.op"() : () -> i32
    "test.op"(%0) : (i32) -> ()
  }) : () -> ()
}
  """

    expected = """\
"builtin.module"() ({
  "test.op"() ({
  ^bb0:
  }) : () -> ()
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if op.results or op.operands:
                rewriter.erase_op(op, safe_erase=False)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=True),
        op_removed=2,
    )


def test_pattern_rewriter_as_op_builder():
    """Test that the PatternRewriter works as an OpBuilder."""
    prog = """
"builtin.module"() ({
  "test.op"() : () -> ()
  "test.op"() {nomatch} : () -> ()
  "test.op"() : () -> ()
}) : () -> ()"""
    expected = """
"builtin.module"() ({
  "test.op"() {inserted} : () -> ()
  "test.op"() {replaced} : () -> ()
  "test.op"() {nomatch} : () -> ()
  "test.op"() {inserted} : () -> ()
  "test.op"() {replaced} : () -> ()
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if "nomatch" in op.attributes:
                return
            with ImplicitBuilder(rewriter):
                test.TestOp.create(attributes={"inserted": UnitAttr()})
            rewriter.replace_op(
                op, test.TestOp.create(attributes={"replaced": UnitAttr()})
            )

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=4,
        op_removed=2,
        op_replaced=2,
    )


def test_type_conversion():
    """Test rewriter on ops without results"""
    prog = """\
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg : i32):
  }) : () -> ()
  %0 = "test.op"() {nested = memref<*xi32>} : () -> i32
  %1 = "test.op"() {type = () -> memref<*xi32>} : () -> f32
  %2 = "test.op"() <{prop = memref<*xi32>}> : () -> f32
  %3 = "test.op"(%0, %1) : (i32, f32) -> memref<*xi32>
  %4 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32
  "func.return"() : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg : index):
  }) : () -> ()
  %0 = "test.op"() {nested = memref<*xindex>} : () -> index
  %1 = "test.op"() {type = () -> memref<*xindex>} : () -> f32
  %2 = "test.op"() <{prop = memref<*xindex>}> : () -> f32
  %3 = "test.op"(%0, %1) : (index, f32) -> memref<*xindex>
  %4 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  "func.return"() : () -> ()
}) : () -> ()
"""

    class Rewrite(TypeConversionPattern):
        @attr_type_rewrite_pattern
        def convert_type(self, typ: IntegerType) -> IndexType:
            return IndexType()

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(recursive=True), apply_recursively=False),
        op_inserted=5,
        op_removed=5,
        op_replaced=5,
        op_modified=5,
    )
    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(recursive=True), apply_recursively=True),
        op_inserted=5,
        op_removed=5,
        op_replaced=5,
        op_modified=5,
    )
    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(
            Rewrite(recursive=True), apply_recursively=False, walk_reverse=True
        ),
        op_inserted=5,
        op_removed=5,
        op_replaced=5,
        op_modified=5,
    )

    non_rec_expected = """\
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg : index):
  }) : () -> ()
  %0 = "test.op"() {nested = memref<*xi32>} : () -> index
  %1 = "test.op"() {type = () -> memref<*xi32>} : () -> f32
  %2 = "test.op"() <{prop = memref<*xi32>}> : () -> f32
  %3 = "test.op"(%0, %1) : (index, f32) -> memref<*xi32>
  %4 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index
  "func.return"() : () -> ()
}) : () -> ()
"""

    rewrite_and_compare(
        prog,
        non_rec_expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_inserted=2,
        op_removed=2,
        op_replaced=2,
        op_modified=4,
    )
    rewrite_and_compare(
        prog,
        non_rec_expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=True),
        op_inserted=2,
        op_removed=2,
        op_replaced=2,
        op_modified=4,
    )
    rewrite_and_compare(
        prog,
        non_rec_expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False, walk_reverse=True),
        op_inserted=2,
        op_removed=2,
        op_replaced=2,
        op_modified=4,
    )

    expected = """\
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg : i32):
  }) : () -> ()
  %0 = "test.op"() {nested = memref<*xindex>} : () -> index
  %1 = "test.op"() {type = () -> memref<*xindex>} : () -> f32
  %2 = "test.op"() <{prop = memref<*xindex>}> : () -> f32
  %3 = "test.op"(%0, %1) : (index, f32) -> memref<*xindex>
  %4 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> i32
  "func.return"() : () -> ()
}) : () -> ()
"""

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(
            Rewrite(ops=(test.TestOp,), recursive=True), apply_recursively=False
        ),
        op_inserted=4,
        op_removed=4,
        op_replaced=4,
        op_modified=4,
    )
    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(
            Rewrite(ops=(test.TestOp,), recursive=True), apply_recursively=True
        ),
        op_inserted=4,
        op_removed=4,
        op_replaced=4,
        op_modified=4,
    )
    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(
            Rewrite(ops=(test.TestOp,), recursive=True),
            apply_recursively=False,
            walk_reverse=True,
        ),
        op_inserted=4,
        op_removed=4,
        op_replaced=4,
        op_modified=4,
    )

    class RewriteMaybe(TypeConversionPattern):
        @attr_type_rewrite_pattern
        def convert_type(self, typ: IntegerType) -> IndexType | None:
            return IndexType() if typ.width.data >= 8 else None

    prog = """\
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg : i6):
  }) : () -> ()
  %0 = "test.op"() {nested = memref<*xi4>} : () -> i6
  %1 = "test.op"() {type = () -> memref<*xi32>} : () -> f32
  %2 = "test.op"() <{prop = memref<*xi4>}> : () -> i32
  %3 = "test.op"(%0, %1) : (i6, f32) -> memref<*xi32>
  %4 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i6, i6) -> i6
  "func.return"() : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  "func.func"() ({
  ^bb0(%arg : i6):
  }) : () -> ()
  %0 = "test.op"() {nested = memref<*xi4>} : () -> i6
  %1 = "test.op"() {type = () -> memref<*xindex>} : () -> f32
  %2 = "test.op"() <{prop = memref<*xi4>}> : () -> index
  %3 = "test.op"(%0, %1) : (i6, f32) -> memref<*xindex>
  %4 = "arith.addi"(%0, %0) <{overflowFlags = #arith.overflow<none>}> : (i6, i6) -> i6
  "func.return"() : () -> ()
}) : () -> ()
"""

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(RewriteMaybe(recursive=True), apply_recursively=True),
        op_inserted=3,
        op_removed=3,
        op_replaced=3,
        op_modified=1,
    )

    prog = """\
"builtin.module"() ({
  "test.op"() {dict_nest = {hello = i32}} : () -> ()
}) : () -> ()
"""

    expected_recursive = """
"builtin.module"() ({
  "test.op"() {dict_nest = {hello = index}} : () -> ()
}) : () -> ()
"""

    rewrite_and_compare(
        prog,
        prog,
        PatternRewriteWalker(Rewrite(recursive=False)),
        expect_rewrite=False,
    )

    rewrite_and_compare(
        prog,
        expected_recursive,
        PatternRewriteWalker(Rewrite(recursive=True)),
        op_inserted=1,
        op_removed=1,
        op_replaced=1,
    )


def test_recursive_type_conversion_in_regions():
    prog = """\
"builtin.module"() ({
  "func.func"() <{function_type = (memref<2x4xui16>) -> (), sym_name = "main", sym_visibility = "private"}> ({
  ^bb0(%arg0 : memref<2x4xui16>):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""
    expected_prog = """\
"builtin.module"() ({
  "func.func"() <{function_type = (memref<2x4xindex>) -> (), sym_name = "main", sym_visibility = "private"}> ({
  ^bb0(%arg0 : memref<2x4xindex>):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""

    class IndexConversion(TypeConversionPattern):
        @attr_type_rewrite_pattern
        def convert_type(self, typ: IntegerType) -> IndexType:
            return IndexType()

    rewrite_and_compare(
        prog,
        expected_prog,
        PatternRewriteWalker(IndexConversion(recursive=True)),
        op_inserted=1,
        op_removed=1,
        op_replaced=1,
        op_modified=1,
    )


def test_no_change():
    """Test that doing nothing successfully does not report doing something."""

    prog = """\
"builtin.module"() ({
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            return

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        expect_rewrite=False,
    )


def test_error():
    prog = """\
builtin.module {
  "test.op"() {erroneous = false} : () -> ()
  "test.op"() : () -> ()
  "test.op"() {erroneous = true} : () -> ()
  "test.op"() : () -> ()
}
"""
    expected = """\
"builtin.module"() ({
  "test.op"() {erroneous = false} : () -> ()
  "test.op"() : () -> ()
  "test.op"() {erroneous = true} : () -> ()
  ^^^^^^^^^--------------------------------------------------------------
  | Error while applying pattern: Expected operation to not be erroneous!
  -----------------------------------------------------------------------
  "test.op"() : () -> ()
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.attributes.get(
                "erroneous", IntegerAttr.from_int_and_width(0, 1)
            ) == IntegerAttr.from_int_and_width(1, 1):
                raise ValueError("Expected operation to not be erroneous!")
            return

    ctx = Context(allow_unregistered=True)
    ctx.load_dialect(Builtin)
    ctx.load_dialect(Arith)
    ctx.load_dialect(test.Test)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    walker = PatternRewriteWalker(Rewrite())
    with pytest.raises(ValueError, match=re.escape(expected)):
        walker.rewrite_module(module)


def test_attr_constr_rewrite_pattern():
    prog = """\
"builtin.module"() ({
  "func.func"() <{function_type = (memref<2x4xui16>) -> (), sym_name = "main", sym_visibility = "private"}> ({
  ^bb0(%arg0 : memref<2x4xui16>):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""
    expected_prog = """\
"builtin.module"() ({
  "func.func"() <{function_type = (memref<2x4xindex>) -> (), sym_name = "main", sym_visibility = "private"}> ({
  ^bb0(%arg0 : memref<2x4xindex>):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""

    class IndexConversion(TypeConversionPattern):
        @attr_constr_rewrite_pattern(BaseAttr(IntegerType))
        def convert_type(self, typ: IntegerType) -> IndexType:
            return IndexType()

    rewrite_and_compare(
        prog,
        expected_prog,
        PatternRewriteWalker(IndexConversion(recursive=True)),
        op_inserted=1,
        op_removed=1,
        op_replaced=1,
        op_modified=1,
    )


def test_pattern_rewriter_erase_op_with_region():
    """Test that erasing an operation with a region works correctly."""
    prog = """
"builtin.module"() ({
  "test.op"() ({
    "test.op"() {error_if_matching} : () -> ()
  }): () -> ()
}) : () -> ()"""
    expected = """
"builtin.module"() ({
^bb0:
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if "error_if_matching" in op.attributes:
                raise Exception("operation that is supposed to be deleted was matched")
            assert not op.attributes
            rewriter.erase_op(op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
        op_removed=1,
    )


def test_pattern_rewriter_notify_op_modified():
    """Test that notifying on op modifications works correctly."""
    prog = """
"builtin.module"() ({
  "test.op"() : () -> ()
  "test.op"() : () -> ()
  "test.op"() : () -> ()
}) : () -> ()"""
    expected = """
"builtin.module"() ({
  "test.op"() {modified} : () -> ()
  "test.op"() {modified} : () -> ()
  "test.op"() {modified} : () -> ()
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: test.TestOp, rewriter: PatternRewriter):
            if "modified" in op.attributes:
                return
            op.attributes["modified"] = UnitAttr()
            rewriter.notify_op_modified(op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=True),
        op_modified=3,
    )
