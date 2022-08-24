from xdsl.dialects.scf import Scf, If

from xdsl.printer import Printer
from xdsl.dialects.builtin import Builtin, IntegerAttr, i32, i64, ModuleOp
from xdsl.parser import Parser
from xdsl.dialects.arith import Arith, Constant, Addi, Muli
from xdsl.ir import MLContext, Region, Operation
from xdsl.pattern_rewriter import (PatternRewriteWalker,
                                   op_type_rewrite_pattern, RewritePattern,
                                   PatternRewriter, AnonymousRewritePattern,
                                   GreedyRewritePatternApplier)

from io import StringIO


def rewrite_and_compare(prog: str, expected_prog: str,
                        walker: PatternRewriteWalker):
    ctx = MLContext()
    builtin = Builtin(ctx)
    arith = Arith(ctx)
    scf = Scf(ctx)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    walker.rewrite_module(module)
    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected_prog.strip()


def test_non_recursive_rewrite():
    """Test a simple non-recursive rewrite"""

    prog = \
"""module() {
%0 : !i32 = arith.constant() ["value" = 42 : !i32]
%1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    class RewriteConst(RewritePattern):

        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(op, Constant):
                new_constant = Constant.from_int_constant(43, i32)
                rewriter.replace_matched_op([new_constant])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(RewriteConst(), apply_recursively=False))


def test_non_recursive_rewrite_reversed():
    """Test a simple non-recursive rewrite with reverse walk order."""

    prog = \
    """module() {
%0 : !i32 = arith.constant() ["value" = 42 : !i32]
%1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    class RewriteConst(RewritePattern):

        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(op, Constant):
                new_constant = Constant.from_int_constant(43, i32)
                rewriter.replace_matched_op([new_constant])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(RewriteConst(),
                             apply_recursively=False,
                             walk_reverse=True))


def test_op_type_rewrite_pattern_method_decorator():
    """Test op_type_rewrite_pattern decorator on methods."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    class RewriteConst(RewritePattern):

        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: Constant, rewriter: PatternRewriter):
            rewriter.replace_matched_op(Constant.from_int_constant(43, i32))

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(RewriteConst(), apply_recursively=False))


def test_op_type_rewrite_pattern_static_decorator():
    """Test op_type_rewrite_pattern decorator on static functions."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: Constant, rewriter: PatternRewriter):
        rewriter.replace_matched_op(Constant.from_int_constant(43, i32))

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_recursive_rewriter():
    """Test recursive walks on operations created by rewrites."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i64]
  %1 : !i32 = arith.constant() ["value" = 1 : !i64]
  %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
  %3 : !i32 = arith.constant() ["value" = 1 : !i64]
  %4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i64]
  %6 : !i32 = arith.addi(%4 : !i32, %5 : !i32)
  %7 : !i32 = arith.constant() ["value" = 1 : !i64]
  %8 : !i32 = arith.addi(%6 : !i32, %7 : !i32)
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: Constant, rewriter: PatternRewriter):
        val = op.value.value.data
        if val == 0 or val == 1:
            return None
        constant_op = Constant.from_attr(
            IntegerAttr.from_int_and_width(val - 1, 64), i32)
        constant_one = Constant.from_attr(
            IntegerAttr.from_int_and_width(1, 64), i32)
        add_op = Addi.get(constant_op, constant_one)
        rewriter.replace_matched_op([constant_op, constant_one, add_op])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=True))


def test_recursive_rewriter_reversed():
    """Test recursive walks on operations created by rewrites, in reverse walk order."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i64]
  %1 : !i32 = arith.constant() ["value" = 1 : !i64]
  %2 : !i32 = arith.addi(%0 : !i32, %1 : !i32)
  %3 : !i32 = arith.constant() ["value" = 1 : !i64]
  %4 : !i32 = arith.addi(%2 : !i32, %3 : !i32)
  %5 : !i32 = arith.constant() ["value" = 1 : !i64]
  %6 : !i32 = arith.addi(%4 : !i32, %5 : !i32)
  %7 : !i32 = arith.constant() ["value" = 1 : !i64]
  %8 : !i32 = arith.addi(%6 : !i32, %7 : !i32)
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: Constant, rewriter: PatternRewriter):
        val = op.value.value.data
        if val == 0 or val == 1:
            return None
        constant_op = Constant.from_attr(
            IntegerAttr.from_int_and_width(val - 1, 64), i32)
        constant_one = Constant.from_attr(
            IntegerAttr.from_int_and_width(1, 64), i32)
        add_op = Addi.get(constant_op, constant_one)
        rewriter.replace_matched_op([constant_op, constant_one, add_op])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=True,
                             walk_reverse=True))


def test_greedy_rewrite_pattern_applier():
    """Test GreedyRewritePatternApplier."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
  %1 : !i32 = arith.muli(%0 : !i32, %0 : !i32)
}"""

    @op_type_rewrite_pattern
    def constant_rewrite(op: Constant, rewriter: PatternRewriter):
        rewriter.replace_matched_op([Constant.from_int_constant(43, i32)])

    @op_type_rewrite_pattern
    def addi_rewrite(op: Addi, rewriter: PatternRewriter):
        rewriter.replace_matched_op([Muli.get(op.input1, op.input2)])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(GreedyRewritePatternApplier([
            AnonymousRewritePattern(constant_rewrite),
            AnonymousRewritePattern(addi_rewrite)
        ]),
                             apply_recursively=False))


def test_insert_op_before_matched_op():
    """Test rewrites where operations are inserted before the matched operation."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(cst: Constant, rewriter: PatternRewriter):
        new_cst = Constant.from_int_constant(42, i32)
        rewriter.insert_op_before_matched_op(new_cst)

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_insert_op_at_pos():
    """Test rewrites where operations are inserted with a given position."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(mod: ModuleOp, rewriter: PatternRewriter):
        new_cst = Constant.from_int_constant(42, i32)
        rewriter.insert_op_at_pos(new_cst, mod.regions[0].blocks[0], 0)

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_insert_op_before():
    """Test rewrites where operations are inserted before a given operation."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
  %1 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(mod: ModuleOp, rewriter: PatternRewriter):
        new_cst = Constant.from_int_constant(42, i32)
        rewriter.insert_op_before(new_cst, mod.ops[0])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_insert_op_after():
    """Test rewrites where operations are inserted after a given operation."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
  %1 : !i32 = arith.constant() ["value" = 42 : !i32]
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(mod: ModuleOp, rewriter: PatternRewriter):
        new_cst = Constant.from_int_constant(42, i32)
        rewriter.insert_op_after(new_cst, mod.ops[0])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_insert_op_after_matched_op():
    """Test rewrites where operations are inserted after a given operation."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
  %1 : !i32 = arith.constant() ["value" = 42 : !i32]
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(cst: Constant, rewriter: PatternRewriter):
        new_cst = Constant.from_int_constant(42, i32)
        rewriter.insert_op_after_matched_op(new_cst)

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_insert_op_after_matched_op_reversed():
    """Test rewrites where operations are inserted after a given operation."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
  %1 : !i32 = arith.constant() ["value" = 42 : !i32]
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(cst: Constant, rewriter: PatternRewriter):
        new_cst = Constant.from_int_constant(42, i32)
        rewriter.insert_op_after_matched_op(new_cst)

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False,
                             walk_reverse=True))


def test_operation_deletion():
    """Test rewrites where SSA values are deleted."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: Constant, rewriter: PatternRewriter):
        rewriter.erase_matched_op()

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite)))


def test_operation_deletion_reversed():
    """
    Test rewrites where SSA values are deleted.
    They have to be deleted in order for the rewrite to not fail.
    """

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {}"""

    def match_and_rewrite(op: Operation, rewriter: PatternRewriter):
        if not isinstance(op, ModuleOp):
            rewriter.erase_matched_op()

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             walk_reverse=True))


def test_operation_deletion_failure():
    """Test rewrites where SSA values are deleted with still uses."""

    ctx = MLContext()
    builtin = Builtin(ctx)
    arith = Arith(ctx)

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: Constant, rewriter: PatternRewriter):
        rewriter.erase_matched_op()

    parser = Parser(ctx, prog)
    module = parser.parse_op()
    walker = PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite))

    # Check that the rewrite fails
    try:
        walker.rewrite_module(module)
        assert False
    except Exception as e:
        pass


def test_delete_inner_op():
    """Test rewrites where an operation inside a region of the matched op is deleted."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""


    expected = \
"""module() {}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: ModuleOp, rewriter: PatternRewriter):
        rewriter.erase_op(op.ops[0])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite)))


def test_replace_inner_op():
    """Test rewrites where an operation inside a region of the matched op is deleted."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""


    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 42 : !i32]
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: ModuleOp, rewriter: PatternRewriter):
        rewriter.replace_op(op.ops[0], [Constant.from_int_constant(42, i32)])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite)))


def test_block_argument_type_change():
    """Test the modification of a block argument type."""

    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
^0(%1 : !i32):
  %2 : !i32 = arith.addi(%1 : !i32, %1 : !i32)
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {
  ^0(%1 : !i64):
    %2 : !i32 = arith.addi(%1 : !i64, %1 : !i64)
  }
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: If, rewriter: PatternRewriter):
        rewriter.modify_block_argument_type(op.true_region.blocks[0].args[0],
                                            i64)

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_block_argument_erasure():
    """Test the erasure of a block argument."""

    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
^0(%1 : !i32):
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {}
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: If, rewriter: PatternRewriter):
        rewriter.erase_block_argument(op.true_region.blocks[0].args[0])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_block_argument_insertion():
    """Test the insertion of a block argument."""

    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {
  ^0(%1 : !i32):
  }
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: If, rewriter: PatternRewriter):
        rewriter.insert_block_argument(op.true_region.blocks[0], 0, i32)

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_inline_block_at_pos():
    """Test the inlining of a block at a certain position."""

    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
  scf.if(%0 : !i1) {
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  }
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
    scf.if(%0 : !i1) {}
  }
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: If, rewriter: PatternRewriter):
        if len(op.true_region.blocks[0].ops) > 0 and isinstance(
                op.true_region.blocks[0].ops[0], If):
            inner_if_block = op.true_region.blocks[0].ops[
                0].true_region.blocks[0]
            rewriter.inline_block_at_pos(inner_if_block,
                                         op.true_region.blocks[0], 0)

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite)))


def test_inline_block_before_matched_op():
    """Test the inlining of a block before the matched operation."""

    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  scf.if(%0 : !i1) {}
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: If, rewriter: PatternRewriter):
        rewriter.inline_block_before_matched_op(op.true_region.blocks[0])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_inline_block_before():
    """Test the inlining of a block before an operation."""

    prog = \
"""module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
  scf.if(%0 : !i1) {
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  }
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
    scf.if(%0 : !i1) {}
  }
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: If, rewriter: PatternRewriter):
        if len(op.true_region.blocks[0].ops) > 0 and isinstance(
                op.true_region.blocks[0].ops[0], If):
            inner_if_block = op.true_region.blocks[0].ops[
                0].true_region.blocks[0]
            rewriter.inline_block_before(inner_if_block,
                                         op.true_region.blocks[0].ops[0])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_inline_block_at_before_when_op_is_matched_op():
    """Test the inlining of a block before an operation, being the matched one."""

    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  scf.if(%0 : !i1) {}
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: If, rewriter: PatternRewriter):
        rewriter.inline_block_before(op.true_region.blocks[0], op)

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_inline_block_after():
    """Test the inlining of a block after an operation."""

    prog = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {
    scf.if(%0 : !i1) {
      %1 : !i32 = arith.constant() ["value" = 2 : !i32]
    }
  }
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {
    scf.if(%0 : !i1) {}
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  }
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: If, rewriter: PatternRewriter):
        if len(op.true_region.blocks[0].ops) > 0 and isinstance(
                op.true_region.blocks[0].ops[0], If):
            inner_if_block = op.true_region.blocks[0].ops[
                0].true_region.blocks[0]
            rewriter.inline_block_after(inner_if_block,
                                        op.true_region.blocks[0].ops[0])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_move_region_contents_to_new_regions():
    """Test moving a region outside of a region."""

    prog = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  }
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  scf.if(%0 : !i1) {}
  scf.if(%0 : !i1) {
    %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  } {}
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(op: ModuleOp, rewriter: PatternRewriter):
        new_region = rewriter.move_region_contents_to_new_regions(
            op.ops[1].regions[0])
        new_if = If.get(op.ops[1].cond, [], new_region,
                        Region.from_operation_list([]))
        rewriter.insert_op_after(new_if, op.ops[1])

    rewrite_and_compare(
        prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))
