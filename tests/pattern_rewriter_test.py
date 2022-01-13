from xdsl.printer import Printer
from xdsl.dialects.builtin import Builtin, IntegerAttr, i32
from xdsl.parser import Parser
from xdsl.dialects.std import Std
from xdsl.dialects.arith import Arith, Constant, Addi, Muli
from xdsl.ir import MLContext
from xdsl.pattern_rewriter import *
from io import StringIO


def rewrite_and_compare(ctx: MLContext, prog: str, expected_prog: str,
                        walker: PatternRewriteWalker):
    parser = Parser(ctx, prog)
    module = parser.parse_op()

    walker.rewrite_module(module)
    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected_prog.strip()


# Test a simple non-recursive rewrite
def test_non_recursive_rewrite():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(RewriteConst(), apply_recursively=False))


# Test a simple non-recursive rewrite
def test_non_recursive_rewrite_reversed():
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(RewriteConst(),
                             apply_recursively=False,
                             walk_reverse=True))


def test_op_type_rewrite_pattern_method_decorator():
    """Test op_type_rewrite_pattern decorator on methods."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(RewriteConst(), apply_recursively=False))


def test_op_type_rewrite_pattern_static_decorator():
    """Test op_type_rewrite_pattern decorator on static functions."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_recursive_rewriter():
    """Test recursive walks on operations created by rewrites."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=True))


def test_recursive_rewriter_reversed():
    """Test recursive walks on operations created by rewrites."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=True,
                             walk_reverse=True))


def test_greedy_rewrite_pattern_applier():
    """Test GreedyRewritePatternApplier."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(GreedyRewritePatternApplier([
            AnonymousRewritePattern(constant_rewrite),
            AnonymousRewritePattern(addi_rewrite)
        ]),
                             apply_recursively=False))


def test_operation_deletion():
    """Test rewrites where SSA values are deleted."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite)))


def test_operation_deletion_reversed():
    """
    Test rewrites where SSA values are deleted.
    They have to be deleted in order for the rewrite to not fail.
    """
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             walk_reverse=True))


def test_operation_deletion_failure():
    """Test rewrites where SSA values are deleted."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
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
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite)))


def test_replace_inner_op():
    """Test rewrites where an operation inside a region of the matched op is deleted."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)
    arith = Arith(ctx)

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
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite)))
