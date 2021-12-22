from xdsl.printer import Printer
from xdsl.dialects.builtin import Builtin, IntegerAttr
from xdsl.parser import Parser
from xdsl.dialects.std import Std, Constant, Addi, Muli
from xdsl.ir import MLContext
from xdsl.rewriter import *
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

    prog = \
    """module() {
%0 : !i32 = std.constant() ["value" = 42 : !i32]
%1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 43 : !i64]
  %1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    class RewriteConst(RewritePattern):
        def match_and_rewrite(
                self, op: Operation, new_operands: Optional[List[SSAValue]]
        ) -> Optional[RewriteAction]:
            if isinstance(op, Constant):
                return RewriteAction.from_op_list([
                    std.constant_from_attr(
                        IntegerAttr.from_int_and_width(43, 64), std.i32)
                ])
            return None

    rewrite_and_compare(
        ctx, prog, expected,
        PatternRewriteWalker(RewriteConst(), apply_recursively=False))


def test_op_type_rewrite_pattern_method_decorator():
    """Test op_type_rewrite_pattern decorator on methods."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)

    prog = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 42 : !i32]
  %1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 43 : !i64]
  %1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    class RewriteConst(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(
                self, op: Constant, new_operands: Optional[List[SSAValue]]
        ) -> Optional[RewriteAction]:
            return RewriteAction.from_op_list([
                std.constant_from_attr(IntegerAttr.from_int_and_width(43, 64),
                                       std.i32)
            ])

    rewrite_and_compare(
        ctx, prog, expected,
        PatternRewriteWalker(RewriteConst(), apply_recursively=False))


def test_op_type_rewrite_pattern_static_decorator():
    """Test op_type_rewrite_pattern decorator on static functions."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)

    prog = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 42 : !i32]
  %1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 43 : !i64]
  %1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(
            op: Constant,
            new_operands: Optional[List[SSAValue]]) -> Optional[RewriteAction]:
        return RewriteAction.from_op_list([
            std.constant_from_attr(IntegerAttr.from_int_and_width(43, 64),
                                   std.i32)
        ])

    rewrite_and_compare(
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=False))


def test_recursive_rewriter():
    """Test recursive walks on operations created by rewrites."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)

    prog = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 1 : !i64]
  %1 : !i32 = std.constant() ["value" = 1 : !i64]
  %2 : !i32 = std.addi(%0 : !i32, %1 : !i32)
  %3 : !i32 = std.constant() ["value" = 1 : !i64]
  %4 : !i32 = std.addi(%2 : !i32, %3 : !i32)
  %5 : !i32 = std.constant() ["value" = 1 : !i64]
  %6 : !i32 = std.addi(%4 : !i32, %5 : !i32)
  %7 : !i32 = std.constant() ["value" = 1 : !i64]
  %8 : !i32 = std.addi(%6 : !i32, %7 : !i32)
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(
            op: Constant,
            new_operands: Optional[List[SSAValue]]) -> Optional[RewriteAction]:
        val = op.value.value.data
        if val == 0 or val == 1:
            return None
        constant_op = std.constant_from_attr(
            IntegerAttr.from_int_and_width(val - 1, 64), std.i32)
        constant_one = std.constant_from_attr(
            IntegerAttr.from_int_and_width(1, 64), std.i32)
        add_op = std.addi(constant_op, constant_one)
        return RewriteAction.from_op_list([constant_op, constant_one, add_op])

    rewrite_and_compare(
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite),
                             apply_recursively=True))


def test_greedy_rewrite_pattern_applier():
    """Test GreedyRewritePatternApplier."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)

    prog = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 42 : !i32]
  %1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 43 : !i64]
  %1 : !i32 = std.muli(%0 : !i32, %0 : !i32)
}"""

    @op_type_rewrite_pattern
    def constant_rewrite(
            op: Constant,
            new_operands: List[SSAValue]) -> Optional[RewriteAction]:
        return RewriteAction.from_op_list([
            std.constant_from_attr(IntegerAttr.from_int_and_width(43, 64),
                                   std.i32)
        ])

    @op_type_rewrite_pattern
    def addi_rewrite(op: Addi,
                     new_operands: List[SSAValue]) -> Optional[RewriteAction]:
        return RewriteAction.from_op_list(
            [std.muli(op.input1.op, op.input2.op)])

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

    prog = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(
            op: Constant,
            new_operands: Optional[List[SSAValue]]) -> Optional[RewriteAction]:
        return RewriteAction([], [None])

    rewrite_and_compare(
        ctx, prog, expected,
        PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite)))


def test_operation_deletion_failure():
    """Test rewrites where SSA values are deleted."""
    ctx = MLContext()
    builtin = Builtin(ctx)
    std = Std(ctx)

    prog = \
"""module() {
  %0 : !i32 = std.constant() ["value" = 5 : !i32]
  %1 : !i32 = std.addi(%0 : !i32, %0 : !i32)
}"""

    @op_type_rewrite_pattern
    def match_and_rewrite(
            op: Constant,
            new_operands: Optional[List[SSAValue]]) -> Optional[RewriteAction]:
        return RewriteAction([], [None])

    parser = Parser(ctx, prog)
    module = parser.parse_op()
    walker = PatternRewriteWalker(AnonymousRewritePattern(match_and_rewrite))

    # Check that the rewrite fails
    try:
        walker.rewrite_module(module)
        assert False
    except Exception as e:
        pass
