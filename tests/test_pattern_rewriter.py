from conftest import assert_print_op

from xdsl.dialects import test
from xdsl.dialects.arith import Addi, Arith, Constant, Muli
from xdsl.dialects.builtin import Builtin, IntegerAttr, ModuleOp, StringAttr, i32, i64
from xdsl.ir import MLContext, Operation
from xdsl.parser import Parser
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


def rewrite_and_compare(prog: str, expected_prog: str, walker: PatternRewriteWalker):
    ctx = MLContext(allow_unregistered=True)
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Arith)
    ctx.register_dialect(test.Test)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    walker.rewrite_module(module)

    assert_print_op(module, expected_prog, None)


def test_non_recursive_rewrite():
    """Test a simple non-recursive rewrite"""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 43 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    class RewriteConst(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(op, Constant):
                new_constant = Constant.from_int_and_width(43, i32)
                rewriter.replace_matched_op([new_constant])

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(RewriteConst(), apply_recursively=False)
    )


def test_non_recursive_rewrite_reversed():
    """Test a simple non-recursive rewrite with reverse walk order."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 43 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    class RewriteConst(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if isinstance(op, Constant):
                new_constant = Constant.from_int_and_width(43, i32)
                rewriter.replace_matched_op([new_constant])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(
            RewriteConst(), apply_recursively=False, walk_reverse=True
        ),
    )


def test_op_type_rewrite_pattern_method_decorator():
    """Test op_type_rewrite_pattern decorator on methods."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 43 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    class RewriteConst(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, __op__: Constant, rewriter: PatternRewriter):
            rewriter.replace_matched_op(Constant.from_int_and_width(43, i32))

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(RewriteConst(), apply_recursively=False)
    )


def test_op_type_rewrite_pattern_union_type():
    """Test op_type_rewrite_pattern decorator on static functions."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
  %2 = "test"(%0, %1) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %2 = "test"(%0, %1) : (i32, i32) -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, __op__: Constant | Addi, rewriter: PatternRewriter):
            rewriter.replace_matched_op(Constant.from_int_and_width(42, i32))

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=False)
    )


def test_recursive_rewriter():
    """Test recursive walks on operations created by rewrites."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
  %3 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %4 = "arith.addi"(%2, %3) : (i32, i32) -> i32
  %5 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %6 = "arith.addi"(%4, %5) : (i32, i32) -> i32
  %7 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %8 = "arith.addi"(%6, %7) : (i32, i32) -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: Constant, rewriter: PatternRewriter):
            if not isa(op.value, IntegerAttr):
                return
            val = op.value.value.data
            if val == 0 or val == 1:
                return
            constant_op = Constant.from_attr(
                IntegerAttr.from_int_and_width(val - 1, 32), i32
            )
            constant_one = Constant.from_attr(
                IntegerAttr.from_int_and_width(1, 32), i32
            )
            add_op = Addi(constant_op, constant_one)
            rewriter.replace_matched_op([constant_op, constant_one, add_op])

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=True)
    )


def test_recursive_rewriter_reversed():
    """Test recursive walks on operations created by rewrites, in reverse walk order."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
  %3 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %4 = "arith.addi"(%2, %3) : (i32, i32) -> i32
  %5 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %6 = "arith.addi"(%4, %5) : (i32, i32) -> i32
  %7 = "arith.constant"() {"value" = 1 : i32} : () -> i32
  %8 = "arith.addi"(%6, %7) : (i32, i32) -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: Constant, rewriter: PatternRewriter):
            if not isa(op.value, IntegerAttr):
                return
            val = op.value.value.data
            if val == 0 or val == 1:
                return
            constant_op = Constant.from_attr(
                IntegerAttr.from_int_and_width(val - 1, 32), i32
            )
            constant_one = Constant.from_attr(
                IntegerAttr.from_int_and_width(1, 32), i32
            )
            add_op = Addi(constant_op, constant_one)
            rewriter.replace_matched_op([constant_op, constant_one, add_op])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=True, walk_reverse=True),
    )


def test_greedy_rewrite_pattern_applier():
    """Test GreedyRewritePatternApplier."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 43 : i32} : () -> i32
  %1 = "arith.muli"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    class ConstantRewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, __op__: Constant, rewriter: PatternRewriter):
            rewriter.replace_matched_op([Constant.from_int_and_width(43, i32)])

    class AddiRewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: Addi, rewriter: PatternRewriter):
            rewriter.replace_matched_op([Muli(op.lhs, op.rhs)])

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(
            GreedyRewritePatternApplier([ConstantRewrite(), AddiRewrite()]),
            apply_recursively=False,
        ),
    )


def test_insert_op_before_matched_op():
    """Test rewrites where operations are inserted before the matched operation."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, __cst__: Constant, rewriter: PatternRewriter):
            new_cst = Constant.from_int_and_width(42, i32)

            rewriter.insert_op_before_matched_op(new_cst)

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=False)
    )


def test_insert_op_at_start():
    """Test rewrites where operations are inserted with a given position."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, mod: ModuleOp, rewriter: PatternRewriter):
            new_cst = Constant.from_int_and_width(42, i32)

            rewriter.insert_op_at_start(new_cst, mod.regions[0].blocks[0])

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=False)
    )


def test_insert_op_at_end():
    """
    Test rewrites where operations are inserted with a negative position.
    """

    prog = ModuleOp([Constant.from_int_and_width(5, 32)])

    to_be_inserted = Constant.from_int_and_width(42, 32)

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, mod: ModuleOp, rewriter: PatternRewriter):
            rewriter.insert_op_at_end(to_be_inserted, mod.regions[0].blocks[0])

    PatternRewriteWalker(Rewrite(), apply_recursively=False).rewrite_module(prog)

    assert to_be_inserted in prog.ops
    assert prog.body.block.get_operation_index(to_be_inserted) == 1


def test_insert_op_before():
    """Test rewrites where operations are inserted before a given operation."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, mod: ModuleOp, rewriter: PatternRewriter):
            new_cst = Constant.from_int_and_width(42, i32)

            first_op = mod.ops.first
            assert first_op is not None
            rewriter.insert_op_before(new_cst, first_op)

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=False)
    )


def test_insert_op_after():
    """Test rewrites where operations are inserted after a given operation."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 42 : i32} : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, mod: ModuleOp, rewriter: PatternRewriter):
            new_cst = Constant.from_int_and_width(42, i32)

            first_op = mod.ops.first
            assert first_op is not None
            rewriter.insert_op_after(new_cst, first_op)

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=False)
    )


def test_insert_op_after_matched_op():
    """Test rewrites where operations are inserted after a given operation."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 42 : i32} : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, __cst__: Constant, rewriter: PatternRewriter):
            new_cst = Constant.from_int_and_width(42, i32)

            rewriter.insert_op_after_matched_op(new_cst)

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=False)
    )


def test_insert_op_after_matched_op_reversed():
    """Test rewrites where operations are inserted after a given operation."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
  %1 = "arith.constant"() {"value" = 42 : i32} : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, __cst__: Constant, rewriter: PatternRewriter):
            new_cst = Constant.from_int_and_width(42, i32)

            rewriter.insert_op_after_matched_op(new_cst)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False, walk_reverse=True),
    )


def test_operation_deletion():
    """Test rewrites where SSA values are deleted."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
^0:
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, __op__: Constant, rewriter: PatternRewriter):
            rewriter.erase_matched_op()

    rewrite_and_compare(prog, expected, PatternRewriteWalker(Rewrite()))


def test_operation_deletion_reversed():
    """
    Test rewrites where SSA values are deleted.
    They have to be deleted in order for the rewrite to not fail.
    """

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
^0:
}) : () -> ()"""

    class EraseAll(RewritePattern):
        def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter):
            if not isinstance(op, ModuleOp):
                rewriter.erase_matched_op()

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(EraseAll(), walk_reverse=True),
    )


def test_operation_deletion_failure():
    """Test rewrites where SSA values are deleted with still uses."""

    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Arith)

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, __op__: Constant, rewriter: PatternRewriter):
            rewriter.erase_matched_op()

    parser = Parser(ctx, prog)
    module = parser.parse_module()
    walker = PatternRewriteWalker(Rewrite())

    # Check that the rewrite fails
    try:
        walker.rewrite_module(module)
        assert False
    except Exception:
        pass


def test_delete_inner_op():
    """Test rewrites where an operation inside a region of the matched op is deleted."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
^0:
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
            first_op = op.ops.first

            assert first_op is not None
            rewriter.erase_op(first_op)

    rewrite_and_compare(prog, expected, PatternRewriteWalker(Rewrite()))


def test_replace_inner_op():
    """Test rewrites where an operation inside a region of the matched op is deleted."""

    prog = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 5 : i32} : () -> i32
}) : () -> ()"""

    expected = """"builtin.module"() ({
  %0 = "arith.constant"() {"value" = 42 : i32} : () -> i32
}) : () -> ()"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
            first_op = op.ops.first

            assert first_op is not None
            rewriter.replace_op(first_op, [Constant.from_int_and_width(42, i32)])

    rewrite_and_compare(prog, expected, PatternRewriteWalker(Rewrite()))


def test_block_argument_type_change():
    """Test the modification of a block argument type."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() ({
  ^0(%1 : !test.type<"int">):
    %2 = "test.op"() : () -> !test.type<"int">
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() ({
  ^0(%1 : i64):
    %2 = "test.op"() : () -> !test.type<"int">
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                rewriter.modify_block_argument_type(
                    matched_op.regs[0].blocks[0].args[0], i64
                )

    rewrite_and_compare(
        prog, expected, PatternRewriteWalker(Rewrite(), apply_recursively=False)
    )


def test_block_argument_erasure():
    """Test the erasure of a block argument."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() ({
  ^0(%1 : !test.type<"int">):
    %2 = "test.op"() : () -> !test.type<"int">
  }, {
  ^0:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() ({
    %1 = "test.op"() : () -> !test.type<"int">
  }, {
  ^0:
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
  ^0:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() ({
  ^0(%1 : !test.type<"int">):
    %2 = "test.op"() : () -> !test.type<"int">
  }, {
  ^1:
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
  ^0:
    %2 = "test.op"() : () -> !test.type<"int">
  }, {
  ^1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() : () -> !test.type<"int">
  %2 = "test.op"() ({
  ^0:
  }, {
  ^1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                rewriter.inline_block_before_matched_op(matched_op.regs[0].blocks[0])

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
    ^1:
    }) : () -> !test.type<"int">
  }, {
  ^2:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %2 = "test.op"() : () -> !test.type<"int">
    %3 = "test.op"() ({
    ^0:
    }, {
    ^1:
    }) : () -> !test.type<"int">
  }, {
  ^2:
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
                    rewriter.inline_block_before(inner_block, first_op)

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
  ^0:
    %2 = "test.op"() : () -> !test.type<"int">
  }, {
  ^1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() : () -> !test.type<"int">
  %2 = "test.op"() ({
  ^0:
  }, {
  ^1:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, matched_op: test.TestOp, rewriter: PatternRewriter):
            if matched_op.regs and matched_op.regs[0].blocks:
                rewriter.inline_block_before(matched_op.regs[0].blocks[0], matched_op)

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
      ^1:
    }) : () -> !test.type<"int">
  }, {
  ^2:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %2 = "test.op"() ({
    ^0:
    }, {
    ^1:
    }) : () -> !test.type<"int">
    %3 = "test.op"() : () -> !test.type<"int">
  }, {
  ^2:
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
                        rewriter.inline_block_after(inner_block, first_op)

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
    ^1:
    }) : () -> !test.type<"int">
  }, {
  ^2:
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
    %2 = "test.op"() ({
    ^0:
    }, {
    ^1:
    }) : () -> !test.type<"int">
  }, {
  ^2:
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
                        rewriter.inline_block_after(inner_block, matched_op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_move_region_contents_to_new_regions():
    """Test moving a region outside of a region."""

    prog = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
  ^0:
    %2 = "test.op"() : () -> !test.type<"int">
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() : () -> !test.type<"int">
  %1 = "test.op"() ({
  }) : () -> !test.type<"int">
  %2 = "test.op"() ({
    %3 = "test.op"() : () -> !test.type<"int">
  }) : () -> !test.type<"int">
}) : () -> ()
"""

    class Rewrite(RewritePattern):
        @op_type_rewrite_pattern
        def match_and_rewrite(self, op: ModuleOp, rewriter: PatternRewriter):
            ops_iter = iter(op.ops)

            _ = next(ops_iter)  # skip first op
            old_op = next(ops_iter)
            assert isinstance(old_op, test.TestOp)
            new_region = rewriter.move_region_contents_to_new_regions(old_op.regions[0])
            res_types = [r.type for r in old_op.results]
            new_op = test.TestOp.create(result_types=res_types, regions=[new_region])
            rewriter.insert_op_after(new_op, old_op)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )


def test_insert_same_block():
    """Test rewriter on ops without results"""
    prog = """\
"builtin.module"() ({
  %0 = "test.op"() {"label" = "a"} : () -> i32
  %1 = "test.op"() {"label" = "b"} : () -> i32
  %2 = "test.op"() {"label" = "c"} : () -> i32
  "func.return"() : () -> ()
}) : () -> ()
"""

    expected = """\
"builtin.module"() ({
  %0 = "test.op"() {"label" = "alloc"} : () -> i32
  %1 = "test.op"() {"label" = "a"} : () -> i32
  "test.op"(%0) {"label" = "init"} : (i32) -> ()
  %2 = "test.op"() {"label" = "c"} : () -> i32
  "test.op"(%0) {"label" = "dealloc"} : (i32) -> ()
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
            rewriter.insert_op_at_start(alloc, block)
            # Deallocate after last use
            rewriter.insert_op_before(dealloc, last_op)
            # Init instead of creating, and replace result with allocated value
            rewriter.replace_matched_op(init, alloc.res)

    rewrite_and_compare(
        prog,
        expected,
        PatternRewriteWalker(Rewrite(), apply_recursively=False),
    )
