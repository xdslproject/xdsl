from typing import Callable
from conftest import assert_print_op

from xdsl.dialects.arith import Arith, Constant, Addi
from xdsl.dialects.builtin import ModuleOp, Builtin, i32, i64
from xdsl.dialects.scf import Scf, Yield
from xdsl.dialects.func import Func
from xdsl.ir import MLContext, Block
from xdsl.parser import Parser
from xdsl.rewriter import Rewriter


def rewrite_and_compare(
    prog: str, expected_prog: str, transformation: Callable[[ModuleOp, Rewriter], None]
):
    ctx = MLContext()
    ctx.register_dialect(Builtin)
    ctx.register_dialect(Arith)
    ctx.register_dialect(Scf)
    ctx.register_dialect(Func)

    parser = Parser(ctx, prog)
    module = parser.parse_module()

    rewriter = Rewriter()
    transformation(module, rewriter)

    assert_print_op(module, expected_prog, None)


def test_operation_deletion():
    """Test rewrites where SSA values are deleted."""

    prog = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = """builtin.module() {
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops[0]
        rewriter.erase_op(constant_op)

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement
def test_replace_op_one_op():
    prog = """builtin.module() {
%0 : !i32 = arith.constant() ["value" = 42 : !i32]
%1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops[0]
        new_constant_op = Constant.from_int_and_width(43, i32)
        rewriter.replace_op(constant_op, new_constant_op)

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement with multiple ops
def test_replace_op_multiple_op():
    prog = """builtin.module() {
%0 : !i32 = arith.constant() ["value" = 2 : !i32]
%1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
  %2 : !i32 = arith.addi(%1 : !i32, %1 : !i32)
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops[0]
        new_constant = Constant.from_int_and_width(1, i32)
        new_add = Addi(new_constant, new_constant)

        rewriter.replace_op(constant_op, [new_constant, new_add])

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement with manually specified results
def test_replace_op_new_results():
    prog = """builtin.module() {
%0 : !i32 = arith.constant() ["value" = 2 : !i32]
%1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
%2 : !i32 = arith.muli(%1 : !i32, %1 : !i32)
}"""

    expected = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 2 : !i32]
  %1 : !i32 = arith.muli(%0 : !i32, %0 : !i32)
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        add_op = module.ops[1]
        assert isinstance(add_op, Addi)

        rewriter.replace_op(add_op, [], [add_op.lhs])

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_at_pos():
    """Test the inlining of a block at a certain position."""
    prog = """builtin.module() {
%0 : !i1 = arith.constant() ["value" = true]
scf.if(%0 : !i1) {
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
}
}"""

    expected = """builtin.module() {
  %0 : !i1 = arith.constant() ["value" = true]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  scf.if(%0 : !i1) {
  }
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        if_op = module.ops[1]
        module_block = module.regions[0].blocks[0]
        if_block = if_op.regions[0].blocks[0]

        rewriter.inline_block_at_pos(if_block, module_block, 1)

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_before():
    """Test the inlining of a block before an operation."""
    prog = """builtin.module() {
%0 : !i1 = arith.constant() ["value" = true]
scf.if(%0 : !i1) {
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
}
}"""

    expected = """builtin.module() {
  %0 : !i1 = arith.constant() ["value" = true]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  scf.if(%0 : !i1) {
  }
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        if_op = module.ops[1]
        if_block = if_op.regions[0].blocks[0]

        rewriter.inline_block_before(if_block, if_op)

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_after():
    """Test the inlining of a block after an operation."""
    prog = """builtin.module() {
%0 : !i1 = arith.constant() ["value" = true]
scf.if(%0 : !i1) {
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
}
}"""

    expected = """builtin.module() {
  %0 : !i1 = arith.constant() ["value" = true]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  scf.if(%0 : !i1) {
  }
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        if_op = module.ops[1]
        constant_op = module.ops[0]
        if_block = if_op.regions[0].blocks[0]

        rewriter.inline_block_after(if_block, constant_op)

    rewrite_and_compare(prog, expected, transformation)


def test_insert_block():
    """Test the insertion of a block in a region."""
    prog = """builtin.module() {
  %0 : !i1 = arith.constant() ["value" = true]
}"""

    expected = """builtin.module() {
^0:
^1:
  %0 : !i1 = arith.constant() ["value" = true]
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        module.regions[0].insert_block(Block(), 0)

    rewrite_and_compare(prog, expected, transformation)


def test_insert_block2():
    """Test the insertion of a block in a region."""
    prog = """builtin.module() {
  %0 : !i1 = arith.constant() ["value" = true]
}"""

    expected = """builtin.module() {
^0:
  %0 : !i1 = arith.constant() ["value" = true]
^1:
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        module.regions[0].insert_block(Block(), 1)

    rewrite_and_compare(prog, expected, transformation)


def test_insert_block_before():
    """Test the insertion of a block before another block."""
    prog = """builtin.module() {
  %0 : !i1 = arith.constant() ["value" = true]
}"""

    expected = """builtin.module() {
^0:
^1:
  %0 : !i1 = arith.constant() ["value" = true]
}"""

    def insert_empty_block_before(module: ModuleOp, rewriter: Rewriter) -> None:
        rewriter.insert_block_before(Block(), module.regions[0].blocks[0])

    rewrite_and_compare(prog, expected, insert_empty_block_before)


def test_insert_block_after():
    """Test the insertion of a block after another block."""
    prog = """builtin.module() {
  %0 : !i1 = arith.constant() ["value" = true]
}"""

    expected = """builtin.module() {
^0:
  %0 : !i1 = arith.constant() ["value" = true]
^1:
}"""

    def insert_empty_block_after(module: ModuleOp, rewriter: Rewriter) -> None:
        rewriter.insert_block_after(Block(), module.regions[0].blocks[0])

    rewrite_and_compare(prog, expected, insert_empty_block_after)


def test_insert_op_before():
    """Test the insertion of an operation before another operation."""
    prog = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
}"""

    expected = """builtin.module() {
  %0 : !i64 = arith.constant() ["value" = 34 : !i64]
  %1 : !i32 = arith.constant() ["value" = 43 : !i32]
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant = Constant.from_int_and_width(34, i64)
        first_op = module.regions[0].blocks[0].first_op
        assert first_op is not None
        rewriter.insert_op_before(first_op, constant)

    rewrite_and_compare(prog, expected, transformation)


def test_insert_op_after():
    """Test the insertion of an operation after another operation."""
    prog = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
}"""

    expected = """builtin.module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
  %1 : !i64 = arith.constant() ["value" = 34 : !i64]
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant = Constant.from_int_and_width(34, i64)
        first_op = module.regions[0].blocks[0].first_op
        assert first_op is not None
        rewriter.insert_op_after(first_op, constant)

    rewrite_and_compare(prog, expected, transformation)


def test_preserve_naming_single_op():
    """Test the preservation of names of SSAValues"""
    prog = """builtin.module() {
   %i : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%i : !i32, %i : !i32)
}"""

    expected = """builtin.module() {
  %i : !i32 = arith.constant() ["value" = 1 : !i32]
  %0 : !i32 = arith.addi(%i : !i32, %i : !i32)
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops[0]
        new_constant = Constant.from_int_and_width(1, i32)

        rewriter.replace_op(constant_op, [new_constant])

    rewrite_and_compare(prog, expected, transformation)


def test_preserve_naming_multiple_ops():
    """Test the preservation of names of SSAValues for transformations to multiple ops"""
    prog = """builtin.module() {
   %i : !i32 = arith.constant() ["value" = 42 : !i32]
    %1 : !i32 = arith.addi(%i : !i32, %i : !i32)
}"""

    expected = """builtin.module() {
  %i : !i32 = arith.constant() ["value" = 1 : !i32]
  %i_1 : !i32 = arith.addi(%i : !i32, %i : !i32)
  %0 : !i32 = arith.addi(%i_1 : !i32, %i_1 : !i32)
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops[0]
        new_constant = Constant.from_int_and_width(1, i32)
        new_add = Addi(new_constant, new_constant)

        rewriter.replace_op(constant_op, [new_constant, new_add])

    rewrite_and_compare(prog, expected, transformation)


def test_no_result_rewriter():
    """Test rewriter on ops without results"""
    prog = """builtin.module() {
   func.return()
}"""

    expected = """builtin.module() {
  scf.yield()
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        return_op = module.ops[0]
        new_op = Yield.get()

        rewriter.replace_op(return_op, [new_op])

    rewrite_and_compare(prog, expected, transformation)
