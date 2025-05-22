from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    OpResult,
    Region,
    SSAValue,
)


@dataclass(frozen=True)
class InsertPoint:
    """
    An insert point.
    It is either a point before an operation, or at the end of a block.

    See external [documentation](https://mlir.llvm.org/doxygen/classmlir_1_1OpBuilder_1_1InsertPoint.html).
    """

    block: Block
    """The block where the insertion point is in."""

    insert_before: Operation | None = field(default=None)
    """
    The insertion point is right before this operation.
    If the operation is None, the insertion point is at the end of the block.
    """

    def __post_init__(self) -> None:
        # Check that the insertion point is valid.
        # An insertion point can only be invalid if `insert_before` is an `Operation`,
        # and its parent is not `block`.
        if self.insert_before is not None:
            if self.insert_before.parent is not self.block:
                raise ValueError("Insertion point must be in the builder's `block`")

    @staticmethod
    def before(op: Operation) -> InsertPoint:
        """Gets the insertion point before an operation."""
        if (block := op.parent_block()) is None:
            raise ValueError("Operation insertion point must have a parent block")
        return InsertPoint(block, op)

    @staticmethod
    def after(op: Operation) -> InsertPoint:
        """Gets the insertion point after an operation."""
        block = op.parent_block()
        if block is None:
            raise ValueError("Operation insertion point must have a parent block")
        return InsertPoint(block, op.next_op)

    @staticmethod
    def at_start(block: Block) -> InsertPoint:
        """Gets the insertion point at the start of a block."""
        return InsertPoint(block, block.ops.first)

    @staticmethod
    def at_end(block: Block) -> InsertPoint:
        """Gets the insertion point at the end of a block."""
        return InsertPoint(block)


@dataclass(frozen=True)
class BlockInsertPoint:
    """
    An insert point for a block.
    It is either a point before a block, or after a block.
    """

    region: Region
    """The region where the insertion point is in."""

    insert_before: Block | None = field(default=None)
    """
    The insertion point is right before this block.
    If the block is None, the insertion point is at the end of the region.
    """

    def __post_init__(self) -> None:
        # Check that the insertion point is valid.
        # An insertion point can only be invalid if `insert_before` is a `Block`,
        # and its parent is not `region`.
        if self.insert_before is not None:
            if self.insert_before.parent is not self.region:
                raise ValueError("Insertion point must be in the builder's `region`")

    @staticmethod
    def before(block: Block) -> BlockInsertPoint:
        """Gets the insertion point before a block."""
        if (region := block.parent) is None:
            raise ValueError("Block insertion point must have a parent region")
        return BlockInsertPoint(region, block)

    @staticmethod
    def after(block: Block) -> BlockInsertPoint:
        """Gets the insertion point after a block."""
        region = block.parent
        if region is None:
            raise ValueError("Block insertion point must have a parent region")
        return BlockInsertPoint(region, block.next_block)

    @staticmethod
    def at_start(region: Region) -> BlockInsertPoint:
        """Gets the insertion point at the start of a region."""
        return BlockInsertPoint(region, region.first_block)

    @staticmethod
    def at_end(region: Region) -> BlockInsertPoint:
        """Gets the insertion point at the end of a region."""
        return BlockInsertPoint(region)


class Rewriter:
    @staticmethod
    def erase_op(op: Operation, safe_erase: bool = True):
        """
        Erase an operation.
        Check that the operation has no uses, and has a parent.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        if (block := op.parent) is not None:
            block.erase_op(op, safe_erase=safe_erase)
        else:
            op.erase(safe_erase=safe_erase)

    @staticmethod
    def replace_op(
        op: Operation,
        new_ops: Operation | Sequence[Operation],
        new_results: Sequence[SSAValue | None] | None = None,  # noqa
        safe_erase: bool = True,
    ):
        """
        Replace an operation with multiple new ones.
        If new_results is specified, map the results of the deleted operations with these
        SSA values.
        Otherwise, use the results of the last operation added.
        None elements in new_results are the SSA values to delete.
        If safe_erase is False, then operations can be deleted even if they are
        still used.
        """
        if op.parent is None:
            raise ValueError("Cannot replace an operation without a parent")
        block = op.parent

        if isinstance(new_ops, Operation):
            new_ops = [new_ops]
        if new_results is None:
            new_results = [] if len(new_ops) == 0 else new_ops[-1].results

        if len(op.results) != len(new_results):
            raise ValueError(
                f"Expected {len(op.results)} new results, but got {len(new_results)}"
            )

        for old_result, new_result in zip(op.results, new_results):
            if new_result is None:
                old_result.erase(safe_erase=safe_erase)
            else:
                old_result.replace_by(new_result)

        block.insert_ops_after(new_ops, op)

        if len(op.results):
            for new_op in new_ops:
                for res in new_op.results:
                    res.name_hint = op.results[0].name_hint

        block.erase_op(op, safe_erase=safe_erase)

    @staticmethod
    def replace_value_with_new_type(val: SSAValue, new_type: Attribute) -> SSAValue:
        """
        Replace a value with a value of a new type, and return the new value.
        This will insert the new value in the operation or block, and remove the existing
        value.
        """
        if isinstance(val, OpResult):
            operation = val.op
            index = val.index
            new_value = OpResult(new_type, operation, val.index)
            results = operation.results
            operation.results = (*results[:index], new_value, *results[index + 1 :])
        elif isinstance(val, BlockArgument):
            block = val.block
            index = val.index
            new_value = BlockArgument(new_type, block, index)
            args = block.args
            block._args = (  # pyright: ignore[reportPrivateUsage]
                *args[:index],
                new_value,
                *args[index + 1 :],
            )
        else:
            raise ValueError(
                f"Expected OpResult or BlockArgument, got {type(val).__name__}"
            )

        new_value.name_hint = val.name_hint
        val.replace_by(new_value)
        return new_value

    @staticmethod
    def inline_block(
        source: Block, insertion_point: InsertPoint, arg_values: Sequence[SSAValue] = ()
    ):
        """
        Move the block operations before another operation.
        The block should not be a parent of the operation.
        """
        # MLIR equivalent:
        # https://github.com/llvm/llvm-project/blob/96a3d05ed923d2abd51acb52984b83b9e8044924/mlir/lib/IR/PatternMatch.cpp#L290
        assert arg_values == () or len(arg_values) == len(source.args), (
            f"Expected {len(source.args)} replacement argument values, got "
            f"{len(arg_values)}"
        )

        # The source block will be deleted, so it should not have any users (i.e.,
        # there should be no predecessors).
        # TODO: check that the block has no predecessors

        #  assert not block.predecessors, "expected 'source' to have no predecessors"

        dest = insertion_point.block

        # TODO: verify that the successors will make sense after inlining
        # We currently cannot perform this check, just like the TODO above, due to lack
        # of infrastructure in xDSL
        # https://github.com/xdslproject/xdsl/issues/2066

        # if dest.last_op != op:
        #       The source block will be inserted in the middle of the dest block, so the
        #       source block should have no successors. Otherwise, the remainder of the dest
        #       block would be unreachable.
        #       assert not source.successors, "expected 'source' to have no successors");
        # else:
        #       The source block will be inserted at the end of the dest block, so the dest
        #       block should have no successors. Otherwise, the inserted operations will be
        #       unreachable.
        #       assert not dest.successors,  "expected 'dest' to have no successors");

        # Replace all of the successor arguments with the provided values.
        if arg_values:
            for arg, val in zip(source.args, arg_values, strict=True):
                arg.replace_by(val)

        # Move operations from the source block to the dest block and erase the
        # source block.
        ops = list(source.ops)
        for block_op in ops:
            block_op.detach()

        if (insert_before := insertion_point.insert_before) is not None:
            dest.insert_ops_before(ops, insert_before)
        else:
            dest.add_ops(ops)

        parent_region = source.parent
        if parent_region is not None:
            parent_region.detach_block(source)
        source.erase()

    @staticmethod
    def insert_block(block: Block | Iterable[Block], insert_point: BlockInsertPoint):
        """
        Insert one or multiple blocks at a given location.
        The blocks to insert should be detached from any region.
        The insertion point should not be contained in the block to insert.
        """
        region = insert_point.region
        if insert_point.insert_before is not None:
            region.insert_block_before(block, insert_point.insert_before)
        else:
            region.add_block(block)

    @staticmethod
    def insert_op(
        op_or_ops: Operation | Sequence[Operation], insertion_point: InsertPoint
    ):
        """Insert operations at a certain location in a block."""
        ops = (op_or_ops,) if isinstance(op_or_ops, Operation) else op_or_ops
        if insertion_point.insert_before is not None:
            insertion_point.block.insert_ops_before(ops, insertion_point.insert_before)
        else:
            insertion_point.block.add_ops(ops)

    @staticmethod
    def move_region_contents_to_new_regions(region: Region) -> Region:
        """Move the region blocks to a new region."""
        new_region = Region()
        region.move_blocks(new_region)
        return new_region

    @staticmethod
    def inline_region(region: Region, insertion_point: BlockInsertPoint) -> None:
        """Move the region blocks to a given location."""
        if insertion_point.insert_before is not None:
            region.move_blocks_before(insertion_point.insert_before)
        else:
            region.move_blocks(insertion_point.region)
