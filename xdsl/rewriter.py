from typing import Sequence

from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue


class Rewriter:
    @staticmethod
    def erase_op(op: Operation, safe_erase: bool = True):
        """
        Erase an operation.
        Check that the operation has no uses, and has a parent.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        assert op.parent is not None, "Cannot erase an operation that has no parents"

        block = op.parent
        block.erase_op(op, safe_erase=safe_erase)

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
    def inline_block_at_end(inlined_block: Block, extended_block: Block):
        """
        Move the block operations to the end of another block.
        This block should not be a parent of the block to move to.
        The block operations should not use the block arguments.
        """
        if inlined_block.is_ancestor(extended_block):
            raise Exception("Cannot inline a block in a child block.")
        for op in inlined_block.ops:
            for operand in op.operands:
                if (
                    isinstance(operand, BlockArgument)
                    and operand.block is extended_block
                ):
                    raise Exception(
                        "Cannot inline block which has operations using "
                        "the block arguments."
                    )

        ops = list(inlined_block.ops)
        for block_op in ops:
            block_op.detach()

        extended_block.add_ops(ops)

    @staticmethod
    def inline_block_at_start(inlined_block: Block, extended_block: Block):
        """
        Move the block operations to the start of another block.
        This block should not be a parent of the block to move to.
        The block operations should not use the block arguments.
        """
        first_op_of_extended_block = extended_block.first_op
        if first_op_of_extended_block is None:
            Rewriter.inline_block_at_end(inlined_block, extended_block)
        else:
            Rewriter.inline_block_before(inlined_block, first_op_of_extended_block)

    @staticmethod
    def inline_block_before(block: Block, op: Operation):
        """
        Move the block operations before another operation.
        The block should not be a parent of the operation.
        The block operations should not use the block arguments.
        """
        if op.parent is None:
            raise Exception("Cannot inline a block before a toplevel operation")

        ops = list(block.ops)
        for block_op in ops:
            block_op.detach()

        op.parent.insert_ops_before(ops, op)

    @staticmethod
    def inline_block_after(block: Block, op: Operation):
        """
        Move the block operations after another operation.
        The block should not be a parent of the operation.
        The block operations should not use the block arguments.
        """
        if op.parent is None:
            raise Exception("Cannot inline a block before a toplevel operation")

        ops = list(block.ops)
        for block_op in ops:
            block_op.detach()

        op.parent.insert_ops_after(ops, op)

    @staticmethod
    def insert_block_after(block: Block | list[Block], target: Block):
        """
        Insert one or multiple blocks after another block.
        The blocks to insert should be detached from any region.
        The target block should not be contained in the block to insert.
        """
        if target.parent is None:
            raise Exception("Cannot move a block after a toplevel op")
        region = target.parent
        block_list = block if isinstance(block, list) else [block]
        if len(block_list) == 0:
            return
        pos = region.get_block_index(target)
        region.insert_block(block_list, pos + 1)

    @staticmethod
    def insert_block_before(block: Block | list[Block], target: Block):
        """
        Insert one or multiple block before another block.
        The blocks to insert should be detached from any region.
        The target block should not be contained in the block to insert.
        """
        if target.parent is None:
            raise Exception("Cannot move a block after a toplevel op")
        region = target.parent
        block_list = block if isinstance(block, list) else [block]
        pos = region.get_block_index(target)
        region.insert_block(block_list, pos)

    @staticmethod
    def insert_op_after(op: Operation, new_op: Operation):
        """Inserts a new operation after another operation."""
        if op.parent is None:
            raise Exception("Cannot insert an operation after a toplevel operation")
        op.parent.insert_ops_after((new_op,), op)

    @staticmethod
    def insert_op_before(op: Operation, new_op: Operation):
        """Inserts a new operation before another operation."""
        if op.parent is None:
            raise Exception("Cannot insert an operation before a toplevel operation")
        op.parent.insert_ops_before((new_op,), op)

    @staticmethod
    def move_region_contents_to_new_regions(region: Region) -> Region:
        """Move the region blocks to a new region."""
        new_region = Region()
        for block in region.blocks:
            block.parent = None
            new_region.add_block(block)
        region.blocks = []
        return new_region
