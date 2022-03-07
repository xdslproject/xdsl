from typing import Tuple

from xdsl.ir import *


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
    def replace_op(op: Operation,
                   new_ops: Union[Operation, List[Operation]],
                   new_results: Optional[List[Optional[SSAValue]]] = None,
                   safe_erase: bool = True):
        """
        Replace an operation with multiple new ones.
        If new_results is specified, map the results of the deleted operations with these SSA values.
        Otherwise, use the results of the last operation added.
        None elements in new_results are the SSA values to delete.
        If safe_erase is False, then operations can be deleted even if they are still used.
        """
        if op.parent is None:
            raise ValueError("Cannot replace an operation without a parent")
        block = op.parent

        if not isinstance(new_ops, list):
            new_ops = [new_ops]
        if new_results is None:
            new_results = [] if len(new_ops) == 0 else new_ops[-1].results

        if len(op.results) != len(new_results):
            raise ValueError(
                f"Expected {len(op.results)} new results, but got {len(new_results)}"
            )

        for old_result, new_result in zip(op.results, new_results):
            if new_result is None:
                if safe_erase:
                    old_result.replace_by(ErasedSSAValue(old_result.typ))
                elif len(old_result.uses) != 0:
                    raise Exception(
                        "SSA value was supposed to be destroyed, but still has uses."
                    )
            else:
                old_result.replace_by(new_result)

        op_idx = block.get_operation_index(op)
        block.erase_op(op_idx, safe_erase=safe_erase)
        block.insert_op(new_ops, op_idx)

    @staticmethod
    def inline_block_at_pos(block: Block, target_block: Block, pos: int):
        """
        Move the block operations to a given position in another block.
        This block should not be a parent of the block to move to.
        The block operations should not use the block arguments.
        """
        if block.is_ancestor(target_block):
            raise Exception("Cannot inline a block in a child block.")
        for op in block.ops:
            for operand in op.operands:
                if isinstance(operand,
                              BlockArgument) and operand.block is block:
                    raise Exception(
                        "Cannot inline block which has operations using the block arguments."
                    )
        ops = block.ops.copy()
        for op in ops:
            op.detach()
        target_block.insert_op(ops, pos)

    @staticmethod
    def inline_block_before(block: Block, op: Operation):
        """
        Move the block operations before another operation.
        The block should not be a parent of the operation.
        The block operations should not use the block arguments.
        """
        if op.parent is None:
            raise Exception(
                "Cannot inline a block before a toplevel operation")
        op_block = op.parent
        op_pos = op_block.get_operation_index(op)
        Rewriter.inline_block_at_pos(block, op_block, op_pos)

    @staticmethod
    def inline_block_after(block: Block, op: Operation):
        """
        Move the block operations after another operation.
        The block should not be a parent of the operation.
        The block operations should not use the block arguments.
        """
        if op.parent is None:
            raise Exception(
                "Cannot inline a block before a toplevel operation")
        op_block = op.parent
        op_pos = op_block.get_operation_index(op)
        Rewriter.inline_block_at_pos(block, op_block, op_pos + 1)

    @staticmethod
    def insert_block_after(block: Union[Block, List[Block]], target: Block):
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
    def insert_block_before(block: Union[Block, List[Block]], target: Block):
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
    def move_region_contents_to_new_regions(region: Region) -> Region:
        """Move the region blocks to a new region."""
        new_region = Region()
        for block in region.blocks:
            block.parent = None
            new_region.add_block(block)
        region.blocks = []
        return new_region
