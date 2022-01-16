from xdsl.ir import *


class Rewriter:

    @staticmethod
    def erase_op(op: Operation):
        """
        Erase an operation.
        Check that the operation has no uses, and has a parent.
        """
        assert op.parent is not None, "Cannot erase an operation that has no parents"

        block = op.parent
        block.erase_op(op)

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
        if block.is_ancestor(target_block):
            raise Exception("Cannot inline a block in a children block.")
        ops = block.ops.copy()
        for op in ops:
            op.detach()
        target_block.insert_op(ops, pos)

    @staticmethod
    def inline_block_before(block: Block, op: Operation):
        if op.parent is None:
            raise Exception(
                "Cannot inline a block before a toplevel operation")
        op_block = op.parent
        op_pos = op_block.get_operation_index(op)
        Rewriter.inline_block_at_pos(block, op_block, op_pos)

    @staticmethod
    def inline_block_after(block: Block, op: Operation):
        if op.parent is None:
            raise Exception(
                "Cannot inline a block before a toplevel operation")
        op_block = op.parent
        op_pos = op_block.get_operation_index(op)
        Rewriter.inline_block_at_pos(block, op_block, op_pos + 1)
