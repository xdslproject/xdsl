from xdsl.ir import *


class Rewriter:

    @staticmethod
    def erase_op(op: Operation):
        """
        Erase an operation.
        Check that the operation has no uses, and has a parent.
        """
        for result in op.results:
            assert len(result.uses
                       ) == 0, "Cannot erase an operation that still has uses"
        assert op.parent is not None, "Cannot erase an operation that has no parents"

        block = op.parent
        block.erase_op(op)

    @staticmethod
    def replace_op(op: Operation,
                   new_ops: Union[Operation, List[Operation]],
                   new_results: Optional[List[SSAValue]] = None):
        """
        Replace an operation with multiple new ones.
        If new_results is specified, map the results of the deleted operations with these SSA values.
        Otherwise, use the results of the last operation added.
        """
        if op.parent is None:
            raise ValueError("Cannot replace an operation without a parent")
        block = op.parent

        if not isinstance(new_ops, list):
            new_ops = [new_ops]
        if len(new_ops) == 0:
            new_results = []
        if new_results is None:
            new_results = new_ops[-1].results

        if len(op.results) != len(new_results):
            raise ValueError(
                f"Expected {len(op.results)} new results, but got {len(new_results)}"
            )

        for old_result, new_result in zip(op.results, new_results):
            old_result.replace_by(new_result)

        op_idx = block.get_operation_index(op)
        block.erase_op(op_idx)
        block.insert_op(new_ops, op_idx)
