from collections.abc import Sequence

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialect_interfaces import ConstantMaterializationInterface
from xdsl.interfaces import HasFolderInterface
from xdsl.ir import Attribute, Operation, SSAValue, TypeAttribute
from xdsl.rewriter import Rewriter


class Folder:
    context: Context

    def try_fold(self, op: Operation) -> tuple[list[SSAValue], list[Operation]] | None:
        """
        Try to fold the given operation.
        Returns a tuple the list of SSAValues that replace the results of the operation,
        and a list of new operations that were created during folding.
        If the operation could not be folded, returns None.
        """

        if not isinstance(op, HasFolderInterface):
            return None
        folded = op.fold()
        if folded is None:
            return None
        results: list[SSAValue] = []
        new_ops: list[Operation] = []
        for val, original_result in zip(folded, op.results):
            if isinstance(val, SSAValue):
                results.append(val)
            else:
                assert isinstance(val, Attribute)
                dialect = self.context.get_dialect(op.dialect_name())
                interface = dialect.get_interface(ConstantMaterializationInterface)
                if not interface:
                    return None
                assert isinstance(type := original_result.type, TypeAttribute)
                new_op = interface.materialize_constant(val, type)
                if new_op is None:
                    return None
                new_ops.append(new_op)
                results.append(new_op.results[0])
        return results, new_ops

    def insert_with_fold(
        self, op: Operation, builder: Builder
    ) -> Sequence[SSAValue] | None:
        """
        Inserts the operation using the provided builder, trying to fold it first.
        If folding is successful, the folded results are returned, otherwise None is returned.
        """
        results = self.try_fold(op)
        if results is not None:
            values, new_ops = results
            builder.insert_op(new_ops)
            return values
        else:
            builder.insert(op)
            return op.results

    def replace_with_fold(
        self, op: Operation, safe_erase: bool = True
    ) -> Sequence[SSAValue] | None:
        """
        Replaces the operation with its folded results.
        If folding is successful, the folded results are returned.
        Otherwise, returns None.
        """
        results = self.try_fold(op)
        if results is None:
            return None
        values, new_ops = results
        Rewriter().replace_op(op, new_ops, values, safe_erase)
