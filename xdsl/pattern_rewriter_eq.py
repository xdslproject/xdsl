from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.builder import InsertOpInvT
from xdsl.dialects import equivalence
from xdsl.eqsat_bookkeeper import Eqsat_Bookkeeper
from xdsl.ir import Operation, SSAValue
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint


@dataclass(eq=False, init=False)
class EquivalencePatternRewriter(PatternRewriter):
    eqsat_bookkeeping: Eqsat_Bookkeeper

    def __init__(self, current_operation: Operation):
        super().__init__(current_operation)
        self.eqsat_bookkeeping = Eqsat_Bookkeeper()
        self.eqsat_bookkeeping.populate_known_ops(current_operation)

    def insert_op(
        self,
        op: InsertOpInvT,
        insertion_point: InsertPoint | None = None,
    ) -> InsertOpInvT:
        """Insert operations at a certain location in a block."""
        ops = [op] if isinstance(op, Operation) else op

        for o in ops:
            # before we just insert this operation, we need to check for each of its operands,
            # if it already has an e-class, use that one if it exists
            if not isinstance(o, equivalence.AnyClassOp):
                for i, operand in enumerate(tuple(o.operands)):
                    new_operand = self.eqsat_bookkeeping.run_get_class_result(operand)
                    if new_operand is not None and new_operand != operand:
                        o.operands[i] = new_operand

            if o not in self.eqsat_bookkeeping.known_ops:
                self.eqsat_bookkeeping.known_ops[o] = o
                super().insert_op(o, insertion_point)
            else:  # o is already known, do not insert it
                for old, new in zip(
                    o.results, self.eqsat_bookkeeping.known_ops[o].results
                ):
                    new = self.eqsat_bookkeeping.run_get_class_result(new)
                    if old != new:
                        super().replace_all_uses_with(old, new)

        if isinstance(op, Operation):
            return self.eqsat_bookkeeping.known_ops[op]
        return [self.eqsat_bookkeeping.known_ops[op] for op in ops]  # pyright: ignore[reportReturnType]

    def replace_op(
        self,
        op: Operation,
        new_ops: Operation | Sequence[Operation],
        new_results: Sequence[SSAValue | None] | None = None,
        safe_erase: bool = True,
    ):
        """
        Replace an operation with new operations.
        Also, optionally specify SSA values to replace the operation results.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True

        # First, insert the new operations before the matched operation
        self.insert_op(new_ops, InsertPoint.before(op))

        if isinstance(new_ops, Operation):
            new_ops = (new_ops,)

        # If new results are not specified, use the results of the last new operation by default
        if new_results is None or len(new_results) == 0:
            new_results = new_ops[-1].results if new_ops else ()

        if len(op.results) != len(new_results):
            raise ValueError(
                f"Expected {len(op.results)} new results, but got {len(new_results)}"
            )

        # instead of erasing the old operation,
        # Union the old results with the new results by inserting an e-class operation
        for old_result, new_result in zip(op.results, new_results):
            if new_result is not None:
                self.eqsat_bookkeeping.union_val(self, old_result, new_result)
                # this already replaces every later use of old results with the new eclass result
                # in union_val -> get_or_create_class -> replace_uses_with_if
