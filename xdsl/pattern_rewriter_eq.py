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

        # Only perform hash-consing for single operations, not sequences
        if isinstance(op, Operation):
            if op in self.eqsat_bookkeeping.known_ops:
                return self.eqsat_bookkeeping.known_ops[op]  # type: ignore

            # op not in known_ops
            self.eqsat_bookkeeping.known_ops[op] = op

            return super().insert_op(op, insertion_point)

        if op == []:
            return op  # If op is an empty sequence, do nothing

        raise NotImplementedError(
            "Inserting a sequence of operations is not supported in EquivalencePatternRewriter yet."
        )
        # op is of type Sequence[Operation] -> still need to work on this
        # for o in op:
        #    if o not in self.eqsat_bookkeeping.known_ops:
        #        self.eqsat_bookkeeping.known_ops[o] = o
        #        super().insert_op(o, insertion_point)
        #    # if o is already known ignore it

        # return op
        return super().insert_op(op, insertion_point)  # uncomment this later

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

        if isinstance(op, equivalence.AnyClassOp):
            # if the old operator is itself an e-class, we want to erase this eclass and replace it with a merged one.
            # this is called in eclass_union so new_ops is already the merged eclass
            super().replace_op(op, new_ops, new_results, safe_erase)
            return

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
