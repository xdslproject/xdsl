from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ordered_set import OrderedSet

from xdsl.builder import Builder, InsertOpInvT
from xdsl.dialects import equivalence
from xdsl.ir import (
    Block,
    Operation,
    OpResult,
    SSAValue,
)
from xdsl.pattern_rewriter import PatternRewriterListener
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet


@dataclass(eq=False, init=False)
class EquivalencePatternRewriter(Builder, PatternRewriterListener):
    """
    A rewriter used during pattern matching.
    Once an operation is matched, this rewriter is used to apply
    modification to the operation and its children.
    """

    # operations that already have an eclass
    known_ops: KnownOps = field(default_factory=KnownOps)
    """Used for hashconsing operations. When new operations are created, if they are identical to an existing operation,
    the existing operation is reused instead of creating a new one."""

    eclass_union_find: DisjointSet[equivalence.AnyClassOp] = field(
        default_factory=lambda: DisjointSet[equivalence.AnyClassOp]()
    )
    """Union-find structure tracking which e-classes are equivalent and should be merged."""

    worklist: list[equivalence.AnyClassOp] = field(
        default_factory=list[equivalence.AnyClassOp]
    )
    """Worklist of e-classes that need to be processed for matching."""

    current_operation: Operation
    """The matched operation."""

    has_done_action: bool = field(default=False, init=False)
    """Has the rewriter done any action during the current match."""

    def __init__(self, current_operation: Operation):
        PatternRewriterListener.__init__(self)
        self.current_operation = current_operation
        Builder.__init__(self, InsertPoint.before(current_operation))

    # change this to insert eq class & deduplicate operation
    # check if an eclass for the operation already exists, if so resues it, if not create one
    def insert_op(
        self,
        op: InsertOpInvT,
        insertion_point: InsertPoint | None = None,
    ) -> InsertOpInvT:
        """Insert operations at a certain location in a block."""
        if op in self.known_ops:
            return op

        # op not in known_ops
        self.known_ops.add(op)
        self.has_done_action = True
        return super().insert_op(op, insertion_point)

    # do i need to remove op from known_ops when op is erased, in case it is used nowhere else?
    def erase_op(self, op: Operation, safe_erase: bool = True):
        """
        Erase an operation.
        If safe_erase is True, check that the operation has no uses.
        Otherwise, replace its uses with ErasedSSAValue.
        """
        self.has_done_action = True
        self.handle_operation_removal(op)
        Rewriter.erase_op(op, safe_erase=safe_erase)

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

        if isinstance(new_ops, Operation):
            new_ops = (new_ops,)

        # First, insert the new operations before the matched operation
        self.insert_op(new_ops, InsertPoint.before(op))
        # If new results are not specified, use the results of the last new operation by default
        if new_results is None:
            new_results = new_ops[-1].results if new_ops else []

        if len(op.results) != len(new_results):
            raise ValueError(
                f"Expected {len(op.results)} new results, but got {len(new_results)}"
            )

        # Then, union the results with new ones
        self.handle_operation_replacement(op, new_results)
        for old_result, new_result in zip(op.results, new_results):
            self.union_val(old_result, new_result)

    def union_val(self, a: SSAValue, b: SSAValue) -> None:
        """
        Union two values into the same equivalence class.
        """
        if a == b:
            return

        eclass_a = self.get_or_create_class(a)
        eclass_b = self.get_or_create_class(b)

        if self.eclass_union(eclass_a, eclass_b):
            self.worklist.append(eclass_a)

    def get_or_create_class(self, val: SSAValue) -> equivalence.AnyClassOp:
        """
        Get the equivalence class for a value, creating one if it doesn't exist.
        """
        if isinstance(val, OpResult):
            # If val is defined by a ClassOp, return it
            if isinstance(val.owner, equivalence.AnyClassOp):
                return self.eclass_union_find.find(val.owner)
        else:
            assert isinstance(val.owner, Block)

        # If val has one use and it's a ClassOp, return it
        if (user := val.get_user_of_unique_use()) is not None:
            if isinstance(user, equivalence.AnyClassOp):
                return user

        # If the value is not part of an eclass yet, create one
        eclass_op = equivalence.ClassOp(val)
        self.eclass_union_find.add(eclass_op)

        # Do I need to replace all uses of val with the eclass result here?

        return eclass_op

    def eclass_union(
        self,
        a: equivalence.AnyClassOp,
        b: equivalence.AnyClassOp,
    ) -> bool:
        """Unions two eclasses, merging their operands and results.
        Returns True if the eclasses were merged, False if they were already the same."""
        a = self.eclass_union_find.find(a)
        b = self.eclass_union_find.find(b)

        if a == b:
            return False

        # Meet the analysis states of the two e-classes
        for analysis in self.analyses:
            a_lattice = analysis.get_lattice_element(a.result)
            b_lattice = analysis.get_lattice_element(b.result)
            a_lattice.meet(b_lattice)

        if isinstance(a, equivalence.ConstantClassOp):
            if isinstance(b, equivalence.ConstantClassOp):
                assert a.value == b.value, (
                    "Trying to union two different constant eclasses.",
                )
            to_keep, to_replace = a, b
            self.eclass_union_find.union_left(to_keep, to_replace)
        elif isinstance(b, equivalence.ConstantClassOp):
            to_keep, to_replace = b, a
            self.eclass_union_find.union_left(to_keep, to_replace)
        else:
            self.eclass_union_find.union(
                a,
                b,
            )
            to_keep = self.eclass_union_find.find(a)
            to_replace = b if to_keep is a else a
        # Operands need to be deduplicated because it can happen the same operand was
        # used by different parent eclasses after their children were merged:
        new_operands = OrderedSet(to_keep.operands)
        new_operands.update(to_replace.operands)
        to_keep.operands = new_operands

        for use in to_replace.result.uses:
            # uses are removed from the hashcons before the replacement is carried out.
            # (because the replacement changes the operations which means we cannot find them in the hashcons anymore)
            if use.operation in self.known_ops:
                self.known_ops.pop(use.operation)

        return True
