from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ordered_set import OrderedSet

from xdsl.analysis.sparse_analysis import Lattice, SparseForwardDataFlowAnalysis
from xdsl.dialects import ematch, equivalence
from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Operation, OpResult, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa


@register_impls
@dataclass
class EmatchFunctions(InterpreterFunctions):
    """Interpreter functions for PDL patterns operating on e-graphs."""

    known_ops: KnownOps = field(default_factory=KnownOps)
    """Used for hashconsing operations. When new operations are created, if they are identical to an existing operation,
    the existing operation is reused instead of creating a new one."""

    eclass_union_find: DisjointSet[equivalence.AnyClassOp] = field(
        default_factory=lambda: DisjointSet[equivalence.AnyClassOp]()
    )
    """Union-find structure tracking which e-classes are equivalent and should be merged."""

    pending_rewrites: list[tuple[SymbolRefAttr, Operation, tuple[Any, ...]]] = field(
        default_factory=lambda: []
    )
    """List of pending rewrites to be executed. Each entry is a tuple of (rewriter, root, args)."""

    worklist: list[equivalence.AnyClassOp] = field(
        default_factory=list[equivalence.AnyClassOp]
    )
    """Worklist of e-classes that need to be processed for matching."""

    is_matching: bool = True
    """Keeps track whether the interpreter is currently in a matching context (as opposed to in a rewriting context).
    If it is, finalize behaves differently by backtracking."""

    analyses: list[SparseForwardDataFlowAnalysis[Lattice[Any]]] = field(
        default_factory=lambda: []
    )
    """The sparse forward analyses to be run during equality saturation.
    These must be registered with a NonPropagatingDataFlowSolver where `propagate` is False.
    This way, state propagation is handled purely by the equality saturation logic.
    """

    def modification_handler(self, op: Operation):
        """
        Keeps `known_ops` up to date.
        Whenever an operation is modified, for example when its operands are updated to a different eclass value,
        the operation is added to the hashcons `known_ops`.
        """
        if op not in self.known_ops:
            self.known_ops[op] = op

    def populate_known_ops(self, outer_op: Operation) -> None:
        """
        Populates the known_ops dictionary by traversing the module.

        Args:
            outer_op: The operation containing all operations to be added to known_ops.
        """
        # Walk through all operations in the module
        for op in outer_op.walk():
            # Skip eclasses instances
            if not isinstance(op, equivalence.AnyClassOp):
                self.known_ops[op] = op
            else:
                self.eclass_union_find.add(op)

    @impl(ematch.GetClassValsOp)
    def run_get_class_vals(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassValsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Take a value and return all values in its equivalence class.

        If the value is an equivalence.class result, return the operands of the class,
        otherwise return a tuple containing just the value itself.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return ((val,),)

        assert isinstance(val, SSAValue)

        if isinstance(val, OpResult):
            defining_op = val.owner
            if isinstance(defining_op, equivalence.AnyClassOp):
                return (tuple(defining_op.operands),)

        # Value is not an eclass result, return it as a single-element tuple
        return ((val,),)

    @impl(ematch.GetClassRepresentativeOp)
    def run_get_class_representative(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassRepresentativeOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get one of the values in the equivalence class of v.
        Returns the first operand of the equivalence class.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return (val,)

        assert isa(val, SSAValue)

        if isinstance(val, OpResult):
            defining_op = val.owner
            if isinstance(defining_op, equivalence.AnyClassOp):
                leader = self.eclass_union_find.find(defining_op)
                return (leader.operands[0],)

        # Value is not an eclass result, return it as-is
        return (val,)

    @impl(ematch.GetClassResultOp)
    def run_get_class_result(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassResultOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get the equivalence.class result corresponding to the equivalence class of v.

        If v has exactly one use and that use is a ClassOp, return the ClassOp's result.
        Otherwise return v unchanged.
        """
        assert len(args) == 1
        val = args[0]

        if val is None:
            return (val,)

        assert isa(val, SSAValue)

        if val.has_one_use():
            user = val.get_user_of_unique_use()
            if isinstance(user, equivalence.AnyClassOp):
                leader = self.eclass_union_find.find(user)
                return (leader.result,)

        return (val,)

    @impl(ematch.GetClassResultsOp)
    def run_get_class_results(
        self,
        interpreter: Interpreter,
        op: ematch.GetClassResultsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Get the equivalence.class results corresponding to the equivalence classes
        of a range of values.
        """
        assert len(args) == 1
        vals = args[0]

        if vals is None:
            return ((),)

        results: list[SSAValue] = []
        for val in vals:
            if val is None:
                results.append(val)
            elif val.has_one_use():
                user = val.get_user_of_unique_use()
                if isinstance(user, equivalence.AnyClassOp):
                    leader = self.eclass_union_find.find(user)
                    results.append(leader.result)
                else:
                    results.append(val)
            else:
                results.append(val)

        return (tuple(results),)

    def get_or_create_class(
        self, interpreter: Interpreter, val: SSAValue
    ) -> equivalence.AnyClassOp:
        """
        Get the equivalence class for a value, creating one if it doesn't exist.
        """
        if isinstance(val, OpResult):
            # If val is defined by a ClassOp, return it
            if isinstance(val.owner, equivalence.AnyClassOp):
                return self.eclass_union_find.find(val.owner)
            insertpoint = InsertPoint.before(val.owner)
        else:
            assert isinstance(val.owner, Block)
            insertpoint = InsertPoint.at_start(val.owner)

        # If val has one use and it's a ClassOp, return it
        if (user := val.get_user_of_unique_use()) is not None:
            if isinstance(user, equivalence.AnyClassOp):
                return user

        # If the value is not part of an eclass yet, create one
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)

        eclass_op = equivalence.ClassOp(val)
        rewriter.insert_op(eclass_op, insertpoint)
        self.eclass_union_find.add(eclass_op)

        # Replace uses of val with the eclass result (except in the eclass itself)
        rewriter.replace_uses_with_if(
            val, eclass_op.result, lambda use: use.operation is not eclass_op
        )

        return eclass_op

    def eclass_union(
        self,
        interpreter: Interpreter,
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

        rewriter = PDLInterpFunctions.get_rewriter(interpreter)
        rewriter.replace_op(to_replace, new_ops=[], new_results=to_keep.results)
        return True

    def union_val(self, interpreter: Interpreter, a: SSAValue, b: SSAValue) -> None:
        """
        Union two values into the same equivalence class.
        """
        if a == b:
            return

        eclass_a = self.get_or_create_class(interpreter, a)
        eclass_b = self.get_or_create_class(interpreter, b)

        if self.eclass_union(interpreter, eclass_a, eclass_b):
            self.worklist.append(eclass_a)

    @impl(ematch.UnionOp)
    def run_union(
        self,
        interpreter: Interpreter,
        op: ematch.UnionOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Merge two values, an operation and a value range, or two value ranges
        into equivalence class(es).

        Supported operand type combinations:
        - (value, value): merge two values
        - (operation, range<value>): merge operation results with values
        - (range<value>, range<value>): merge two value ranges
        """
        assert len(args) == 2
        lhs, rhs = args

        if isa(lhs, SSAValue) and isa(rhs, SSAValue):
            # (Value, Value) case
            self.union_val(interpreter, lhs, rhs)

        elif isinstance(lhs, Operation) and isa(rhs, Sequence[SSAValue]):
            # (Operation, ValueRange) case
            assert len(lhs.results) == len(rhs), (
                "Operation result count must match value range size"
            )
            for result, val in zip(lhs.results, rhs, strict=True):
                self.union_val(interpreter, result, val)

        elif isa(lhs, Sequence[SSAValue]) and isa(rhs, Sequence[SSAValue]):
            # (ValueRange, ValueRange) case
            assert len(lhs) == len(rhs), "Value ranges must have equal size"
            for val_lhs, val_rhs in zip(lhs, rhs, strict=True):
                self.union_val(interpreter, val_lhs, val_rhs)

        else:
            raise InterpretationError(
                f"union: unsupported argument types: {type(lhs)}, {type(rhs)}"
            )

        return ()

    @impl(ematch.DedupOp)
    def run_dedup(
        self,
        interpreter: Interpreter,
        op: ematch.DedupOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """
        Check if the operation already exists in the hashcons.

        If an equivalent operation exists, erase the input operation and return
        the existing one. Otherwise, insert the operation into the hashcons and
        return it.
        """
        assert len(args) == 1
        input_op = args[0]
        assert isinstance(input_op, Operation)

        # Check if an equivalent operation exists in hashcons
        existing = self.known_ops.get(input_op)

        if existing is not None and existing is not input_op:
            # Deduplicate: erase the new op and return existing
            rewriter = PDLInterpFunctions.get_rewriter(interpreter)
            rewriter.erase_op(input_op)
            return (existing,)

        # No duplicate found, insert into hashcons
        self.known_ops[input_op] = input_op
        return (input_op,)
