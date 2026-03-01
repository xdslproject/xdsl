from dataclasses import dataclass, field
from typing import Any

from xdsl.analysis.sparse_analysis import Lattice, SparseForwardDataFlowAnalysis
from xdsl.dialects import ematch, equivalence
from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Operation, OpResult, SSAValue
from xdsl.rewriter import InsertPoint
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet
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
