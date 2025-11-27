from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ordered_set import OrderedSet

from xdsl.analysis.dataflow import (
    AnalysisState,
    ChangeResult,
    DataFlowSolver,
    ProgramPoint,
)
from xdsl.analysis.sparse_analysis import Lattice, SparseForwardDataFlowAnalysis
from xdsl.dialects import eqsat, eqsat_pdl_interp, pdl_interp
from xdsl.dialects.builtin import SymbolRefAttr
from xdsl.dialects.pdl import RangeType, ValueType
from xdsl.interpreter import (
    Interpreter,
    ReturnedValues,
    Successor,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Block, Operation, OpResult, SSAValue
from xdsl.irdl import IRDLOperation
from xdsl.rewriter import InsertPoint
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.hints import isa
from xdsl.utils.scoped_dict import ScopedDict


@dataclass
class NonPropagatingDataFlowSolver(DataFlowSolver):
    propagate: bool = field(default=False)

    def propagate_if_changed(self, state: AnalysisState, changed: ChangeResult) -> None:
        if not self.propagate:
            return
        super().propagate_if_changed(state, changed)


@dataclass
class BacktrackPoint:
    """
    Represents a point in pattern matching where backtracking may be needed.

    When a GetDefiningOpOp encounters an EClassOp with multiple operands,
    we need to try matching against each operand. This class captures the
    interpreter state so we can backtrack and try the next operand if
    the current match fails.
    """

    block: Block
    """The block to return to when backtracking."""

    block_args: tuple[SSAValue, ...]
    """Block arguments to restore when backtracking."""

    scope: ScopedDict[SSAValue, Any]
    """Variable scope to restore when backtracking."""

    cause: pdl_interp.GetDefiningOpOp | eqsat_pdl_interp.ChooseOp
    """The GetDefiningOpOp or ChooseOp that created this backtrack point."""

    index: int
    """Current backtrack index being tried.
    When `cause` is a GetDefiningOpOp, this is the index of the operand of the EClassOp being tried.
    When `cause` is a ChooseOp, this is the index of the choice being tried.
    """

    max_index: int
    """Last valid index to backtrack to."""


@register_impls
@dataclass
class EqsatPDLInterpFunctions(PDLInterpFunctions):
    """Interpreter functions for PDL patterns operating on e-graphs."""

    analyses: list[SparseForwardDataFlowAnalysis[Lattice[Any]]] = field(
        default_factory=lambda: []
    )
    """The sparse forward analyses to be run during equality saturation.
    These must be registered with a NonPropagatingDataFlowSolver where `propagate` is False.
    This way, state propagation is handled purely by the equality saturation logic.
    """

    backtrack_stack: list[BacktrackPoint] = field(default_factory=list[BacktrackPoint])
    """Stack of backtrack points for exploring multiple matching paths in e-classes."""

    visited: bool = True
    """Signals whether we have processed the last backtrack point after a finalize.
    This is used to tell whether we are encountering a GetDefiningOpOp or ChooseOp
    for the first time (in which case we need to create a new backtrack point)
    or whether we are backtracking to it (in which case we need to use the existing
    backtrack point and continue from the stored index).

    When visited is False, the next GetDefiningOpOp or ChooseOp we encounter must be
    the one at the top of the backtrack stack and the index is incremented. Otherwise,
    the last finalize has already been handled and a new backtrack point must be created.
    """

    known_ops: KnownOps = field(default_factory=KnownOps)
    """Used for hashconsing operations. When new operations are created, if they are identical to an existing operation,
    the existing operation is reused instead of creating a new one."""

    eclass_union_find: DisjointSet[eqsat.AnyEClassOp] = field(
        default_factory=lambda: DisjointSet[eqsat.AnyEClassOp]()
    )
    """Union-find structure tracking which e-classes are equivalent and should be merged."""

    pending_rewrites: list[tuple[SymbolRefAttr, Operation, tuple[Any, ...]]] = field(
        default_factory=lambda: []
    )
    """List of pending rewrites to be executed. Each entry is a tuple of (rewriter, root, args)."""

    worklist: list[eqsat.AnyEClassOp] = field(default_factory=list[eqsat.AnyEClassOp])
    """Worklist of e-classes that need to be processed for matching."""

    is_matching: bool = True
    """Keeps track whether the interpreter is currently in a matching context (as opposed to in a rewriting context).
    If it is, finalize behaves differently by backtracking."""

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
            if not isinstance(op, eqsat.AnyEClassOp):
                self.known_ops[op] = op
            else:
                self.eclass_union_find.add(op)

    @impl(pdl_interp.GetResultOp)
    def run_get_result(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetResultOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)
        if len(args[0].results) <= op.index.value.data:
            result = None
        else:
            result = args[0].results[op.index.value.data]

        if result is None:
            return (None,)

        if result.has_one_use():
            if isinstance(
                eclass_op := result.get_user_of_unique_use(), eqsat.AnyEClassOp
            ):
                result = eclass_op.result
        else:
            for use in result.uses:
                if isinstance(use.operation, eqsat.AnyEClassOp):
                    raise InterpretationError(
                        "pdl_interp.get_result currently only supports operations with results"
                        " that are used by a single eclass each."
                    )
        return (result,)

    @impl(pdl_interp.GetResultsOp)
    def run_get_results(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetResultsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        assert isinstance(args[0], Operation)
        src_op = args[0]
        assert isinstance(src_op, IRDLOperation)
        if op.index is not None:
            # get the field name of the result group:
            if op.index.value.data >= len(src_op.get_irdl_definition().results):
                return (None,)
            field = src_op.get_irdl_definition().results[op.index.value.data][0]
            results = getattr(src_op, field)
            if isa(results, OpResult):
                results = [results]
        else:
            results = src_op.results

        if isinstance(op.result_types[0], ValueType) and len(results) != 1:
            return (None,)

        eclass_results: list[OpResult] = []
        for result in results:
            if not result.has_one_use():
                raise InterpretationError(
                    "pdl_interp.get_results only supports results"
                    " that are used by a single eclass each."
                )
            if not isinstance(
                eclass_op := result.get_user_of_unique_use(), eqsat.AnyEClassOp
            ):
                raise InterpretationError(
                    "pdl_interp.get_results only supports results"
                    " that are used by a single eclass each."
                )
            eclass_results.append(eclass_op.result)
        if isinstance(op.result_types[0], ValueType):
            return (eclass_results[0],)
        return (tuple(eclass_results),)

    @impl(pdl_interp.GetDefiningOpOp)
    def run_get_defining_op(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetDefiningOpOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert len(args) == 1
        if args[0] is None:
            return (None,)
        assert isinstance(args[0], SSAValue)
        if not isinstance(args[0], OpResult):
            return (None,)
        else:
            defining_op = args[0].owner

        if not isinstance(defining_op, eqsat.AnyEClassOp):
            return (defining_op,)

        eclass_op = defining_op
        if not self.visited:  # we come directly from run_finalize
            if op != self.backtrack_stack[-1].cause:
                # we first encounter a GDO that is not the one we are backtracking to:
                raise InterpretationError(
                    "Case where a block contains multiple pdl_interp.get_defining_op is currently not supported."
                )
            index = self.backtrack_stack[-1].index
            self.visited = True
        else:
            block = op.parent_block()
            assert block
            block_args = interpreter.get_values(block.args)
            scope = interpreter._ctx.parent  # pyright: ignore[reportPrivateUsage]
            assert scope
            index = 0
            self.backtrack_stack.append(
                BacktrackPoint(
                    block, block_args, scope, op, index, len(eclass_op.operands) - 1
                )
            )
        defining_op = eclass_op.operands[index].owner
        if not isinstance(defining_op, Operation):
            return (None,)

        return (defining_op,)

    @impl(pdl_interp.ReplaceOp)
    def run_replace(
        self,
        interpreter: Interpreter,
        op: pdl_interp.ReplaceOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert args
        input_op = args[0]
        assert isinstance(input_op, Operation)
        repl_values: list[SSAValue] = []
        for t, v in zip(op.repl_values.types, args[1:], strict=True):
            assert isa(t, ValueType | RangeType[ValueType])
            match t:
                case ValueType():
                    assert isa(v, SSAValue)
                    repl_values.append(v)
                case RangeType():
                    repl_values.extend(v)
        assert len(input_op.results) == len(repl_values)
        for res, repl in zip(input_op.results, repl_values):
            if not res.has_one_use():
                raise InterpretationError(
                    "Operation's result can only be used once, by an eclass operation."
                )
            assert res.first_use is not None
            if not isinstance(
                original_eclass := res.first_use.operation, eqsat.AnyEClassOp
            ):
                raise InterpretationError(
                    "Replaced operation result must be used by an eclass"
                )

            repl_eclass = repl.owner
            if not isinstance(repl_eclass, eqsat.AnyEClassOp):
                raise InterpretationError(
                    "Replacement value must be the result of an eclass"
                )

            if self.eclass_union(interpreter, original_eclass, repl_eclass):
                self.worklist.append(original_eclass)

        return ()

    def eclass_union(
        self, interpreter: Interpreter, a: eqsat.AnyEClassOp, b: eqsat.AnyEClassOp
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

        if isinstance(a, eqsat.ConstantEClassOp):
            if isinstance(b, eqsat.ConstantEClassOp):
                assert a.value == b.value, (
                    "Trying to union two different constant eclasses.",
                )
            to_keep, to_replace = a, b
            self.eclass_union_find.union_left(to_keep, to_replace)
        elif isinstance(b, eqsat.ConstantEClassOp):
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

    @impl(pdl_interp.CreateOperationOp)
    def run_create_operation(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateOperationOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)
        has_done_action_checkpoint = rewriter.has_done_action

        updated_operands: list[OpResult] = []
        for arg in args[0 : len(op.input_operands)]:
            assert isinstance(arg, OpResult), (
                "pdl_interp.create_operation currently only supports creating operations with operands that are eclass results."
            )
            assert isinstance(arg.owner, eqsat.AnyEClassOp), (
                "pdl_interp.create_operation currently only supports creating operations with operands that are eclass results."
            )
            updated_operands.append(self.eclass_union_find.find(arg.owner).result)
        args = (*updated_operands, *args[len(op.input_operands) :])
        (new_op,) = super().run_create_operation(interpreter, op, args).values
        assert isinstance(new_op, Operation)
        assert new_op.results, (
            "Creating operations without result values is not supported."
        )

        # Check if an identical operation already exists in our known_ops map
        if existing_op := self.known_ops.get(new_op):
            # CSE can have removed the existing operation, here we check if it is still in use:
            assert existing_op.results
            if existing_op.parent is new_op.parent:
                rewriter.erase_op(new_op)
                rewriter.has_done_action = has_done_action_checkpoint
                if not any(
                    isinstance(use.operation, eqsat.AnyEClassOp)
                    for use in existing_op.results[0].uses
                ):
                    # It is possible that the existing_op was stripped from its eclass when it merged with a constant eclass.
                    # In this case we should wrap it in a new eclass:
                    new_op = existing_op
                else:
                    return (existing_op,)
            else:
                # if CSE has removed the existing operation, we can remove it from our known_ops map:
                self.known_ops.pop(existing_op)

        # No existing eclass for this operation yet
        new_eclasses: list[eqsat.AnyEClassOp] = []
        if new_op.has_trait(eqsat.ConstantLike):
            assert len(new_op.results) == 1
            new_eclasses.append(
                eqsat.ConstantEClassOp(
                    new_op.results[0],
                )
            )
        else:
            for res in new_op.results:
                new_eclasses.append(
                    eqsat.EClassOp(
                        res,
                    )
                )

        for eclass_op in new_eclasses:
            for analysis in self.analyses:
                for x in (new_op, eclass_op):
                    point = ProgramPoint.before(x)
                    operands = [
                        analysis.get_lattice_element_for(point, o) for o in x.operands
                    ]
                    results = [analysis.get_lattice_element(r) for r in x.results]
                    assert len(results) == 1
                    analysis.visit_operation_impl(x, operands, results)

            rewriter.insert_op(
                eclass_op,
                InsertPoint.after(new_op),
            )
            self.eclass_union_find.add(eclass_op)

        self.known_ops[new_op] = new_op

        return (new_op,)

    @impl_terminator(pdl_interp.RecordMatchOp)
    def run_recordmatch(
        self,
        interpreter: Interpreter,
        op: pdl_interp.RecordMatchOp,
        args: tuple[Any, ...],
    ):
        self.pending_rewrites.append(
            (op.rewriter, self.get_rewriter(interpreter).current_operation, args)
        )
        return Successor(op.dest, ()), ()

    @impl_terminator(pdl_interp.FinalizeOp)
    def run_finalize(
        self, interpreter: Interpreter, _: pdl_interp.FinalizeOp, args: tuple[Any, ...]
    ):
        if not self.is_matching:
            return ReturnedValues(()), ()
        for backtrack_point in reversed(self.backtrack_stack):
            if backtrack_point.index >= backtrack_point.max_index:
                self.backtrack_stack.pop()
            else:
                backtrack_point.index += 1
                interpreter._ctx = (  # pyright: ignore[reportPrivateUsage]
                    backtrack_point.scope
                )
                self.visited = False
                return Successor(backtrack_point.block, backtrack_point.block_args), ()
        return ReturnedValues(()), ()

    @impl_terminator(eqsat_pdl_interp.ChooseOp)
    def run_choose(
        self,
        interpreter: Interpreter,
        op: eqsat_pdl_interp.ChooseOp,
        args: tuple[Any, ...],
    ):
        if not self.visited:  # we come directly from run_finalize
            if op != self.backtrack_stack[-1].cause:
                raise InterpretationError(
                    "Expected this ChooseOp to be at the top of the backtrack stack."
                )
            index = self.backtrack_stack[-1].index
            self.visited = True
        else:
            block = op.parent_block()
            assert block
            block_args = interpreter.get_values(block.args)
            scope = interpreter._ctx.parent  # pyright: ignore[reportPrivateUsage]
            assert scope
            index = 0
            self.backtrack_stack.append(
                BacktrackPoint(block, block_args, scope, op, index, len(op.choices))
            )
        if index == len(op.choices):
            dest = op.default_dest
        else:
            dest = op.choices[index]
        return Successor(dest, ()), ()

    def repair(self, interpreter: Interpreter, eclass: eqsat.AnyEClassOp):
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)
        unique_parents = KnownOps()
        eclass = self.eclass_union_find.find(eclass)
        for op1 in OrderedSet(use.operation for use in eclass.result.uses):
            if op1 in unique_parents:
                # This means another parent that was processed before is identical to this one,
                # the corresponding eclasses need to be merged.
                op2 = unique_parents[op1]

                assert (op1_use := op1.results[0].first_use), (
                    "Modification handler currently only supports operations with a single (EClassOp) use"
                )
                assert isinstance(eclass1 := op1_use.operation, eqsat.AnyEClassOp)

                assert len(op2.results) == 1, (
                    "Expected a single result for the operation being modified."
                )
                assert (op2_use := op2.results[0].first_use), (
                    "Modification handler currently only supports operations with a single (EClassOp) use"
                )
                assert isinstance(eclass2 := op2_use.operation, eqsat.AnyEClassOp)

                # This temporarily breaks the invariant since eclass2 will now contain the result of op2 twice.
                # Callling `eclass_union` will deduplicate this operand.
                rewriter.replace_op(op1, new_ops=(), new_results=op2.results)

                if eclass1 == eclass2:
                    eclass1.operands = OrderedSet(
                        eclass1.operands
                    )  # deduplicate operands
                    continue  # parents need not be processed because no eclasses were merged

                assert self.eclass_union(interpreter, eclass1, eclass2), (
                    "Expected eclasses to not already be unioned."
                )

                self.worklist.append(eclass1)
            else:
                unique_parents[op1] = op1

        eclass = self.eclass_union_find.find(eclass)
        for op in OrderedSet(use.operation for use in eclass.result.uses):
            point = ProgramPoint.before(op)
            # for every analysis,
            for analysis in self.analyses:
                operands = [
                    analysis.get_lattice_element_for(point, o) for o in op.operands
                ]
                results = [analysis.get_lattice_element(r) for r in op.results]
                if not results:
                    continue
                (result,) = results
                # set state for op to bottom (store original state)
                original_state = result.value
                result._value = result.value_cls()  # pyright: ignore[reportPrivateUsage]
                analysis.visit_operation_impl(op, operands, results)

                changed = result.meet(type(result)(result.anchor, original_state))
                if changed == ChangeResult.CHANGE:
                    assert (op_use := op.results[0].first_use), (
                        "Dataflow analysis currently only supports operations with a single (EClassOp) use"
                    )
                    assert isinstance(eclass_op := op_use.operation, eqsat.AnyEClassOp)
                    self.worklist.append(eclass_op)

    def rebuild(self, interpreter: Interpreter):
        while self.worklist:
            todo = OrderedSet(self.eclass_union_find.find(c) for c in self.worklist)
            self.worklist.clear()
            for c in todo:
                self.repair(interpreter, c)

    def execute_pending_rewrites(self, interpreter: Interpreter):
        """Execute all pending rewrites that were aggregated during matching."""
        rewriter = PDLInterpFunctions.get_rewriter(interpreter)
        for rewriter_op, root, args in self.pending_rewrites:
            rewriter.current_operation = root
            rewriter.insertion_point = InsertPoint.before(root)

            self.is_matching = False
            interpreter.call_op(rewriter_op, args)
            self.is_matching = True
        self.pending_rewrites.clear()
