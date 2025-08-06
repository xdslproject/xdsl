from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from ordered_set import OrderedSet

from xdsl.dialects import eqsat, pdl_interp
from xdsl.dialects.builtin import ModuleOp, SymbolRefAttr
from xdsl.dialects.pdl import ValueType
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
from xdsl.rewriter import InsertPoint
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.scoped_dict import ScopedDict

# Set up logging for the scheduler
logger = logging.getLogger(__name__)


class RuleScheduler(Protocol):
    """Protocol defining the interface for rule schedulers."""

    rule_stats: dict[pdl_interp.FuncOp, RuleStats]
    current_iteration_matches: dict[pdl_interp.FuncOp, int]

    def is_rule_banned(self, rule: pdl_interp.FuncOp) -> bool:
        """Check if a rule is currently banned."""
        ...

    def are_all_rules_banned(self, rules: list[pdl_interp.FuncOp]) -> bool:
        """Check if all rules in the list are banned."""
        ...

    def record_rule_match(self, rule: pdl_interp.FuncOp, match_count: int = 1) -> None:
        """Record that a rule matched in this iteration."""
        ...

    def end_iteration(self) -> None:
        """End the current iteration and apply scheduling logic."""
        ...

    def next_iteration(self) -> None:
        """Move to the next iteration."""
        ...

    def can_stop(self) -> bool:
        """Check if we can stop (scheduler-specific logic)."""
        ...


@dataclass
class RuleStats:
    """Statistics for a single rule in the scheduler."""

    times_applied: int = 0
    banned_until: int = 0
    times_banned: int = 0
    match_limit: int = 1000
    ban_length: int = 5
    total_matches: int = 0  # Total number of matches seen
    last_match_count: int = 0  # Number of matches in the last application


@dataclass
class BackoffScheduler:
    """
    Exponential backoff scheduler for equality saturation rules.

    Similar to egg's BackoffScheduler, this implements exponential rule backoff.
    For each rewrite, there exists a configurable initial match limit.
    If a rewrite search yields more than this limit, then we ban this
    rule for a number of iterations, double its limit, and double the time
    it will be banned next time.
    """

    default_match_limit: int = 1000
    default_ban_length: int = 5
    current_iteration: int = 0
    rule_stats: dict[pdl_interp.FuncOp, RuleStats] = field(default_factory=lambda: {})
    current_iteration_matches: dict[pdl_interp.FuncOp, int] = field(
        default_factory=lambda: {}
    )

    def get_rule_stats(self, rule: pdl_interp.FuncOp) -> RuleStats:
        """Get or create rule statistics for a rule."""
        if rule not in self.rule_stats:
            self.rule_stats[rule] = RuleStats(
                match_limit=self.default_match_limit, ban_length=self.default_ban_length
            )
        return self.rule_stats[rule]

    def is_rule_banned(self, rule: pdl_interp.FuncOp) -> bool:
        """Check if a rule is currently banned."""
        stats = self.get_rule_stats(rule)
        return self.current_iteration < stats.banned_until

    def are_all_rules_banned(self, rules: list[pdl_interp.FuncOp]) -> bool:
        """Check if all rules in the list are banned."""
        return all(self.is_rule_banned(rule) for rule in rules)

    def with_initial_match_limit(self, limit: int) -> BackoffScheduler:
        """Set the initial match limit after which a rule will be banned. Default: 1000"""
        self.default_match_limit = limit
        return self

    def with_ban_length(self, ban_length: int) -> BackoffScheduler:
        """Set the initial ban length. Default: 5 iterations"""
        self.default_ban_length = ban_length
        return self

    def rule_match_limit(self, rule: pdl_interp.FuncOp, limit: int) -> BackoffScheduler:
        """Set the initial match limit for a specific rule."""
        stats = self.get_rule_stats(rule)
        stats.match_limit = limit
        return self

    def rule_ban_length(self, rule: pdl_interp.FuncOp, length: int) -> BackoffScheduler:
        """Set the initial ban length for a specific rule."""
        stats = self.get_rule_stats(rule)
        stats.ban_length = length
        return self

    def record_rule_match(self, rule: pdl_interp.FuncOp, match_count: int = 1) -> None:
        """Record that a rule matched in this iteration."""
        # Don't count matches for banned rules
        if self.is_rule_banned(rule):
            return

        # Accumulate matches for this iteration
        if rule not in self.current_iteration_matches:
            self.current_iteration_matches[rule] = 0
        self.current_iteration_matches[rule] += match_count

        # Update total matches immediately for statistics
        stats = self.get_rule_stats(rule)
        stats.total_matches += match_count

    def end_iteration(self) -> None:
        """End the current iteration and apply scheduling logic."""
        # Apply scheduling logic for rules that matched in this iteration
        for rule_name, match_count in self.current_iteration_matches.items():
            stats = self.get_rule_stats(rule_name)

            # Calculate current threshold with exponential backoff
            threshold = stats.match_limit * (2**stats.times_banned)

            # Check if we should ban this rule
            if match_count > threshold:
                # Calculate new ban length with exponential backoff
                ban_length = stats.ban_length * (2**stats.times_banned)

                stats.times_banned += 1
                stats.banned_until = self.current_iteration + ban_length + 1

                # Log the ban with detailed information
                logger.info(
                    f"Banning rule '{rule_name}' for {ban_length} iterations: "
                    f"{match_count} matches > {threshold} threshold "
                    f"(ban #{stats.times_banned})"
                )
            else:
                stats.times_applied += 1

            # Record the match count for this iteration
            stats.last_match_count = match_count

        # Clear matches for next iteration
        self.current_iteration_matches.clear()

    def next_iteration(self) -> None:
        """Move to the next iteration."""
        self.current_iteration += 1

        # Check for rules that are being unbanned
        unbanned_rules: list[pdl_interp.FuncOp] = []
        for rule, stats in self.rule_stats.items():
            if stats.banned_until == self.current_iteration:
                unbanned_rules.append(rule)

        if unbanned_rules:
            logger.info(
                f"Unbanning rules at iteration {self.current_iteration}: {', '.join(str(rule) for rule in unbanned_rules)}"
            )

    def can_stop(self) -> bool:
        """Check if we can stop (no rules are banned)."""
        return not any(
            self.is_rule_banned(rule_name) for rule_name in self.rule_stats.keys()
        )


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

    gdo_op: pdl_interp.GetDefiningOpOp
    """The GetDefiningOpOp that created this backtrack point."""

    index: int
    """Current operand index being tried in the EClassOp."""

    max_index: int
    """Last valid operand index in the EClassOp (len(operands) - 1)."""

    reachable_rules: ScopedDict[pdl_interp.FuncOp, bool]
    """Set of reachable rules to restore at this point in the interpreter."""

    num_reachable_rules: int
    """Number of reachable rules at this point in the interpreter."""


@register_impls
@dataclass
class EqsatPDLInterpFunctions(PDLInterpFunctions):
    """Interpreter functions for PDL patterns operating on e-graphs."""

    backtrack_stack: list[BacktrackPoint] = field(default_factory=list[BacktrackPoint])
    """Stack of backtrack points for exploring multiple matching paths in e-classes."""

    visited: bool = True
    """Signals whether the GetDefiningOp (GDO) in the block that run_finalize jumps to has been encountered.

    This is to handle when a block contains GDOs before the GDO we're backtracking to.
    If this is the case, run_finalize jumps to the block but will encounter the wrong GDOs first.
    e.g.:
    ```
    BB0:
        %0 = pdl_interp.get_defining_op ...
        %1 = pdl_interp.get_defining_op ...
    BB1:
        pdl_interp.finalize # backtrack to BB0 at this point
    ```
    Backtracking works by jumping to the start of the block containing the GDO (`BB0`).
    When we need to backtrack to the second GDO (`%1`), `visited` is still `False` when encountering the first GDO (`%0`).
    This allows us to know that we have to skip the first GDO and continue with the second one.
    """

    known_ops: KnownOps = field(default_factory=KnownOps)
    """Used for hashconsing operations. When new operations are created, if they are identical to an existing operation,
    the existing operation is reused instead of creating a new one."""

    eclass_union_find: DisjointSet[eqsat.EClassOp] = field(
        default_factory=lambda: DisjointSet[eqsat.EClassOp]()
    )
    """Union-find structure tracking which e-classes are equivalent and should be merged."""

    pending_rewrites: list[tuple[SymbolRefAttr, Operation, tuple[Any, ...]]] = field(
        default_factory=lambda: []
    )
    """List of pending rewrites to be executed. Each entry is a tuple of (rewriter, root, args)."""

    worklist: list[eqsat.EClassOp] = field(default_factory=list[eqsat.EClassOp])
    """Worklist of e-classes that need to be processed for matching."""

    is_matching: bool = True
    """Keeps track whether the interpreter is currently in a matching context (as opposed to in a rewriting context).
    If it is, finalize behaves differently by backtracking."""

    reachable_rules: ScopedDict[pdl_interp.FuncOp, bool] = field(
        default_factory=lambda: ScopedDict[pdl_interp.FuncOp, bool](can_overwrite=True)
    )
    """Set of reachable rules in the current scope."""

    num_reachable_rules: int = 0
    """Number of reachable rules at this point in the interpreter."""

    def initialize_reachable_rules(self, rewriters_module: ModuleOp) -> None:
        for op in rewriters_module.regions[0].block.ops:
            if isinstance(op, pdl_interp.FuncOp):
                self.reachable_rules[op] = True

    scheduler: RuleScheduler = field(default_factory=BackoffScheduler)
    """Rule scheduler implementing exponential backoff for rule management."""

    def next_iteration(self) -> None:
        """Advance the scheduler to the next iteration."""
        self.scheduler.end_iteration()
        self.scheduler.next_iteration()

    def can_stop_due_to_scheduler(self) -> bool:
        """Check if we can stop based on scheduler state."""
        return self.scheduler.can_stop()

    def modification_handler(self, op: Operation):
        """
        Keeps `known_ops` up to date.
        Whenever an operation is modified, for example when its operands are updated to a different eclass value,
        the operation is added to the hashcons `known_ops`.
        """
        if op not in self.known_ops:
            self.known_ops[op] = op

    def populate_known_ops(self, module: ModuleOp) -> None:
        """
        Populates the known_ops dictionary by traversing the module.

        Args:
            module: The module to traverse
        """
        # Walk through all operations in the module
        for op in module.walk():
            # Skip EClassOp instances
            if not isinstance(op, eqsat.EClassOp):
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
            if isinstance(eclass_op := result.get_user_of_unique_use(), eqsat.EClassOp):
                result = eclass_op.result
        else:
            for use in result.uses:
                if isinstance(use.operation, eqsat.EClassOp):
                    raise InterpretationError(
                        "pdl_interp.get_result currently only supports operations with results"
                        " that are used by a single EClassOp each."
                    )
        return (result,)

    @impl(pdl_interp.GetResultsOp)
    def run_get_results(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetResultsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        assert isinstance(args[0], Operation)
        src_op = args[0]
        assert op.index is None, (
            "pdl_interp.get_results with index is not yet supported."
        )
        if isinstance(op.result_types[0], ValueType) and len(src_op.results) != 1:
            return (None,)

        results: list[OpResult] = []
        for result in src_op.results:
            if result.has_one_use():
                if isinstance(
                    eclass_op := result.get_user_of_unique_use(), eqsat.EClassOp
                ):
                    assert len(eclass_op.results) == 1
                    result = eclass_op.results[0]
            else:
                for use in result.uses:
                    if isinstance(use.operation, eqsat.EClassOp):
                        raise InterpretationError(
                            "pdl_interp.get_results currently only supports operations with results"
                            " that are used by a single EClassOp each."
                        )
            results.append(result)
        return (results,)

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

        if not isinstance(defining_op, eqsat.EClassOp):
            return (defining_op,)

        eclass_op = defining_op
        if not self.visited:  # we come directly from run_finalize
            if op != self.backtrack_stack[-1].gdo_op:
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
                    block,
                    block_args,
                    scope,
                    op,
                    index,
                    len(eclass_op.operands) - 1,
                    self.reachable_rules,
                    self.num_reachable_rules,
                )
            )
            self.reachable_rules = ScopedDict(
                parent=self.reachable_rules, can_overwrite=True
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
        assert len(input_op.results) == 1, (
            "ReplaceOp currently only supports replacing operations that have a single result"
        )

        it = iter(input_op.results[0].uses)
        original_eclass = next(it).operation
        if not isinstance(original_eclass, eqsat.EClassOp):
            raise InterpretationError(
                "Replaced operation result must be used by an EClassOp"
            )

        repl_values = (
            (args[1],) if isinstance(op.repl_values.types[0], ValueType) else args[1]
        )
        assert len(repl_values) == 1, (
            "pdl_interp.replace currently only a supports replacing with a single e-class result."
        )
        repl_value: SSAValue = repl_values[0]
        repl_eclass = repl_value.owner
        if not isinstance(repl_eclass, eqsat.EClassOp):
            raise InterpretationError(
                "Replacement value must be the result of an EClassOp"
            )

        if self.eclass_union(original_eclass, repl_eclass):
            self.worklist.append(original_eclass)

        return ()

    def eclass_union(self, a: eqsat.EClassOp, b: eqsat.EClassOp) -> bool:
        """Unions two e-classes, merging their operands and results.
        Returns True if the e-classes were merged, False if they were already the same."""
        a = self.eclass_union_find.find(a)
        b = self.eclass_union_find.find(b)

        if a == b:
            return False

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

        self.rewriter.replace_op(to_replace, new_ops=[], new_results=to_keep.results)
        return True

    @impl(pdl_interp.CreateOperationOp)
    def run_create_operation(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateOperationOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        has_done_action_checkpoint = self.rewriter.has_done_action

        updated_operands: list[OpResult] = []
        for arg in args[0 : len(op.input_operands)]:
            assert isinstance(arg, OpResult), (
                "pdl_interp.create_operation currently only supports creating operations with operands that are OpResult."
            )
            assert isinstance(arg.owner, eqsat.EClassOp), (
                "pdl_interp.create_operation currently only supports creating operations with operands that are EClassOp results."
            )
            updated_operands.append(self.eclass_union_find.find(arg.owner).result)
        args = (*updated_operands, *args[len(op.input_operands) :])
        (new_op,) = super().run_create_operation(interpreter, op, args).values

        assert isinstance(new_op, Operation)

        # Check if an identical operation already exists in our known_ops map
        if existing_op := self.known_ops.get(new_op):
            # CSE can have removed the existing operation, here we check if it is still in use:
            if existing_op.results and existing_op.results[0].first_use is not None:
                self.rewriter.erase_op(new_op)
                self.rewriter.has_done_action = has_done_action_checkpoint
                return (existing_op,)
            else:
                # if CSE has removed the existing operation, we can remove it from our known_ops map:
                self.known_ops.pop(existing_op)

        # No existing eclass for this operation yet

        eclass_op = eqsat.EClassOp(
            new_op.results[0],
        )
        self.rewriter.insert_op(
            eclass_op,
            InsertPoint.after(new_op),
        )

        self.known_ops[new_op] = new_op
        self.eclass_union_find.add(eclass_op)

        return (new_op,)

    @impl(eqsat.MarkUnreachableOp)
    def run_mark_unreachable(
        self,
        interpreter: Interpreter,
        op: eqsat.MarkUnreachableOp,
        args: tuple[Any, ...],
    ):
        for rule in op.rules.data:
            rule_op = interpreter.get_op_for_symbol(rule)
            assert isinstance(rule_op, pdl_interp.FuncOp), (
                "Expected a pdl_interp.FuncOp as rule."
            )
            self.reachable_rules[rule_op] = False
        return ()

    @impl(eqsat.MarkReachableOp)
    def run_mark_reachable(
        self,
        interpreter: Interpreter,
        op: eqsat.MarkReachableOp,
        args: tuple[Any, ...],
    ):
        for rule in op.rules.data:
            rule_op = interpreter.get_op_for_symbol(rule)
            assert isinstance(rule_op, pdl_interp.FuncOp), (
                "Expected a pdl_interp.FuncOp as rule."
            )
            self.reachable_rules[rule_op] = True
        return ()

    @impl_terminator(eqsat.CheckAllUnreachableOp)
    def run_check_all_unreachable(
        self,
        interpreter: Interpreter,
        op: eqsat.CheckAllUnreachableOp,
        args: tuple[Any, ...],
    ):
        if not self.reachable_rules:
            return Successor(op.unreachable_dest, ()), ()
        else:
            return Successor(op.reachable_dest, ()), ()

    @impl_terminator(pdl_interp.RecordMatchOp)
    def run_recordmatch(
        self,
        interpreter: Interpreter,
        op: pdl_interp.RecordMatchOp,
        args: tuple[Any, ...],
    ):
        # Record that this rule matched for scheduling purposes
        rule = interpreter.get_op_for_symbol(op.rewriter)
        assert isinstance(rule, pdl_interp.FuncOp)
        self.scheduler.record_rule_match(rule)

        logger.debug(f"Rule '{op.rewriter}' matched and will be applied")

        self.pending_rewrites.append(
            (op.rewriter, self.rewriter.current_operation, args)
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
                self.num_reachable_rules = backtrack_point.num_reachable_rules
                self.reachable_rules = backtrack_point.reachable_rules
                return Successor(backtrack_point.block, backtrack_point.block_args), ()
        return ReturnedValues(()), ()

    def repair(self, eclass: eqsat.EClassOp):
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
                assert isinstance(eclass1 := op1_use.operation, eqsat.EClassOp)

                assert len(op2.results) == 1, (
                    "Expected a single result for the operation being modified."
                )
                assert (op2_use := op2.results[0].first_use), (
                    "Modification handler currently only supports operations with a single (EClassOp) use"
                )
                assert isinstance(eclass2 := op2_use.operation, eqsat.EClassOp)

                # This temporarily breaks the invariant since eclass2 will now contain the result of op2 twice.
                # Callling `eclass_union` will deduplicate this operand.
                self.rewriter.replace_op(op1, new_ops=(), new_results=op2.results)

                if eclass1 == eclass2:
                    eclass1.operands = OrderedSet(
                        eclass1.operands
                    )  # deduplicate operands
                    continue  # parents need not be processed because no eclasses were merged

                assert self.eclass_union(eclass1, eclass2), (
                    "Expected eclasses to not already be unioned."
                )

                self.worklist.append(eclass1)
            else:
                unique_parents[op1] = op1

    def rebuild(self):
        while self.worklist:
            todo = OrderedSet(self.eclass_union_find.find(c) for c in self.worklist)
            self.worklist.clear()
            for c in todo:
                self.repair(c)

    def execute_pending_rewrites(self, interpreter: Interpreter):
        """Execute all pending rewrites that were aggregated during matching."""
        for rewriter, root, args in self.pending_rewrites:
            self.rewriter.current_operation = root

            self.is_matching = False
            interpreter.call_op(rewriter, args)
            self.is_matching = True
        self.pending_rewrites.clear()
