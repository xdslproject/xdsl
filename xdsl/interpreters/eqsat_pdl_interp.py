from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects import eqsat, pdl_interp
from xdsl.dialects.builtin import ModuleOp
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
    def run_getresult(
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

        if len(result.uses) == 1:
            if isinstance(
                eclass_op := next(iter(result.uses)).operation, eqsat.EClassOp
            ):
                result = eclass_op.result
        elif result.uses:  # multiple uses
            for use in result.uses:
                if isinstance(use.operation, eqsat.EClassOp):
                    raise InterpretationError(
                        "pdl_interp.get_result currently only supports operations with results"
                        " that are used by a single EClassOp each."
                    )
        return (result,)

    @impl(pdl_interp.GetResultsOp)
    def run_getresults(
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
            if len(result.uses) == 1:
                if isinstance(
                    eclass_op := next(iter(result.uses)).operation, eqsat.EClassOp
                ):
                    assert len(eclass_op.results) == 1
                    result = eclass_op.results[0]
            elif result.uses:  # multiple uses
                for use in result.uses:
                    if isinstance(use.operation, eqsat.EClassOp):
                        raise InterpretationError(
                            "pdl_interp.get_results currently only supports operations with results"
                            " that are used by a single EClassOp each."
                        )
            results.append(result)
        return (results,)

    @impl(pdl_interp.GetDefiningOpOp)
    def run_getdefiningop(
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
            assert isinstance(args[0].owner, Operation), (
                "Cannot get defining op of a Block argument"
            )
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
                    block, block_args, scope, op, index, len(eclass_op.operands) - 1
                )
            )
        defining_op = eclass_op.operands[index].owner
        assert isinstance(defining_op, Operation)

        return (defining_op,)

    @impl(pdl_interp.CreateOperationOp)
    def run_create_operation(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateOperationOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        has_done_action_checkpoint = self.rewriter.has_done_action
        (new_op,) = super().run_create_operation(interpreter, op, args).values

        assert isinstance(new_op, Operation)

        # Check if an identical operation already exists in our known_ops map
        if existing_op := self.known_ops.get(new_op):
            # CSE can have removed the existing operation, here we check if it is still in use:
            if existing_op.results and existing_op.results[0].uses:
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

    @impl_terminator(pdl_interp.FinalizeOp)
    def run_finalize(
        self, interpreter: Interpreter, _: pdl_interp.FinalizeOp, args: tuple[Any, ...]
    ):
        for backtrack_point in reversed(self.backtrack_stack):
            if backtrack_point.index >= backtrack_point.max_index:
                self.backtrack_stack.pop()
            else:
                backtrack_point.index += 1
                interpreter._ctx = backtrack_point.scope  # pyright: ignore[reportPrivateUsage]
                self.visited = False
                return Successor(backtrack_point.block, backtrack_point.block_args), ()
        return ReturnedValues(()), ()
