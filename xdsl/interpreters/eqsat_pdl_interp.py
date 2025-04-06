from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from xdsl.dialects import eqsat, pdl_interp
from xdsl.interpreter import (
    Interpreter,
    ReturnedValues,
    Successor,
    impl,
    impl_terminator,
    register_impls,
)
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Attribute, Block, Operation, OpResult, SSAValue
from xdsl.utils.exceptions import InterpretationError
from xdsl.utils.scoped_dict import ScopedDict


@dataclass
class BacktrackPoint:
    block: Block
    block_args: tuple[SSAValue, ...]
    scope: ScopedDict[SSAValue, Any]
    gdo_op: pdl_interp.GetDefiningOpOp
    index: int
    max_index: int


@register_impls
@dataclass
class EqsatPDLInterpFunctions(PDLInterpFunctions):
    backtrack_stack: list[BacktrackPoint] = field(default_factory=list)
    visited: bool = True

    @impl(pdl_interp.GetResultOp)
    def run_getresult(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetResultOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        result = cast(
            None | OpResult[Attribute],
            PDLInterpFunctions.run_getresult(self, interpreter, op, args).values[0],
        )

        if (
            result
            and len(result.uses) == 1
            and isinstance(
                eclass_op := next(iter(result.uses)).operation, eqsat.EClassOp
            )
        ):
            assert len(eclass_op.results) == 1
            result = eclass_op.results[0]

        return (result,)

    @impl(pdl_interp.GetResultsOp)
    def run_getresults(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetResultsOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        raise InterpretationError("TODO: pdl_interp.get_results")

    @impl(pdl_interp.GetDefiningOpOp)
    def run_getdefiningop(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetDefiningOpOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        defining_op = cast(
            None | Operation,
            PDLInterpFunctions.run_getdefiningop(self, interpreter, op, args).values[0],
        )
        if isinstance(eclass_op := defining_op, eqsat.EClassOp):
            if not self.visited:
                if op != self.backtrack_stack[-1].gdo_op:
                    raise InterpretationError(
                        "TODO: handle the case where a block contains multiple pdl_interp.get_defining_op."
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
            return PDLInterpFunctions.run_getdefiningop(
                self, interpreter, op, (eclass_op.operands[index],)
            ).values

        return (defining_op,)

    @impl(pdl_interp.ReplaceOp)
    def run_replace(
        self,
        interpreter: Interpreter,
        op: pdl_interp.ReplaceOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        raise InterpretationError("TODO: pdl_interp.replace")

    @impl_terminator(pdl_interp.RecordMatchOp)
    def run_recordmatch(
        self,
        interpreter: Interpreter,
        op: pdl_interp.RecordMatchOp,
        args: tuple[Any, ...],
    ):
        raise InterpretationError("TODO: pdl_interp.record_match")

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
