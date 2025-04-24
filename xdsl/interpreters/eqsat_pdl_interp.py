from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from xdsl.dialects import eqsat, pdl_interp
from xdsl.dialects.builtin import SymbolRefAttr
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
from xdsl.ir import Attribute, Block, Operation, OpResult, SSAValue, Use
from xdsl.rewriter import InsertPoint
from xdsl.transforms.common_subexpression_elimination import KnownOps
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


@dataclass
class Match:
    rewriter: SymbolRefAttr
    args: tuple[Any, ...]


@register_impls
@dataclass
class EqsatPDLInterpFunctions(PDLInterpFunctions):
    backtrack_stack: list[BacktrackPoint] = field(default_factory=list[BacktrackPoint])
    visited: bool = True
    known_ops: KnownOps = field(default_factory=KnownOps)

    @impl(pdl_interp.GetResultOp)
    def run_getresult(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetResultOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        result = cast(
            None | OpResult[Attribute],
            PDLInterpFunctions.run_get_result(self, interpreter, op, args).values[0],
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
        assert isinstance(args[0], Operation)
        src_op = args[0]
        assert op.index is None, (
            "pdl_interp.get_results with index is not yet supported."
        )
        if isinstance(op.result_types[0], ValueType) and len(src_op.results) != 1:
            return (None,)

        results: list[OpResult] = []
        for result in src_op.results:
            if (
                result
                and len(result.uses) == 1
                and isinstance(
                    eclass_op := next(iter(result.uses)).operation, eqsat.EClassOp
                )
            ):
                assert len(eclass_op.results) == 1
                results.append(eclass_op.results[0])
        return (results,)

    @impl(pdl_interp.GetDefiningOpOp)
    def run_getdefiningop(
        self,
        interpreter: Interpreter,
        op: pdl_interp.GetDefiningOpOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        defining_op = cast(
            None | Operation,
            PDLInterpFunctions.run_get_defining_op(self, interpreter, op, args).values[
                0
            ],
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
            return PDLInterpFunctions.run_get_defining_op(
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
        assert args
        input_op = args[0]
        assert isinstance(input_op, Operation)
        assert len(input_op.results) == 1, (
            "ReplaceOp currently only supports replacing operations that have a single result"
        )
        if len(input_op.results[0].uses) != 1 or not isinstance(
            original_eclass := next(iter(input_op.results[0].uses)).operation,
            eqsat.EClassOp,
        ):
            raise InterpretationError(
                "Replaced operation result can only be used by a single e-class operation"
            )

        repl_values = args[1]
        assert len(repl_values) == 1, (
            "pdl_interp.replace currently only a supports replacing with a single e-class result."
        )
        repl_value: SSAValue = repl_values[0]
        repl_eclass = repl_value.owner
        if not isinstance(repl_eclass, eqsat.EClassOp):
            raise InterpretationError(
                "Replacement value must be the result of an EClassOp"
            )

        if repl_eclass == original_eclass:
            return ()

        # TODO: is the below of any use?
        # Check if the repl_eclass operation is already in the original_eclass's operands
        for i, val in enumerate(original_eclass.operands):
            if val.owner and val.owner == repl_eclass:
                # Already present, no need to add it again
                return ()

        operands = original_eclass._operands  # pyright: ignore[reportPrivateUsage]
        startlen = len(operands)
        for i, val in enumerate(repl_eclass.operands):
            val.add_use(Use(original_eclass, startlen + i))
        original_eclass._operands = operands + repl_eclass._operands  # pyright: ignore[reportPrivateUsage]

        # Replace the operation with the replacement values
        self.rewriter.replace_op(
            repl_eclass, new_ops=[], new_results=original_eclass.results
        )
        return ()

    @impl(pdl_interp.CreateOperationOp)
    def run_createoperation(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateOperationOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        (new_op,) = PDLInterpFunctions.run_create_operation(
            self, interpreter, op, args
        ).values

        assert isinstance(new_op, Operation)

        # Check if an identical operation already exists
        if existing_op := self.known_ops.get(new_op):
            self.rewriter.erase_op(new_op)
            return (existing_op,)

        # Record the newly created operation
        self.known_ops[new_op] = new_op

        eclass_op = eqsat.EClassOp(
            new_op.results[0],
        )
        self.rewriter.insert_op(
            eclass_op,
            InsertPoint.after(new_op),
        )

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
