from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

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
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet
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
    backtrack_stack: list[BacktrackPoint] = field(default_factory=list[BacktrackPoint])
    visited: bool = True
    known_ops: KnownOps = field(default_factory=KnownOps)
    eclass_union_find: DisjointSet[eqsat.EClassOp] = field(
        default_factory=lambda: DisjointSet[eqsat.EClassOp]()
    )

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
