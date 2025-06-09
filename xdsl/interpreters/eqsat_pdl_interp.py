from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xdsl.dialects import eqsat, pdl_interp
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.pdl import ValueType
from xdsl.interpreter import (
    Interpreter,
    impl,
    register_impls,
)
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.ir import Operation, OpResult
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet
from xdsl.utils.exceptions import InterpretationError


@register_impls
@dataclass
class EqsatPDLInterpFunctions(PDLInterpFunctions):
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

        if len(result.uses) != 1 or not isinstance(
            eclass_op := next(iter(result.uses)).operation, eqsat.EClassOp
        ):
            raise InterpretationError(
                "pdl_interp.get_result currently only supports operations with results"
                " that are used by a single EClassOp each."
            )
        result = eclass_op.result

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
            if len(result.uses) == 1 and isinstance(
                eclass_op := next(iter(result.uses)).operation, eqsat.EClassOp
            ):
                assert len(eclass_op.results) == 1
                results.append(eclass_op.results[0])
            else:
                raise InterpretationError(
                    "pdl_interp.get_results currently only supports operations with results"
                    " that are used by a single EClassOp each."
                )
        return (results,)
