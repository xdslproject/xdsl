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
from xdsl.ir import Attribute, Block, Operation, OpResult, SSAValue, Use
from xdsl.rewriter import InsertPoint
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


@dataclass
class MergeTodo:
    to_keep: eqsat.EClassOp
    to_replace: eqsat.EClassOp


@register_impls
@dataclass
class EqsatPDLInterpFunctions(PDLInterpFunctions):
    backtrack_stack: list[BacktrackPoint] = field(default_factory=list[BacktrackPoint])
    visited: bool = True
    known_ops: KnownOps = field(default_factory=KnownOps)
    known_ops_restore_list: list[Operation] = field(default_factory=list[Operation])
    eclass_union_find: DisjointSet[eqsat.EClassOp] = field(
        default_factory=lambda: DisjointSet[eqsat.EClassOp]()
    )
    merge_list: list[MergeTodo] = field(default_factory=list[MergeTodo])
    deterministic_operand_ordering: bool = True

    def modification_handler(self, op: Operation):
        if op in self.known_ops:
            removed = self.known_ops.pop(op)
            self.known_ops_restore_list.append(removed)

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

        it = iter(input_op.results[0].uses)
        original_eclass = next(it).operation
        if not isinstance(original_eclass, eqsat.EClassOp):
            raise InterpretationError(
                "Replaced operation result must be used by an EClassOp"
            )
        operands_to_delete: list[int] = []
        if len(input_op.results[0].uses) != 1:
            for use in it:
                if use.operation is not original_eclass:
                    raise InterpretationError(
                        "Replaced operation result must be used by a single EClassOp"
                    )
                operands_to_delete.append(use.index)
        for index in operands_to_delete:
            original_eclass.delete_operand(index)

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

        repl_eclass = self.eclass_union_find.find(repl_eclass)
        original_eclass = self.eclass_union_find.find(original_eclass)

        if repl_eclass == original_eclass:
            return ()

        self.eclass_union_find.union(
            original_eclass,
            repl_eclass,
        )
        if self.eclass_union_find.find(original_eclass) == repl_eclass:
            # In the union-find the canonical representative of the original_eclass
            # is now the repl_eclass, so we have to keep the repl_eclass:
            self.merge_list.append(MergeTodo(repl_eclass, original_eclass))
        else:
            # otherwise we keep the original_eclass:
            self.merge_list.append(MergeTodo(original_eclass, repl_eclass))

        return ()

    @impl(pdl_interp.CreateOperationOp)
    def run_createoperation(
        self,
        interpreter: Interpreter,
        op: pdl_interp.CreateOperationOp,
        args: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        has_done_action_checkpoint = self.rewriter.has_done_action
        (new_op,) = PDLInterpFunctions.run_create_operation(
            self, interpreter, op, args
        ).values

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

    def apply_matches(self):
        todo = set(
            (self.eclass_union_find.find(todo.to_keep), todo.to_replace)
            for todo in self.merge_list
        )
        self.merge_list.clear()
        for to_keep, to_replace in todo:
            operands = to_keep._operands  # pyright: ignore[reportPrivateUsage]
            startlen = len(operands)
            for i, val in enumerate(to_replace.operands):
                val.add_use(Use(to_keep, startlen + i))
                new_operands = operands + to_replace._operands  # pyright: ignore[reportPrivateUsage]
                if self.deterministic_operand_ordering:
                    new_operands = tuple(
                        sorted(new_operands, key=lambda v: v.__hash__())
                    )
                to_keep._operands = new_operands  # pyright: ignore[reportPrivateUsage]

            self.rewriter.replace_op(
                to_replace, new_ops=[], new_results=to_keep.results
            )

        while self.known_ops_restore_list:
            op = self.known_ops_restore_list.pop()
            assert op not in self.known_ops
            self.known_ops[op] = op
