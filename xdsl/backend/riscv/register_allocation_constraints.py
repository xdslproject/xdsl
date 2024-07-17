from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from xdsl.backend.register_type import RegisterType
from xdsl.irdl import (
    IRDLOperation,
    OperandDef,
    RangeOf,
    ResultDef,
    SingleOf,
    VarConstraint,
    VarIRConstruct,
    get_construct_defs,
)


class RegisterAllocationConstraints(abc.ABC):
    """
    Base class for register allocation constraints.
    """

    @abc.abstractmethod
    def operand_has_constraints(self, idx: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def operand_is_constrained_to(self, idx: int) -> RegisterType | None:
        raise NotImplementedError()

    @abc.abstractmethod
    def operand_satisfy_constraint(self, idx: int, reg: RegisterType) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def result_has_constraints(self, idx: int) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def result_is_constrained_to(self, idx: int) -> RegisterType | None:
        raise NotImplementedError()

    @abc.abstractmethod
    def result_satisfy_constraint(self, idx: int, reg: RegisterType) -> None:
        raise NotImplementedError()


@dataclass(frozen=True)
class OpOperandIdx:
    idx: int


@dataclass(frozen=True)
class OpResultIdx:
    idx: int


@dataclass
class Tie:
    elements: frozenset[OpResultIdx | OpOperandIdx]
    register: RegisterType | None


class OpTieConstraints(RegisterAllocationConstraints):

    ConstraintsDict: TypeAlias = dict[OpOperandIdx | OpResultIdx, Tie]

    _constraints: ConstraintsDict

    @staticmethod
    def _gather_constraint_vars(
        vars: dict[str, list[OpOperandIdx | OpResultIdx]],
        defs: Sequence[tuple[str, OperandDef | ResultDef]],
    ) -> None:
        for idx, (_, reg_def) in enumerate(defs):
            constr = reg_def.constr
            if isinstance(constr, RangeOf | SingleOf) and isinstance(
                constr.constr, VarConstraint
            ):
                # We use the VarConstraint name as a dict key just like
                # ConstraintContext
                var_name: str = constr.constr.name
                if isinstance(reg_def, OperandDef):
                    tag = OpOperandIdx(idx)
                else:
                    assert isinstance(reg_def, ResultDef)
                    tag = OpResultIdx(idx)
                vars.setdefault(var_name, []).append(tag)

    @staticmethod
    def _gather_preallocated(constr: OpTieConstraints, op: IRDLOperation):
        for idx, arg in enumerate(op.operands):
            if not (
                isinstance(arg.type, RegisterType)
                and arg.type.is_allocated
                and constr.operand_has_constraints(idx)
            ):
                continue
            constr.operand_satisfy_constraint(idx, arg.type)
        for idx, res in enumerate(op.results):
            if not (
                isinstance(res.type, RegisterType)
                and res.type.is_allocated
                and constr.result_has_constraints(idx)
            ):
                continue
            constr.result_satisfy_constraint(idx, res.type)

    @staticmethod
    def from_op(op: IRDLOperation) -> OpTieConstraints:
        op_def = op.get_irdl_definition()
        result_defs = cast(
            Sequence[tuple[str, ResultDef]],
            get_construct_defs(op_def, VarIRConstruct.RESULT),
        )
        arg_defs = cast(
            Sequence[tuple[str, OperandDef]],
            get_construct_defs(op_def, VarIRConstruct.OPERAND),
        )
        # Gather all ConstraintVars associated to operands/results
        constraint_vars: dict[str, list[OpOperandIdx | OpResultIdx]] = {}
        OpTieConstraints._gather_constraint_vars(constraint_vars, result_defs)
        OpTieConstraints._gather_constraint_vars(constraint_vars, arg_defs)
        tie_constraints = [
            Tie(elements=frozenset(value), register=None)
            for value in constraint_vars.values()
        ]
        # Init the constraints dict
        ret = OpTieConstraints(
            {idx: tie for tie in tie_constraints for idx in tie.elements}
        )
        # Take into account any pre-allocated result/operand
        OpTieConstraints._gather_preallocated(ret, op)
        return ret

    def __init__(self, constr: ConstraintsDict | None = None) -> None:
        if constr is not None:
            self._constraints = constr
        else:
            self._constraints = {}
        super().__init__()

    def _has_constraints(self, element: OpOperandIdx | OpResultIdx) -> bool:
        return element in self._constraints

    def _is_constrained_to(
        self, element: OpOperandIdx | OpResultIdx
    ) -> RegisterType | None:
        if element not in self._constraints:
            return None
        return self._constraints[element].register

    def _satisfy_constraint(
        self, element: OpOperandIdx | OpResultIdx, reg: RegisterType
    ) -> None:
        if element not in self._constraints:
            raise KeyError()
        current_reg = self._constraints[element].register
        if current_reg is not None and current_reg != reg:
            raise ValueError()
        self._constraints[element].register = reg

    def operand_has_constraints(self, idx: int) -> bool:
        return self._has_constraints(OpOperandIdx(idx))

    def operand_is_constrained_to(self, idx: int) -> RegisterType | None:
        return self._is_constrained_to(OpOperandIdx(idx))

    def operand_satisfy_constraint(self, idx: int, reg: RegisterType) -> None:
        self._satisfy_constraint(OpOperandIdx(idx), reg)

    def result_has_constraints(self, idx: int) -> bool:
        return self._has_constraints(OpResultIdx(idx))

    def result_is_constrained_to(self, idx: int) -> RegisterType | None:
        return self._is_constrained_to(OpResultIdx(idx))

    def result_satisfy_constraint(self, idx: int, reg: RegisterType) -> None:
        self._satisfy_constraint(OpResultIdx(idx), reg)
