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
class OpOperand:
    idx: int


@dataclass(frozen=True)
class OpResult:
    idx: int


@dataclass
class Tie:
    elements: frozenset[OpResult | OpOperand]
    register: RegisterType | None


class OpTieConstraints(RegisterAllocationConstraints):

    ConstraintsDict: TypeAlias = dict[OpOperand | OpResult, Tie]

    _constraints: ConstraintsDict

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
        var_constraints: dict[str, list[OpOperand | OpResult]] = {}
        for idx, (_, reg_def) in enumerate(arg_defs):
            constr = reg_def.constr
            if isinstance(constr, RangeOf | SingleOf) and isinstance(
                constr.constr, VarConstraint
            ):
                # We use the VarConstraint name as a dict key just like
                # ConstraintContext
                var_name: str = constr.constr.name
                var_constraints.setdefault(var_name, []).append(OpOperand(idx))
        for idx, (_, reg_def) in enumerate(result_defs):
            constr = reg_def.constr
            if isinstance(constr, RangeOf | SingleOf) and isinstance(
                constr.constr, VarConstraint
            ):
                # We use the VarConstraint name as a dict key just like
                # ConstraintContext
                var_name: str = constr.constr.name
                var_constraints.setdefault(var_name, []).append(OpResult(idx))
        tie_constraints = [
            Tie(elements=frozenset(value), register=None)
            for value in var_constraints.values()
        ]
        ret = OpTieConstraints(
            {idx: tie for tie in tie_constraints for idx in tie.elements}
        )
        # Take into account any pre-allocated result/operand
        for idx, arg in enumerate(op.operands):
            if not (
                isinstance(arg.type, RegisterType)
                and arg.type.is_allocated
                and ret.operand_has_constraints(idx)
            ):
                continue
            ret.operand_satisfy_constraint(idx, arg.type)
        for idx, res in enumerate(op.results):
            if not (
                isinstance(res.type, RegisterType)
                and res.type.is_allocated
                and ret.result_has_constraints(idx)
            ):
                continue
            ret.result_satisfy_constraint(idx, res.type)
        return ret

    def __init__(self, constr: ConstraintsDict | None = None) -> None:
        if constr is not None:
            self._constraints = constr
        else:
            self._constraints = {}
        super().__init__()

    def _has_constraints(self, element: OpOperand | OpResult) -> bool:
        return element in self._constraints

    def _is_constrained_to(self, element: OpOperand | OpResult) -> RegisterType | None:
        if element not in self._constraints:
            return None
        return self._constraints[element].register

    def _satisfy_constraint(
        self, element: OpOperand | OpResult, reg: RegisterType
    ) -> None:
        if element not in self._constraints:
            raise KeyError()
        current_reg = self._constraints[element].register
        if current_reg is not None and current_reg != reg:
            raise ValueError()
        self._constraints[element].register = reg

    def operand_has_constraints(self, idx: int) -> bool:
        return self._has_constraints(OpOperand(idx))

    def operand_is_constrained_to(self, idx: int) -> RegisterType | None:
        return self._is_constrained_to(OpOperand(idx))

    def operand_satisfy_constraint(self, idx: int, reg: RegisterType) -> None:
        self._satisfy_constraint(OpOperand(idx), reg)

    def result_has_constraints(self, idx: int) -> bool:
        return self._has_constraints(OpResult(idx))

    def result_is_constrained_to(self, idx: int) -> RegisterType | None:
        return self._is_constrained_to(OpResult(idx))

    def result_satisfy_constraint(self, idx: int, reg: RegisterType) -> None:
        self._satisfy_constraint(OpResult(idx), reg)
