from dataclasses import dataclass
from typing import Any

from xdsl.dialects import riscv_cf
from xdsl.interpreter import (
    Interpreter,
    InterpreterFunctions,
    PythonValues,
    Successor,
    impl_terminator,
    register_impls,
)


@dataclass
@register_impls
class RiscvCfFunctions(InterpreterFunctions):
    bitwidth: int = 32

    @impl_terminator(riscv_cf.JOp)
    def run_j(
        self,
        interpreter: Interpreter,
        op: riscv_cf.JOp,
        args: tuple[Any, ...],
    ):
        return Successor(op.successor, args), ()

    @impl_terminator(riscv_cf.BranchOp)
    def run_branch(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BranchOp,
        args: tuple[Any, ...],
    ):
        return Successor(op.successor, args), ()

    def run_cond_branch(
        self,
        interpreter: Interpreter,
        op: riscv_cf.ConditionalBranchOperation,
        lhs: int,
        rhs: int,
        bitwidth: int,
    ) -> tuple[Successor, PythonValues]:
        cond = op.const_evaluate(lhs, rhs, bitwidth)
        if cond:
            block_args = interpreter.get_values(op.then_arguments)
            return Successor(op.then_block, block_args), ()
        else:
            block_args = interpreter.get_values(op.else_arguments)
            return Successor(op.else_block, block_args), ()

    @impl_terminator(riscv_cf.BeqOp)
    def run_beq(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BeqOp,
        args: tuple[Any, ...],
    ):
        return self.run_cond_branch(interpreter, op, args[0], args[1], self.bitwidth)

    @impl_terminator(riscv_cf.BneOp)
    def run_bne(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BneOp,
        args: tuple[Any, ...],
    ):
        return self.run_cond_branch(interpreter, op, args[0], args[1], self.bitwidth)

    @impl_terminator(riscv_cf.BltOp)
    def run_blt(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BltOp,
        args: tuple[Any, ...],
    ):
        return self.run_cond_branch(interpreter, op, args[0], args[1], self.bitwidth)

    @impl_terminator(riscv_cf.BgeOp)
    def run_bge(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BgeOp,
        args: tuple[Any, ...],
    ):
        return self.run_cond_branch(interpreter, op, args[0], args[1], self.bitwidth)

    @impl_terminator(riscv_cf.BltuOp)
    def run_bltu(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BltuOp,
        args: tuple[Any, ...],
    ):
        return self.run_cond_branch(interpreter, op, args[0], args[1], self.bitwidth)

    @impl_terminator(riscv_cf.BgeuOp)
    def run_bgeu(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BgeuOp,
        args: tuple[Any, ...],
    ):
        return self.run_cond_branch(interpreter, op, args[0], args[1], self.bitwidth)
