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
from xdsl.interpreters.comparisons import (
    to_signed,
    to_unsigned,
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
        cond: bool,
    ) -> tuple[Successor, PythonValues]:
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
        unsigned_lhs = to_unsigned(args[0], self.bitwidth)
        unsigned_rhs = to_unsigned(args[1], self.bitwidth)
        return self.run_cond_branch(interpreter, op, unsigned_lhs == unsigned_rhs)

    @impl_terminator(riscv_cf.BneOp)
    def run_bne(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BneOp,
        args: tuple[Any, ...],
    ):
        unsigned_lhs = to_unsigned(args[0], self.bitwidth)
        unsigned_rhs = to_unsigned(args[1], self.bitwidth)
        return self.run_cond_branch(interpreter, op, unsigned_lhs != unsigned_rhs)

    @impl_terminator(riscv_cf.BltOp)
    def run_blt(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BltOp,
        args: tuple[Any, ...],
    ):
        signed_lhs = to_signed(args[0], self.bitwidth)
        signed_rhs = to_signed(args[1], self.bitwidth)
        return self.run_cond_branch(interpreter, op, signed_lhs < signed_rhs)

    @impl_terminator(riscv_cf.BgeOp)
    def run_bge(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BgeOp,
        args: tuple[Any, ...],
    ):
        signed_lhs = to_signed(args[0], self.bitwidth)
        signed_rhs = to_signed(args[1], self.bitwidth)
        return self.run_cond_branch(interpreter, op, signed_lhs >= signed_rhs)

    @impl_terminator(riscv_cf.BltuOp)
    def run_bltu(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BltuOp,
        args: tuple[Any, ...],
    ):
        unsigned_lhs = to_unsigned(args[0], self.bitwidth)
        unsigned_rhs = to_unsigned(args[1], self.bitwidth)
        return self.run_cond_branch(interpreter, op, unsigned_lhs < unsigned_rhs)

    @impl_terminator(riscv_cf.BgeuOp)
    def run_bgeu(
        self,
        interpreter: Interpreter,
        op: riscv_cf.BgeuOp,
        args: tuple[Any, ...],
    ):
        unsigned_lhs = to_unsigned(args[0], self.bitwidth)
        unsigned_rhs = to_unsigned(args[1], self.bitwidth)
        return self.run_cond_branch(interpreter, op, unsigned_lhs >= unsigned_rhs)
