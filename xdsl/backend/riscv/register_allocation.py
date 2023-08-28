from __future__ import annotations

import abc
from collections.abc import Callable

from xdsl.backend.riscv.register_queue import RegisterQueue
from xdsl.dialects import riscv_func, riscv_scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import (
    FloatRegisterType,
    IntRegisterType,
    RISCVOp,
)
from xdsl.ir import Operation, SSAValue


class RegisterAllocator(abc.ABC):
    """
    Base class for register allocation strategies.
    """

    _available_registers: RegisterQueue | None
    available_registers_factory: Callable[[], RegisterQueue]

    def __init__(self, limit_registers: int | None = None) -> None:
        self._available_registers = None
        if limit_registers is not None:

            def factory():
                available_registers = RegisterQueue()
                available_registers.limit_registers(limit_registers)
                return available_registers

            self.available_registers_factory = factory
        else:
            self.available_registers_factory = lambda: RegisterQueue()

    @property
    def available_registers(self) -> RegisterQueue:
        if self._available_registers is None:
            self._available_registers = self.available_registers_factory()
        return self._available_registers

    def reset_available_registers(self) -> None:
        self._available_registers = None

    @abc.abstractmethod
    def process_operation(self, op: Operation) -> None:
        raise NotImplementedError()

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        if len(func.body.blocks) != 1:
            raise NotImplementedError(
                f"Cannot register allocate func with {len(func.body.blocks)} blocks."
            )

        for op in func.body.block.ops_reverse:
            self.process_operation(op)

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Allocates unallocated registers in the module.
        """
        for op in module.walk():
            if isinstance(op, riscv_func.FuncOp):
                self.allocate_func(op)

    def process_for_op(self, op: riscv_scf.ForOp) -> None:
        yield_op = op.body.block.last_op
        assert (
            yield_op is not None
        ), "last op of riscv_scf.ForOp is guaranteed to be riscv_scf.Yield"
        block_args = op.body.block.args

        # Induction variable
        assert isinstance(block_args[0].type, IntRegisterType)
        if not block_args[0].type.is_allocated:
            block_args[0].type = self.available_registers.pop(IntRegisterType)

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for i, (block_arg, operand, yield_operand, op_result) in enumerate(
            zip(block_args[1:], op.iter_args, yield_op.operands, op.results)
        ):
            assert isinstance(block_arg.type, IntRegisterType)
            assert isinstance(operand.type, IntRegisterType)
            assert isinstance(yield_operand.type, IntRegisterType)
            assert isinstance(op_result.type, IntRegisterType)

            shared_type: IntRegisterType | None = None
            if block_arg.type.is_allocated:
                shared_type = block_arg.type

            if operand.type.is_allocated:
                if shared_type is not None:
                    if shared_type != operand.type:
                        raise ValueError(
                            "Operand iteration variable types must match: "
                            f"operand {i} type: {operand.type}, block argument {i+1} "
                            f"type: {block_arg.type}, yield operand {0} type: "
                            f"{yield_operand.type}"
                        )
                else:
                    shared_type = operand.type

            if yield_operand.type.is_allocated:
                if shared_type is not None:
                    if shared_type != yield_operand.type:
                        raise ValueError(
                            "Operand iteration variable types must match: "
                            f"operand {i} type: {operand.type}, block argument {i+1} "
                            f"type: {block_arg.type}, yield operand {0} type: "
                            f"{yield_operand.type}"
                        )
                else:
                    shared_type = yield_operand.type

            if op_result.type.is_allocated:
                if shared_type is not None:
                    if shared_type != op_result.type:
                        raise ValueError(
                            "Operand iteration variable types must match: "
                            f"operand {i} type: {operand.type}, block argument {i+1} "
                            f"type: {block_arg.type}, yield operand {0} type: "
                            f"{yield_operand.type}"
                        )
                else:
                    shared_type = op_result.type

            if shared_type is None:
                shared_type = self.available_registers.pop(IntRegisterType)

            block_arg.type = shared_type
            operand.type = shared_type
            yield_operand.type = shared_type
            op_result.type = shared_type

        # Allocate loop body
        for loop_op in op.body.block.ops_reverse:
            self.process_operation(loop_op)


class RegisterAllocatorLivenessBlockNaive(RegisterAllocator):
    """
    It traverses the use-def SSA chain backwards (i.e., from uses to defs) and:
      1. allocates registers for operands
      2. frees registers for results (since that will be the last time they appear when
      going backwards)
    It currently operates on blocks.

    This is a simplified version of the standard bottom-up local register allocation
    algorithm.

    A relevant reference in "Engineering a Compiler, Edition 3" ISBN: 9780128154120.

    ```
    for op in block.walk_reverse():
    for operand in op.operands:
        if operand is not allocated:
            allocate(operand)

    for result in op.results:
    if result is not allocated:
        allocate(result)
        free_before_next_instruction.append(result)
    else:
        free(result)
    ```
    """

    to_free: list[SSAValue]

    def __init__(self, limit_registers: int | None = None) -> None:
        super().__init__(limit_registers=limit_registers)
        self.to_free = []

    def _allocate(self, reg: SSAValue):
        if (
            isinstance(reg.type, IntRegisterType | FloatRegisterType)
            and not reg.type.is_allocated
        ):
            reg.type = self.available_registers.pop(type(reg.type))

    def _free(self, reg: SSAValue) -> None:
        if (
            isinstance(reg.type, IntRegisterType | FloatRegisterType)
            and reg.type.is_allocated
        ):
            self.available_registers.push(reg.type)

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        self.to_free.clear()
        super().allocate_func(func)

    def process_operation(self, op: Operation) -> None:
        for reg in self.to_free:
            self._free(reg)
        self.to_free.clear()

        # Do not allocate registers on non-RISCV-ops
        if not isinstance(op, RISCVOp):
            return

        # Allocate registers to operands since they are defined further up
        # in the use-def SSA chain
        for operand in op.operands:
            self._allocate(operand)

        # Allocate registers to results if not already allocated,
        # otherwise free that register since the SSA value is created here
        for result in op.results:
            # Unallocated results still need a register,
            # so allocate and keep track of them to be freed
            # before processing the next instruction
            self._allocate(result)
            self.to_free.append(result)

    def process_for_op(self, op: riscv_scf.ForOp) -> None:
        raise NotImplementedError("Cannot allocate for op with live ranges")


class RegisterAllocatorBlockNaive(RegisterAllocator):
    idx: int

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        """
        Sets unallocated registers per block to a finite set of real available registers.
        When it runs out of real registers for a block, it allocates j registers.
        """
        self.reset_available_registers()
        super().allocate_func(func)

    def process_operation(self, op: Operation) -> None:
        # Do not allocate registers on non-RISCV-ops
        if not isinstance(op, RISCVOp):
            return

        for result in op.results:
            if isinstance(result.type, IntRegisterType | FloatRegisterType):
                if not result.type.is_allocated:
                    result.type = self.available_registers.pop(type(result.type))


class RegisterAllocatorJRegs(RegisterAllocator):
    """
    Sets unallocated registers to an infinite set of `j` registers
    """

    def __init__(self, limit_registers: int | None = None) -> None:
        assert limit_registers is None
        super().__init__(0)

    def process_operation(self, op: Operation) -> None:
        if isinstance(op, riscv_scf.ForOp):
            self.process_for_op(op)
            return

        # Do not allocate registers on non-RISCV-ops
        if not isinstance(op, RISCVOp):
            return

        for result in op.results:
            if isinstance(result.type, IntRegisterType | FloatRegisterType):
                if not result.type.is_allocated:
                    result.type = self.available_registers.pop(type(result.type))
