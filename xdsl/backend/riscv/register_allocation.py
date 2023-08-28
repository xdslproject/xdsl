from __future__ import annotations

import abc
from collections.abc import Callable

from xdsl.backend.riscv.register_queue import RegisterQueue
from xdsl.dialects import riscv_func, riscv_scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import FloatRegisterType, IntRegisterType, RISCVOp
from xdsl.ir import SSAValue


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
    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        raise NotImplementedError()

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Allocates unallocated registers in the module.
        """
        for op in module.walk():
            if isinstance(op, riscv_func.FuncOp):
                self.allocate_func(op)


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
        for region in func.regions:
            for block in region.blocks:
                to_free: list[SSAValue] = []

                for op in block.walk_reverse():
                    for reg in to_free:
                        self._free(reg)
                    to_free.clear()

                    # Do not allocate registers on non-RISCV-ops
                    if not isinstance(op, RISCVOp):
                        continue

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
                        to_free.append(result)


class RegisterAllocatorBlockNaive(RegisterAllocator):
    idx: int

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        """
        Sets unallocated registers per block to a finite set of real available registers.
        When it runs out of real registers for a block, it allocates j registers.
        """

        for region in func.regions:
            for block in region.blocks:
                self.reset_available_registers()

                for op in block.walk():
                    # Do not allocate registers on non-RISCV-ops
                    if not isinstance(op, RISCVOp):
                        continue

                    for result in op.results:
                        if isinstance(result.type, IntRegisterType | FloatRegisterType):
                            if not result.type.is_allocated:
                                result.type = self.available_registers.pop(
                                    type(result.type)
                                )


class RegisterAllocatorJRegs(RegisterAllocator):
    def __init__(self, limit_registers: int | None = None) -> None:
        assert limit_registers is None
        super().__init__(0)

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        """
        Sets unallocated registers to an infinite set of `j` registers
        """
        for op in func.walk():
            if isinstance(op, riscv_scf.ForOp):
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
                    # TODO: instead of checking that they're all the same, check whether they are all None, or if all the not-None are the same reg.
                    # if some allocated then assign all to that type, otherwise get new j reg
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

            # Do not allocate registers on non-RISCV-ops
            if not isinstance(op, RISCVOp):
                continue

            for result in op.results:
                if isinstance(result.type, IntRegisterType | FloatRegisterType):
                    if not result.type.is_allocated:
                        result.type = self.available_registers.pop(type(result.type))
