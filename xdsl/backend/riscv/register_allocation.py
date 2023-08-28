import abc
from dataclasses import dataclass, field
from typing import overload

from xdsl.dialects import riscv_func, riscv_scf
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import FloatRegisterType, IntRegisterType, Registers, RISCVOp
from xdsl.ir import SSAValue


@dataclass
class RegisterQueue:
    """
    LIFO queue of registers available for allocation.
    """

    _idx: int = 0
    reserved_registers: set[IntRegisterType | FloatRegisterType] = field(
        default_factory=lambda: {
            Registers.ZERO,
            Registers.SP,
            Registers.GP,
            Registers.TP,
            Registers.FP,
            Registers.S0,
            Registers.FS0,
        }
    )
    "Registers unavailable to be used by the register allocator."

    available_int_registers: list[IntRegisterType] = field(
        default_factory=lambda: [
            reg
            for reg_class in (Registers.A, Registers.T, Registers.S[1:])
            for reg in reg_class
        ]
    )
    "Registers that integer values can be allocated to in the current context."

    available_float_registers: list[FloatRegisterType] = field(
        default_factory=lambda: [
            reg
            for reg_class in (Registers.FA, Registers.FT, Registers.FS[1:])
            for reg in reg_class
        ]
    )
    "Registers that floating-point values can be allocated to in the current context."

    def push(self, reg: IntRegisterType | FloatRegisterType) -> None:
        """
        Return a register to be made available for allocation.
        """
        if reg in self.reserved_registers:
            return
        if reg.register_name.startswith("j"):
            return
        if not reg.is_allocated:
            raise ValueError("Cannot push an unallocated register")
        if isinstance(reg, IntRegisterType):
            self.available_int_registers.append(reg)
        else:
            self.available_float_registers.append(reg)

    @overload
    def pop(self, reg_type: type[IntRegisterType]) -> IntRegisterType:
        ...

    @overload
    def pop(self, reg_type: type[FloatRegisterType]) -> FloatRegisterType:
        ...

    def pop(
        self, reg_type: type[IntRegisterType] | type[FloatRegisterType]
    ) -> IntRegisterType | FloatRegisterType:
        """
        Get the next available register for allocation.
        """
        if issubclass(reg_type, IntRegisterType):
            available_registers = self.available_int_registers
        else:
            available_registers = self.available_float_registers

        if available_registers:
            reg = available_registers.pop()
        else:
            reg = reg_type(f"j{self._idx}")
            self._idx += 1
        return reg

    def limit_registers(self, limit: int) -> None:
        """
        Limits the number of currently available registers to the provided limit.
        """
        self.available_int_registers = self.available_int_registers[:limit]
        self.available_float_registers = self.available_float_registers[:limit]


class RegisterAllocator(abc.ABC):
    """
    Base class for register allocation strategies.
    """

    available_registers: RegisterQueue

    def __init__(self, limit_registers: int | None = None) -> None:
        self.available_registers = RegisterQueue()
        if limit_registers is not None:
            self.available_registers.limit_registers(limit_registers)

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
