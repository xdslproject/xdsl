import abc

from xdsl.dialects import riscv_func, riscv_scf
from xdsl.dialects.riscv import (
    FloatRegisterType,
    IntRegisterType,
    RISCVOp,
    RISCVRegisterType,
)
from xdsl.ir import SSAValue


class RegisterAllocator(abc.ABC):
    """
    Base class for register allocation strategies.
    """

    def __init__(self, limit_registers: int | None = None) -> None:
        pass

    @abc.abstractmethod
    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        raise NotImplementedError()


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

    idx: int

    def __init__(self, limit_registers: int | None = None) -> None:
        self.idx = 0
        self._register_types = (IntRegisterType, FloatRegisterType)

        """
        Assume that all the registers are available except the ones explicitly reserved
        by the default RISCV ABI
        """
        self.reserved_registers = {"zero", "sp", "gp", "tp", "fp", "s0"}

        self.register_sets = {
            IntRegisterType: [
                reg
                for reg in IntRegisterType.RV32I_INDEX_BY_NAME
                if reg not in self.reserved_registers
            ],
            FloatRegisterType: list(FloatRegisterType.RV32F_INDEX_BY_NAME.keys()),
        }

        for reg_type, reg_set in self.register_sets.items():
            if limit_registers is not None:
                self.register_sets[reg_type] = reg_set[:limit_registers]

    def _allocate(self, reg: SSAValue) -> bool:
        if isinstance(reg.type, self._register_types) and not reg.type.is_allocated:
            # If we run out of real registers, allocate a j register
            reg_type = type(reg.type)
            available_regs = self.register_sets.get(reg_type, [])

            if not available_regs:
                reg.type = reg_type(f"j{self.idx}")
                self.idx += 1
            else:
                reg.type = reg_type(available_regs.pop())

            return True

        return False

    def _free(self, reg: SSAValue) -> None:
        if isinstance(reg.type, self._register_types) and reg.type.is_allocated:
            available_regs = self.register_sets.get(type(reg.type), [])
            reg_name = reg.type.register_name

            if not reg_name.startswith("j") and reg_name not in self.reserved_registers:
                available_regs.append(reg_name)

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

    def __init__(self, limit_registers: int | None = None) -> None:
        self.idx = 0
        self._register_types = (IntRegisterType, FloatRegisterType)

        """
        Assume that all the registers are available except the ones explicitly reserved
        by the default RISCV ABI
        """
        reserved_registers = {"zero", "sp", "gp", "tp", "fp", "s0"}

        self.register_sets = {
            IntRegisterType: [
                reg
                for reg in IntRegisterType.RV32I_INDEX_BY_NAME
                if reg not in reserved_registers
            ],
            FloatRegisterType: list(FloatRegisterType.RV32F_INDEX_BY_NAME.keys()),
        }

        for reg_type, reg_set in self.register_sets.items():
            if limit_registers is not None:
                self.register_sets[reg_type] = reg_set[:limit_registers]

    def _allocate(self, reg: SSAValue) -> bool:
        if isinstance(reg.type, self._register_types) and not reg.type.is_allocated:
            # If we run out of real registers, allocate a j register
            reg_type = type(reg.type)
            available_regs = self.register_sets.get(reg_type, [])

            if not available_regs:
                reg.type = reg_type(f"j{self.idx}")
                self.idx += 1
            else:
                reg.type = reg_type(available_regs.pop())

            return True

        return False

    def allocate_for_loop(self, loop: riscv_scf.ForOp) -> None:
        yield_op = loop.body.block.last_op
        assert (
            yield_op is not None
        ), "last op of riscv_scf.ForOp is guaranteed to be riscv_scf.Yield"
        block_args = loop.body.block.args

        # Induction variable
        assert isinstance(block_args[0].type, IntRegisterType)
        self._allocate(block_args[0])

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for block_arg, operand, yield_operand, op_result in zip(
            block_args[1:], loop.iter_args, yield_op.operands, loop.results
        ):
            # If some allocated then assign all to that type, otherwise get new reg
            assert isinstance(block_arg.type, RISCVRegisterType)
            assert isinstance(operand.type, RISCVRegisterType)
            assert isinstance(yield_operand.type, RISCVRegisterType)
            assert isinstance(op_result.type, RISCVRegisterType)

            if not operand.type.is_allocated:
                # We only need to check one of the four since they're constrained to be
                # the same
                self._allocate(operand)

            shared_type = operand.type
            block_arg.type = shared_type
            yield_operand.type = shared_type
            op_result.type = shared_type

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        """
        Sets unallocated registers per block to a finite set of real available registers.
        When it runs out of real registers for a block, it allocates j registers.
        """

        for region in func.regions:
            for block in region.blocks:
                for op in block.walk():
                    if isinstance(op, riscv_scf.ForOp):
                        self.allocate_for_loop(op)

                    # Do not allocate registers on non-RISCV-ops
                    if not isinstance(op, RISCVOp):
                        continue

                    for result in op.results:
                        self._allocate(result)
