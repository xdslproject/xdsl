from abc import ABC

from xdsl.dialects import riscv, riscv_cf
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import FloatRegisterType, Register, RegisterType, RISCVOp
from xdsl.ir import SSAValue


class RegisterAllocator(ABC):
    """
    Base class for register allocation strategies.
    """

    def __init__(self) -> None:
        pass

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Allocates unallocated registers in the module.
        """

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

    def __init__(self, limit_registers: int = 0) -> None:
        self.idx = 0

        """
        Assume that all the registers are available except the ones explicitly reserved
        by the default RISCV ABI
        """
        self.reserved_registers = {"zero", "sp", "gp", "tp", "fp", "s0"}

        self.register_sets = {
            RegisterType: [
                reg
                for reg in Register.RV32I_INDEX_BY_NAME
                if reg not in self.reserved_registers
            ],
            FloatRegisterType: list(Register.RV32F_INDEX_BY_NAME.keys()),
        }

        for reg_type, reg_set in self.register_sets.items():
            if limit_registers:
                self.register_sets[reg_type] = reg_set[:limit_registers]

    @staticmethod
    def _is_allocated(reg: SSAValue) -> bool:
        return (
            isinstance(reg.type, RegisterType | FloatRegisterType)
            and reg.type.data.name is not None
        )

    def _allocate(self, reg: SSAValue) -> bool:
        if isinstance(reg.type, RegisterType | FloatRegisterType):
            if reg.type.data.name is None:
                # if we run out of real registers, allocate a j register
                reg_type = type(reg.type)
                available_regs = self.register_sets.get(reg_type)

                if not available_regs:
                    reg.type = reg_type(Register(f"j{self.idx}"))
                    self.idx += 1
                else:
                    reg.type = reg_type(Register(available_regs.pop()))

                return True

        return False

    def _free(self, reg: SSAValue) -> None:
        if isinstance(reg.type, RegisterType | FloatRegisterType) and (
            available_regs := self.register_sets.get(type(reg.type))
        ):
            if reg.type.data.name is not None:
                if (
                    not reg.type.data.name.startswith("j")
                    and reg.type.data.name not in self.reserved_registers
                ):
                    available_regs.append(reg.type.data.name)

    def allocate_registers(self, module: ModuleOp) -> None:
        for region in module.regions:
            for block in region.blocks:
                to_free: list[SSAValue] = []

                for op in block.walk_reverse():
                    for reg in to_free:
                        self._free(reg)
                    to_free.clear()

                    if not isinstance(op, RISCVOp):
                        # do not allocate registers on non-RISCV-ops
                        continue

                    # allocate registers to operands since they are defined further up
                    # in the use-def SSA chain
                    for operand in op.operands:
                        if not self._is_allocated(operand):
                            self._allocate(operand)

                    # allocate registers to results if not already allocated,
                    # otherwise free that register since the SSA value is created here
                    for result in op.results:
                        if not self._is_allocated(result):
                            # results not already allocated, they still need a register,
                            # so allocate and record them to be freed before processing
                            # the next instruction
                            if self._allocate(result):
                                to_free.append(result)
                        else:
                            self._free(result)


class RegisterAllocatorBlockNaive(RegisterAllocator):
    idx: int

    def __init__(self, limit_registers: int = 0) -> None:
        self.idx = 0
        _ = limit_registers

        """
        Since we've got neither right now a handling of a consistent ABI nor of a
        calling convention, let's just assume that we have all the registers available
        for our use except the one explicitly reserved by the default riscv ABI.
        """

        self.integer_available_registers = list(Register.RV32I_INDEX_BY_NAME.keys())
        reserved_registers = {"zero", "sp", "gp", "tp", "fp", "s0"}
        self.integer_available_registers = [
            reg
            for reg in self.integer_available_registers
            if reg not in reserved_registers
        ]
        self.floating_available_registers = list(Register.RV32F_INDEX_BY_NAME.keys())

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Sets unallocated registers for each block to a finite set of real available registers.
        When it runs out of real registers for a block, it allocates j registers.
        """

        for region in module.regions:
            for block in region.blocks:
                integer_block_registers = self.integer_available_registers.copy()
                floating_block_registers = self.floating_available_registers.copy()

                for op in block.walk():
                    if not isinstance(op, RISCVOp):
                        # Don't perform register allocations on non-RISCV-ops
                        continue

                    for result in op.results:
                        if isinstance(result.type, RegisterType):
                            if result.type.data.name is None:
                                # If we run out of real registers, allocate a j register
                                if not integer_block_registers:
                                    result.type = RegisterType(Register(f"j{self.idx}"))
                                    self.idx += 1
                                else:
                                    result.type = RegisterType(
                                        Register(integer_block_registers.pop())
                                    )
                        elif isinstance(result.type, FloatRegisterType):
                            if result.type.data.name is None:
                                # If we run out of real registers, allocate a j register
                                if not floating_block_registers:
                                    result.type = FloatRegisterType(
                                        Register(f"j{self.idx}")
                                    )
                                    self.idx += 1
                                else:
                                    result.type = FloatRegisterType(
                                        Register(floating_block_registers.pop())
                                    )


class RegisterAllocatorJRegs(RegisterAllocator):
    idx: int

    def __init__(self, limit_registers: int = 0) -> None:
        self.idx = 0
        _ = limit_registers

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Sets unallocated registers to an infinite set of `j` registers
        """
        # First allocate registers for block arguments
        for op in module.walk():
            if isinstance(op, riscv_cf.BranchOperation):
                for i, arg in enumerate(op.then_block.args):
                    assert isinstance(arg.type, RegisterType)
                    if arg.type.data.name is None:
                        typ = RegisterType(Register(f"j{self.idx}"))
                        arg.type = typ
                        self.idx += 1
                    else:
                        typ = arg.type
                    op.then_arguments[i].type = typ

                for i, arg in enumerate(op.else_block.args):
                    assert isinstance(arg.type, RegisterType)
                    if arg.type.data.name is None:
                        typ = RegisterType(Register(f"j{self.idx}"))
                        arg.type = typ
                        self.idx += 1
                    else:
                        typ = arg.type
                    op.else_arguments[i].type = typ

            elif isinstance(op, riscv_cf.JOp):
                for i, arg in enumerate(op.successor.args):
                    assert isinstance(arg.type, RegisterType)
                    if arg.type.data.name is None:
                        typ = RegisterType(Register(f"j{self.idx}"))
                        arg.type = typ
                        self.idx += 1
                    else:
                        typ = arg.type
                    op.block_arguments[i].type = typ

        for op in module.walk():
            if not isinstance(op, RISCVOp):
                # Don't perform register allocations on non-RISCV-ops
                continue

            if (
                isinstance(op, riscv.RdRsRsIntegerOperation)
                and op.rd_operand is not None
            ):
                assert isinstance(op.rd.type, RegisterType)
                if op.rd.type.data.name is None:
                    op.rd.type = op.rd_operand.type

            if (
                isinstance(op, riscv.RdRsImmIntegerOperation)
                and op.rd_operand is not None
            ):
                assert isinstance(op.rd.type, RegisterType)
                if op.rd.type.data.name is None:
                    op.rd.type = op.rd_operand.type

            for result in op.results:
                if isinstance(result.type, RegisterType):
                    if result.type.data.name is None:
                        result.type = RegisterType(Register(f"j{self.idx}"))
                        self.idx += 1
                elif isinstance(result.type, FloatRegisterType):
                    if result.type.data.name is None:
                        result.type = FloatRegisterType(Register(f"j{self.idx}"))
                        self.idx += 1
