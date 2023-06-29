from abc import ABC
from xdsl.dialects.riscv import (
    RISCVOp,
    Register,
    RegisterType,
    FloatRegister,
    FloatRegisterType,
)
from xdsl.dialects.builtin import ModuleOp


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


class RegisterAllocatorBlockNaive(RegisterAllocator):
    idx: int

    def __init__(self) -> None:
        self.idx = 0

        """
        Since we've got neither right now a handling of a consistent ABI nor of a calling convention,
        let's just assume that we have all the registers available for our use except the one explicitly reserved by the default riscv ABI.
        """

        self.integer_available_registers = list(Register.RV32I_INDEX_BY_NAME.keys())
        reserved_registers = set(["zero", "sp", "gp", "tp", "fp", "s0"])
        self.integer_available_registers = [
            reg
            for reg in self.integer_available_registers
            if reg not in reserved_registers
        ]
        self.floating_available_registers = list(
            FloatRegister.RV32F_INDEX_BY_NAME.keys()
        )

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
                        if isinstance(result.typ, RegisterType):
                            if result.typ.data.name is None:
                                # If we run out of real registers, allocate a j register
                                if not integer_block_registers:
                                    result.typ = RegisterType(Register(f"j{self.idx}"))
                                    self.idx += 1
                                else:
                                    result.typ = RegisterType(
                                        Register(integer_block_registers.pop())
                                    )
                        elif isinstance(result.typ, FloatRegisterType):
                            if result.typ.data.name is None:
                                # If we run out of real registers, allocate a j register
                                if not floating_block_registers:
                                    result.typ = FloatRegisterType(
                                        FloatRegister(f"j{self.idx}")
                                    )
                                    self.idx += 1
                                else:
                                    result.typ = FloatRegisterType(
                                        FloatRegister(floating_block_registers.pop())
                                    )


class RegisterAllocatorJRegs(RegisterAllocator):
    idx: int

    def __init__(self) -> None:
        self.idx = 0

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Sets unallocated registers to an infinite set of `j` registers
        """
        for op in module.walk():
            if not isinstance(op, RISCVOp):
                # Don't perform register allocations on non-RISCV-ops
                continue

            for result in op.results:
                if isinstance(result.typ, RegisterType):
                    if result.typ.data.name is None:
                        result.typ = RegisterType(Register(f"j{self.idx}"))
                        self.idx += 1
                elif isinstance(result.typ, FloatRegisterType):
                    if result.typ.data.name is None:
                        result.typ = FloatRegisterType(FloatRegister(f"j{self.idx}"))
                        self.idx += 1
