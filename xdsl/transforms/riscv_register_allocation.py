from enum import Enum
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass


class RegisterAllocatorStrategy:
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


class RegisterAllocatorBlockNaive(RegisterAllocatorStrategy):
    idx: int

    def __init__(self) -> None:
        self.idx = 0

        """
        Since we've got neither right now a handling of a consistent ABI nor of a calling convention,
        let's just assume that we have all the registers available for our use except the one explicitly reserved by the default riscv ABI.
        """

        self.available_registers = set(Register.ABI_INDEX_BY_NAME.keys())
        reserved_registers = set(["zero", "sp", "gp", "tp", "fp", "s0"])
        self.available_registers -= reserved_registers

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Sets unallocated registers for each block to a finite set of real available registers.
        When it runs out of real registers for a block, it allocates j registers.
        """

        for region in module.regions:
            for block in region.blocks:
                block_registers = self.available_registers.copy()

                for op in block.walk():
                    if not isinstance(op, RISCVOp):
                        # Don't perform register allocations on non-RISCV-ops
                        continue

                    for result in op.results:
                        assert isinstance(result.typ, RegisterType)
                        if result.typ.data.name is None:
                            # If we run out of real registers, allocate a j register
                            if block_registers == set():
                                result.typ = RegisterType(Register(f"j{self.idx}"))
                                self.idx += 1
                            else:
                                result.typ = RegisterType(
                                    Register(block_registers.pop())
                                )


class RegisterAllocatorJRegs(RegisterAllocatorStrategy):
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
                assert isinstance(result.typ, RegisterType)
                if result.typ.data.name is None:
                    result.typ = RegisterType(Register(f"j{self.idx}"))
                    self.idx += 1


class RegisterAllocationAlgorithm(Enum):
    """
    The register allocation algorithm to use.
    """

    GlobalJRegs = 0
    BlockNaive = 1


class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module.
    """

    name = "riscv-allocate-registers"

    def __init__(
        self,
        allocation_type: RegisterAllocationAlgorithm = RegisterAllocationAlgorithm.GlobalJRegs,
    ) -> None:
        self.allocation_type = allocation_type

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        match self.allocation_type:
            case RegisterAllocationAlgorithm.GlobalJRegs:
                allocator = RegisterAllocatorJRegs()
            case RegisterAllocationAlgorithm.BlockNaive:
                allocator = RegisterAllocatorBlockNaive()

        allocator.allocate_registers(op)
