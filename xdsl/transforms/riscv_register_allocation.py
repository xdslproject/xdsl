from enum import Enum
from xdsl.dialects import riscv
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class RegisterAllocatorStrategy:
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

        self.available_registers = set(Register.ABI_INDEX_BY_NAME.keys())
        reserved_registers = set(["zero", "sp", "gp", "tp", "fp"])
        caller_saved_registers = set(
            [
                "ra",
                "a0",
                "a1",
                "a2",
                "a3",
                "a4",
                "a5",
                "a6",
                "a7",
                "t0",
                "t1",
                "t2",
                "t3",
                "t4",
                "t5",
                "t6",
            ]
        )

        self.available_registers -= reserved_registers
        self.available_registers -= caller_saved_registers

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Currently sets the first N SSA values of each block
        to available registers considering the ABI and the rest to an infinite set of `j` registers.
        """

        # hardcode the ABI for now, this should be configurable in the future
        # for instance, we might not have a frame pointer and thus have more registers available

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
                            if block_registers == set():
                                result.typ = RegisterType(Register(f"j{self.idx}"))
                                self.idx += 1
                            else:
                                result.typ = RegisterType(
                                    Register(block_registers.pop())
                                )


class RegisterAllocatorBlockNaiveSpill(RegisterAllocatorBlockNaive):
    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Same as RegisterAllocatorBlockNaive, but spills to stack if there are no registers left.
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
                            if block_registers == set():
                                # spill to stack
                                blk = op.parent_block()
                                if blk is not None:
                                    # blk.insert_ops_after([], op)
                                    raise NotImplementedError()

                                for use in result.uses:
                                    # reload from stack
                                    blk = use.operation.parent_block()
                                    if blk is not None:
                                        # blk.insert_ops_before([], use.operation)
                                        raise NotImplementedError()

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
        Currently sets them to an infinite set of `j` registers.
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


class RegisterAllocationType(Enum):
    GlobalJRegs = 0
    BlockNaiveSSA = 1


class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module. Currently sets them to an infinite set
    of `j` registers.
    """

    name = "riscv-allocate-registers"

    def __init__(
        self,
        allocation_type: RegisterAllocationType = RegisterAllocationType.GlobalJRegs,
    ) -> None:
        self.allocation_type = allocation_type

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        match self.allocation_type:
            case RegisterAllocationType.GlobalJRegs:
                allocator = RegisterAllocatorJRegs()
            case RegisterAllocationType.BlockNaiveSSA:
                allocator = RegisterAllocatorBlockNaive()

        allocator.allocate_registers(op)
