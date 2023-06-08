from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, List, Set
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import Block, MLContext, OpResult, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import InvalidIRException


class RegisterSet:
    def __init__(self, registers: List[str]) -> None:
        self.registers = registers
        self.free: List[str] = list(registers)
        self.occupied: Set[str] = set()

    def get_free(self) -> str | None:
        if not self.free:
            return None

        reg = self.free.pop()
        self.occupied.add(reg)
        return reg

    def set_free(self, reg: str) -> None:
        self.occupied.remove(reg)
        self.free.append(reg)

    def set_occupied(self, reg: str) -> None:
        assert reg in self.free
        self.free.remove(reg)
        self.occupied.add(reg)

    def is_free(self, reg: str) -> bool:
        return reg in self.free

    def reset(self) -> None:
        self.free = list(self.registers)
        self.occupied = set()

    def num_free_registers(self) -> int:
        return len(self.free)

    def has_available_registers(self) -> bool:
        return not self.free

    def limit_free_registers(self, limit: int) -> None:
        self.free = self.free[:limit]
        self.occupied = set(self.registers) - set(self.free)


_DEFAULT_RESERVED_REGISTERS = set(["zero", "sp", "gp", "tp", "fp", "s0"])
_DEFAULT_REGISTER_SET = RegisterSet(
    [
        reg
        for reg in list(Register.ABI_INDEX_BY_NAME.keys())
        if reg not in _DEFAULT_RESERVED_REGISTERS
    ]
)


class AbstractRegisterAllocator(ABC):
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


class RegisterAllocatorBlockNaive(AbstractRegisterAllocator):
    idx: int
    register_set: RegisterSet

    def __init__(self, register_set: RegisterSet = _DEFAULT_REGISTER_SET) -> None:
        self.idx = 0
        self.register_set = register_set

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Sets unallocated registers for each block to a finite set of real available registers.
        When it runs out of real registers for a block, it allocates j registers.
        """

        for region in module.regions:
            for block in region.blocks:
                self.register_set.reset()
                for op in block.walk():
                    if not isinstance(op, RISCVOp):
                        # Don't perform register allocations on non-RISCV-ops
                        continue

                    for result in op.results:
                        assert isinstance(result.typ, RegisterType)
                        if result.typ.data.name is None:
                            # If we run out of real registers, allocate a j register
                            if self.register_set.has_available_registers():
                                result.typ = RegisterType(Register(f"j{self.idx}"))
                                self.idx += 1
                            else:
                                result.typ = RegisterType(
                                    Register(self.register_set.get_free())
                                )


class RegisterAllocatorJRegs(AbstractRegisterAllocator):
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


class LiveInterval:
    ssa_value: SSAValue
    register: Register | None
    start: int
    abstract_stack_location: int | None
    end: int

    def __init__(self, ssa_value: SSAValue, start: int, end: int) -> None:
        self.ssa_value = ssa_value

        if (
            isinstance(ssa_value.typ, RegisterType)
            and ssa_value.typ.data.name is not None
        ):
            self.register = ssa_value.typ.data
        else:
            self.register = None

        self.abstract_stack_location = None

        self.start = start
        self.end = end

    def regalloc(self, register: Register) -> None:
        self.abstract_stack_location = None
        self.register = register

    def spill(self, stack_location: int) -> None:
        self.register = None
        self.abstract_stack_location = stack_location

    def allocated(self) -> RegisterType | None:
        assert isinstance(self.ssa_value.typ, RegisterType)
        return self.ssa_value.typ

    def __repr__(self) -> str:
        return f"LiveInterval({self.ssa_value}, {self.start}, {self.end}) -> Register: {self.register}, Stack Location: {self.abstract_stack_location}"


class RegisterAllocatorLinearScan(AbstractRegisterAllocator):
    """
    Linear scan register allocation strategy.

    Basic form of the algorithm based on the paper "Linear Scan Register Allocation"
    by Massimiliano Poletto and Vivek Sarkar.
    """

    abstract_stack_location: int
    intervals: list[LiveInterval]
    active: OrderedDict[LiveInterval, None]
    register_set: RegisterSet

    def __init__(
        self,
        intervals: list[LiveInterval],
        register_set: RegisterSet = _DEFAULT_REGISTER_SET,
    ) -> None:
        self.active = OrderedDict()
        self.register_set = register_set
        self.intervals = intervals
        self.abstract_stack_location = 0

    # TO:DO - Refactor the following methods to use a proper SortedSet (C++ fashion)
    # This requires an extra dependency, so right now stick with this subpar implementation

    ###

    def insert_active_interval(self, interval: LiveInterval) -> None:
        self.active[interval] = None
        self.sort_active_intervals()

    def remove_active_interval(self, interval: LiveInterval) -> None:
        self.active.pop(interval)
        self.sort_active_intervals()

    def sort_active_intervals(self) -> None:
        self.active = OrderedDict(sorted(self.active.items(), key=lambda x: x[0].end))

    ###

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Allocates unallocated registers in the module.
        """

        self.intervals = sorted(self.intervals, key=lambda x: x.start)

        # already allocated registers are removed from the available registers set (conservative approach)
        for interval in self.intervals:
            if interval.register is not None and interval.register.name is not None:
                if self.register_set.is_free(interval.register.name):
                    self.register_set.set_occupied(interval.register.name)

        def expire_old_intervals(i: LiveInterval) -> None:
            for j in list(filter(lambda j: j.end < i.start, self.active.keys())):
                assert j.register is not None
                assert j.register.name is not None

                self.remove_active_interval(j)
                self.register_set.set_free(j.register.name)

        def fresh_stack_location() -> int:
            old_stack_location = self.abstract_stack_location
            self.abstract_stack_location += 1
            return old_stack_location

        def spill_at_interval(i: LiveInterval) -> None:
            # Spill the interval with the furthest endpoint
            spill = next(reversed(self.active.keys()))
            if spill.end > i.end:
                i.register = spill.register
                self.remove_active_interval(spill)
                spill.spill(fresh_stack_location())
                self.insert_active_interval(i)
            else:
                i.spill(fresh_stack_location())

        num_registers = self.register_set.num_free_registers()
        for iv in self.intervals:
            """
            Expire all intervals whose endpoint is smaller than the start point of the current interval being processed.
            This means that the current interval can be mapped to a register that was previously assigned
            to one of the expired intervals, which is no longer needed because it has expired.
            """

            expire_old_intervals(iv)
            if len(self.active) >= num_registers:
                spill_at_interval(iv)
            else:
                iv.regalloc(Register(self.register_set.get_free()))
                self.insert_active_interval(iv)

        # final pass to assign registers to intervals
        for iv in self.intervals:
            assert isinstance(iv.ssa_value.typ, RegisterType)

            # if the interval is not allocated to a register, it must be spilled
            if iv.register is None:
                assert iv.abstract_stack_location is not None
                # still use j-register for now
                iv.ssa_value.typ = RegisterType(
                    Register(f"j{iv.abstract_stack_location}")
                )
            elif iv.ssa_value.typ.data.name is None:
                iv.ssa_value.typ = RegisterType(iv.register)
            else:
                # do nothing, register is already assigned
                pass

        return


@dataclass
class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module.
    """

    name = "riscv-allocate-registers"

    allocation_strategy: str = "GlobalJRegs"

    def apply(self, ctx: MLContext, op: ModuleOp, *args: Any) -> None:
        allocator_strategies = {
            "GlobalJRegs": RegisterAllocatorJRegs,
            "BlockNaive": RegisterAllocatorBlockNaive,
            "LinearScan": RegisterAllocatorLinearScan,
        }

        if self.allocation_strategy not in allocator_strategies:
            raise ValueError(
                f"Unknown register allocation strategy {self.allocation_strategy}. "
                f"Available allocation types: {allocator_strategies.keys()}"
            )

        allocator = allocator_strategies[self.allocation_strategy](*args)
        allocator.allocate_registers(op)
