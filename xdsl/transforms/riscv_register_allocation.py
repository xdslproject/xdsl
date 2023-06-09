from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence
from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.riscv import Register, RegisterType, RISCVOp
from xdsl.ir import MLContext, SSAValue
from xdsl.passes import ModulePass
from xdsl.transforms.experimental.live_range import LiveRange


class RegisterSet:
    # The complete set of registers
    registers: Sequence[str]
    # Registers that are free to be allocated
    free: list[str]
    # Registers that are currently used
    occupied: set[str]

    def __init__(self, registers: Sequence[str], reserved: Sequence[str] = []) -> None:
        self.registers = registers
        self.free: list[str] = list(registers)
        self.occupied: set[str] = set(reserved)

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
        self.free.remove(reg)
        self.occupied.add(reg)

    def is_free(self, reg: str) -> bool:
        return reg in self.free

    def reset(self) -> None:
        """
        Resets the register allocator to its initial state.
        """

        self.free = list(self.registers)
        self.occupied = set()

    def num_free_registers(self) -> int:
        return len(self.free)

    def has_available_registers(self) -> bool:
        return not self.free

    def limit_free_registers(self, limit: int) -> None:
        """
        Restricts the number of free registers available to the register allocator
        to the first `limit` registers.
        Useful for testing spill code.
        """

        self.free = self.free[:limit]
        self.occupied = set(self.registers) - set(self.free)


_DEFAULT_RESERVED_REGISTERS = list(["zero", "sp", "gp", "tp", "fp", "s0"])
_DEFAULT_REGISTER_SET = RegisterSet(
    [
        reg
        for reg in list(Register.ABI_INDEX_BY_NAME.keys())
        if reg not in _DEFAULT_RESERVED_REGISTERS
    ],
    _DEFAULT_RESERVED_REGISTERS,
)


class AbstractRegisterAllocator(ABC):
    """
    Base class for register allocation strategies.
    """

    def __init__(self) -> None:
        pass

    def limit_free_registers(self, limit: int) -> None:
        """
        Limits the number of free registers available to the register allocator.
        """

        raise NotImplementedError()

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

    def limit_free_registers(self, limit: int) -> None:
        self.register_set.limit_free_registers(limit)

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

    def limit_free_registers(self, limit: int) -> None:
        pass

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


class RegisterLiveInterval(LiveRange):
    """
    Represents an enhanced live range with additional information for register allocation.
    """

    # This is a value that is used to represents the stack location of a spilled value (not in a register)
    # Right now, we use j-regs to represent spilled values so it's a trivial mapping.
    # At this point, we might not care about various ABI conventions for the stack (e.g. alignment)
    # Thus, we can just use an index and concretize it later.
    abstract_stack_location: int | None

    # Indicates whether this register is already used somewhere in the module (e.g. by a callee) or
    # handallocated by the user. Thus, we should not modify it or spill it because it could have
    # side effects.
    used: bool

    def __init__(
        self,
        value: SSAValue,
        start: int | None = None,
        end: int | None = None,
        used: bool = False,
    ) -> None:
        super().__init__(value, start, end)
        self.abstract_stack_location = None
        self.used = used

    def spill(self, stack_location: int) -> None:
        self.abstract_stack_location = stack_location
        self.value.typ = RegisterType(Register(f"j{stack_location}"))

    def set_riscv_register(self, register: Register) -> None:
        self.abstract_stack_location = None
        self.value.typ = RegisterType(register)

    def get_riscv_register(self) -> RegisterType:
        assert isinstance(self.value.typ, RegisterType)
        return self.value.typ

    def set_used(self, used: bool) -> None:
        self.used = used


class RegisterAllocatorLinearScan(AbstractRegisterAllocator):
    """
    Linear scan register allocation strategy.

    Basic form of the algorithm based on "Massimiliano Poletto and Vivek Sarkar. 1999.
    Linear scan register allocation. ACM Trans. Program. Lang. Syst.
    21, 5 (Sept. 1999), 895–913. https://doi.org/10.1145/330249.330250", directly
    available at https://web.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf.

    """

    idx_abstract_stack_location: int
    intervals: list[RegisterLiveInterval]
    active: OrderedDict[RegisterLiveInterval, None]
    register_set: RegisterSet

    def __init__(
        self,
        intervals: list[RegisterLiveInterval] | None = None,
        register_set: RegisterSet = _DEFAULT_REGISTER_SET,
    ) -> None:
        self.active = OrderedDict()
        self.register_set = register_set
        self.intervals = intervals or []
        self.idx_abstract_stack_location = 0

    # TODO: - Refactor the following methods to use a proper SortedSet (C++ fashion)
    # This requires an extra dependency, so right now stick with this subpar implementation

    ###
    def insert_active_interval(self, interval: RegisterLiveInterval) -> None:
        self.active[interval] = None
        self.active = OrderedDict(sorted(self.active.items(), key=lambda x: x[0].end))

    def remove_active_interval(self, interval: RegisterLiveInterval) -> None:
        self.active.pop(interval)
        self.active = OrderedDict(sorted(self.active.items(), key=lambda x: x[0].end))

    ###

    def fresh_stack_location(self) -> int:
        old_stack_location = self.idx_abstract_stack_location
        self.idx_abstract_stack_location += 1
        return old_stack_location

    def expire_old_intervals(self, i: RegisterLiveInterval) -> None:
        for j in list(filter(lambda j: j.end < i.start, self.active.keys())):
            register = j.get_riscv_register()
            if register.data.name is not None and not j.used:
                self.remove_active_interval(j)
                self.register_set.set_free(register.data.name)

    def spill_at_interval(self, i: RegisterLiveInterval) -> None:
        # Spill the interval with the furthest endpoint
        spill = next(reversed(self.active.keys()))
        if spill.end > i.end and not spill.used:
            i.set_riscv_register(spill.get_riscv_register().data)
            self.remove_active_interval(spill)
            spill.spill(self.fresh_stack_location())
            self.insert_active_interval(i)
        else:
            i.spill(self.fresh_stack_location())

    def prepare_intervals(self, module: ModuleOp) -> None:
        # Build all live intervals if not already provided
        if not self.intervals:
            for op in module.walk():
                if isinstance(op, RISCVOp):
                    for result in op.results:
                        self.intervals.append(RegisterLiveInterval(result))

        # The algorithm requires values intervals to be sorted by their start point
        self.intervals = sorted(self.intervals, key=lambda x: x.start)

        # Already allocated registers are removed from the set of free registers
        for interval in self.intervals:
            register = interval.get_riscv_register()
            if register.data.name is not None:
                if self.register_set.is_free(register.data.name):
                    self.register_set.set_occupied(register.data.name)
                interval.set_used(True)

    def limit_free_registers(self, limit: int) -> None:
        self.register_set.limit_free_registers(limit)

    def allocate_registers(self, module: ModuleOp) -> None:
        """
        Allocates unallocated registers in the module.
        """

        # Prepare intervals
        self.prepare_intervals(module)

        num_registers = self.register_set.num_free_registers()

        for iv in self.intervals:
            """
            Expire all intervals whose endpoint is smaller than the start point of the current interval being processed.
            This means that the current interval can be mapped to a register that was previously assigned
            to one of the expired intervals, which is no longer needed because it has expired.
            """

            self.expire_old_intervals(iv)

            if len(self.active) >= num_registers:
                self.spill_at_interval(iv)
            else:
                if not iv.used:
                    iv.set_riscv_register(Register(self.register_set.get_free()))
                self.insert_active_interval(iv)

        return


@dataclass
class RISCVRegisterAllocation(ModulePass):
    """
    Allocates unallocated registers in the module.
    """

    name = "riscv-allocate-registers"

    allocation_strategy: str = "GlobalJRegs"
    num_registers: int | None = None

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
        if self.num_registers is not None:
            allocator.limit_free_registers(self.num_registers)
        allocator.allocate_registers(op)
