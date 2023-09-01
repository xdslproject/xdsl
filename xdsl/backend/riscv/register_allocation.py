import abc

from xdsl.backend.riscv.register_queue import RegisterQueue
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

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        raise NotImplementedError()


class BaseBlockNaiveRegisterAllocator(RegisterAllocator, abc.ABC):
    available_registers: RegisterQueue

    def __init__(self) -> None:
        self.available_registers = RegisterQueue(
            available_int_registers=[
                IntRegisterType(reg)
                for reg in IntRegisterType.RV32I_INDEX_BY_NAME
                if IntRegisterType(reg) not in RegisterQueue.DEFAULT_RESERVED_REGISTERS
            ],
            available_float_registers=[
                FloatRegisterType(reg) for reg in FloatRegisterType.RV32F_INDEX_BY_NAME
            ],
        )

    def allocate(self, reg: SSAValue) -> bool:
        if (
            isinstance(reg.type, IntRegisterType | FloatRegisterType)
            and not reg.type.is_allocated
        ):
            reg.type = self.available_registers.pop(type(reg.type))
            return True

        return False


class RegisterAllocatorLivenessBlockNaive(BaseBlockNaiveRegisterAllocator):
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
                        self.allocate(operand)

                    # Allocate registers to results if not already allocated,
                    # otherwise free that register since the SSA value is created here
                    for result in op.results:
                        # Unallocated results still need a register,
                        # so allocate and keep track of them to be freed
                        # before processing the next instruction
                        self.allocate(result)
                        to_free.append(result)


class RegisterAllocatorBlockNaive(BaseBlockNaiveRegisterAllocator):
    def allocate_for_loop(self, loop: riscv_scf.ForOp) -> None:
        yield_op = loop.body.block.last_op
        assert (
            yield_op is not None
        ), "last op of riscv_scf.ForOp is guaranteed to be riscv_scf.Yield"
        block_args = loop.body.block.args

        # Induction variable
        assert isinstance(block_args[0].type, IntRegisterType)
        self.allocate(block_args[0])

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for i, (block_arg, operand, yield_operand, op_result) in enumerate(
            zip(block_args[1:], loop.iter_args, yield_op.operands, loop.results)
        ):
            # If some allocated then assign all to that type, otherwise get new reg
            assert isinstance(block_arg.type, RISCVRegisterType)
            assert isinstance(operand.type, RISCVRegisterType)
            assert isinstance(yield_operand.type, RISCVRegisterType)
            assert isinstance(op_result.type, RISCVRegisterType)

            shared_type: RISCVRegisterType | None = None
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
                # arbitrarily pick one of the values to allocate first
                self.allocate(block_arg)
                shared_type = block_arg.type
            else:
                block_arg.type = shared_type

            operand.type = shared_type
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
                        self.allocate(result)
