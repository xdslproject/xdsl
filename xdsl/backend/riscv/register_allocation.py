import abc

from xdsl.backend.riscv.register_queue import RegisterQueue
from xdsl.dialects import riscv_func, riscv_scf
from xdsl.dialects.riscv import (
    FloatRegisterType,
    IntRegisterType,
    RISCVOp,
    RISCVRegisterType,
)
from xdsl.ir import Operation, SSAValue
from xdsl.ir.core import Block


class RegisterAllocator(abc.ABC):
    """
    Base class for register allocation strategies.
    """

    @abc.abstractmethod
    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        raise NotImplementedError()


class RegisterAllocatorBlockNaive(RegisterAllocator):
    """
    Sets unallocated registers per block to a finite set of real available registers.
    When it runs out of real registers for a block, it allocates j registers.
    """

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
        self.live_ins_per_block = {}

    def allocate(self, reg: SSAValue) -> bool:
        """
        Allocate a register if not already allocated.
        """
        if (
            isinstance(reg.type, IntRegisterType | FloatRegisterType)
            and not reg.type.is_allocated
        ):
            reg.type = self.available_registers.pop(type(reg.type))
            return True

        return False

    def process_operation(self, op: Operation) -> None:
        """
        Allocate registers for one operation.
        """
        match op:
            case riscv_scf.ForOp():
                self.allocate_for_loop(op)
            case RISCVOp():
                self.process_riscv_op(op)
            case _:
                # Ignore non-riscv operations
                return

    def process_riscv_op(self, op: RISCVOp) -> None:
        """
        Allocate registers for RISC-V Instruction.
        """
        for result in op.results:
            self.allocate(result)

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        if len(func.body.blocks) != 1:
            raise NotImplementedError(
                f"Cannot register allocate func with {len(func.body.blocks)} blocks."
            )

        block = func.body.block

        self.live_ins_per_block = live_ins_per_block(block)
        assert not self.live_ins_per_block[block]
        for op in block.ops_reverse:
            self.process_operation(op)

    def allocate_for_loop(self, loop: riscv_scf.ForOp) -> None:
        """
        Allocate registers for riscv_scf for loop, recursively calling process_operation
        for operations in the loop.
        """
        yield_op = loop.body.block.last_op
        assert (
            yield_op is not None
        ), "last op of riscv_scf.ForOp is guaranteed to be riscv_scf.Yield"
        block_args = loop.body.block.args

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

            if not op_result.type.is_allocated:
                # We only need to check one of the four since they're constrained to be
                # the same
                self.allocate(op_result)

            shared_type = op_result.type
            block_arg.type = shared_type
            yield_operand.type = shared_type
            operand.type = shared_type

        # Induction variable
        assert isinstance(block_args[0].type, IntRegisterType)
        self.allocate(block_args[0])

        # Operands
        for operand in loop.operands:
            self.allocate(operand)

        for op in loop.body.block.ops_reverse:
            self.process_operation(op)


class RegisterAllocatorLivenessBlockNaive(RegisterAllocatorBlockNaive):
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

    live_ins_per_block: dict[Block, set[SSAValue]]

    def _free(self, reg: SSAValue) -> None:
        if (
            isinstance(reg.type, IntRegisterType | FloatRegisterType)
            and reg.type.is_allocated
        ):
            self.available_registers.push(reg.type)

    def process_riscv_op(self, op: RISCVOp) -> None:
        for result in op.results:
            # Allocate registers to result if not already allocated
            self.allocate(result)
            # Free the register since the SSA value is created here
            self._free(result)

        super().process_riscv_op(op)

    def allocate_for_loop(self, loop: riscv_scf.ForOp) -> None:
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = self.live_ins_per_block[loop.body.block]
        for live_in in live_ins:
            self.allocate(live_in)

        super().allocate_for_loop(loop)

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        if len(func.body.blocks) != 1:
            raise NotImplementedError(
                f"Cannot register allocate func with {len(func.body.blocks)} blocks."
            )

        block = func.body.block

        self.live_ins_per_block = live_ins_per_block(block)
        assert not self.live_ins_per_block[block]

        super().allocate_func(func)


def _live_ins_per_block(block: Block, acc: dict[Block, set[SSAValue]]) -> set[SSAValue]:
    res = set[SSAValue]()

    for op in block.ops_reverse:
        # Remove values defined in the block
        # We are traversing backwards, so cannot use the value removed here again
        res.difference_update(op.results)
        # Add values used in the block
        res.update(op.operands)

        # Process inner blocks
        for region in op.regions:
            for inner in region.blocks:
                # Add the values used in the inner block
                res.update(_live_ins_per_block(inner, acc))

    # Remove the block arguments
    res.difference_update(block.args)

    acc[block] = res

    return res


def live_ins_per_block(block: Block) -> dict[Block, set[SSAValue]]:
    """
    Returns a mapping from a block to the set of values used in it but defined outside of
    it.
    """
    res: dict[Block, set[SSAValue]] = {}
    _ = _live_ins_per_block(block, res)
    return res
