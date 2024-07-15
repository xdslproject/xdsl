import abc
from itertools import chain
from typing import cast

from ordered_set import OrderedSet

from xdsl.backend.register_type import RegisterType
from xdsl.backend.riscv.register_allocation_constraints import OpTieConstraints
from xdsl.backend.riscv.register_queue import RegisterQueue
from xdsl.dialects import riscv, riscv_func, riscv_scf, riscv_snitch
from xdsl.dialects.riscv import (
    FloatRegisterType,
    IntRegisterType,
    Registers,
    RISCVAsmOperation,
    RISCVRegisterType,
)
from xdsl.ir import Block, Operation, SSAValue
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value
from xdsl.transforms.snitch_register_allocation import get_snitch_reserved


def gather_allocated(func: riscv_func.FuncOp) -> set[RISCVRegisterType]:
    """Utility method to gather already allocated registers"""

    allocated: set[RISCVRegisterType] = set()

    for op in func.walk():
        if not isinstance(op, RISCVAsmOperation):
            continue

        if isinstance(op, riscv_func.CallOp):
            # These registers are not guaranteed to hold the same values when the callee
            # returns, according to the RISC-V calling convention.
            # https://riscv.org/wp-content/uploads/2015/01/riscv-calling.pdf
            allocated.update(riscv.Registers.A)
            allocated.update(riscv.Registers.T)
            allocated.update(riscv.Registers.FA)
            allocated.update(riscv.Registers.FT)

        for param in chain(op.operands, op.results):
            if isinstance(param.type, RISCVRegisterType) and param.type.is_allocated:
                if not param.type.register_name.startswith("j"):
                    allocated.add(param.type)

    return allocated


def _uses_snitch_stream(func: riscv_func.FuncOp) -> bool:
    """Utility method to detect use of read/write ops of the `snitch_stream` dialect."""

    return any(
        isinstance(op, riscv_snitch.ReadOp | riscv_snitch.WriteOp) for op in func.walk()
    )


class RegisterAllocator(abc.ABC):
    """
    Base class for register allocation strategies.
    """

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

    available_registers: RegisterQueue
    live_ins_per_block: dict[Block, OrderedSet[SSAValue]]

    exclude_preallocated: bool = True
    exclude_snitch_reserved: bool = True

    def __init__(self) -> None:
        self.available_registers = RegisterQueue()
        self.live_ins_per_block = {}

    def allocate(self, reg: SSAValue) -> bool:
        """
        Allocate a register if not already allocated.
        """
        if (
            isinstance(reg.type, IntRegisterType | FloatRegisterType)
            and not reg.type.is_allocated
        ):
            if (val := get_constant_value(reg)) is not None and val.value.data == 0:
                reg.type = Registers.ZERO
            else:
                reg.type = self.available_registers.pop(type(reg.type))
            return True

        return False

    def _free(self, reg: SSAValue) -> None:
        if (
            isinstance(reg.type, IntRegisterType | FloatRegisterType)
            and reg.type.is_allocated
        ):
            self.available_registers.push(reg.type)

    def process_operation(self, op: Operation) -> None:
        """
        Allocate registers for one operation.
        """
        match op:
            case riscv_scf.ForOp():
                self.allocate_for_loop(op)
            case riscv_snitch.FRepOperation():
                self.allocate_frep_loop(op)
            case RISCVAsmOperation():
                self.process_riscv_op(op)
            case _:
                # Ignore non-riscv operations
                return

    def process_riscv_op(self, op: RISCVAsmOperation) -> None:
        """
        Allocate registers for RISC-V Instruction.
        """
        constr = OpTieConstraints.from_op(op)

        allocated_results: list[SSAValue] = []
        for idx, result in enumerate(op.results):
            if constr.result_has_constraints(idx):
                is_tied_to: RegisterType | None = constr.result_is_constrained_to(idx)
                if is_tied_to is None:
                    self.allocate(result)
                    # Mark the newly allocated register to be freed later.
                    # We cannot free it here to avoid having multiple oprand/results
                    # allocated to it: being tied, it could still be
                    # needed until we have allocated the whole operation
                    allocated_results.append(result)
                else:
                    # No need for allocation, we must use the register that
                    # satisfies the constraint:
                    result.type = is_tied_to
                # Mark the constraint as satisfied for the current result
                reg = cast(RISCVRegisterType, result.type)
                constr.result_satisfy_constraint(idx, reg)
            else:
                # Allocate registers to result if not already allocated
                self.allocate(result)
                allocated_results.append(result)

        # Allocate registers to operands since they are defined further up
        # in the use-def SSA chain
        for idx, operand in enumerate(op.operands):
            if constr.operand_has_constraints(idx):
                is_tied_to: RegisterType | None = constr.operand_is_constrained_to(idx)
                if is_tied_to is None:
                    self.allocate(operand)
                else:
                    operand.type = is_tied_to
                reg = cast(RISCVRegisterType, operand.type)
                constr.operand_satisfy_constraint(idx, reg)
            else:
                self.allocate(operand)

        for result in allocated_results:
            # Free the register since the SSA value is created here
            self._free(result)

    def allocate_for_loop(self, loop: riscv_scf.ForOp) -> None:
        """
        Allocate registers for riscv_scf for loop, recursively calling process_operation
        for operations in the loop.
        """
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = self.live_ins_per_block[loop.body.block]
        for live_in in live_ins:
            self.allocate(live_in)

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

            # Because we are walking backwards, the result of the operation may have been
            # allocated already. If it isn't it's because it's not used below.
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

        # Step and ub are used throughout loop
        self.allocate(loop.ub)
        self.allocate(loop.step)

        # Reserve the loop carried variables for allocation within the body
        for iter_arg in loop.iter_args:
            assert isinstance(iter_arg.type, IntRegisterType | FloatRegisterType)
            self.available_registers.reserve_register(iter_arg.type)

        for op in reversed(loop.body.block.ops):
            self.process_operation(op)

        # Unreserve the loop carried variables for allocation outside of the body
        for iter_arg in loop.iter_args:
            assert isinstance(iter_arg.type, IntRegisterType | FloatRegisterType)
            self.available_registers.unreserve_register(iter_arg.type)

        # lb is only used as an input to the loop, so free induction variable before
        # allocating lb to it in case it's not yet allocated
        self._free(block_args[0])
        self.allocate(loop.lb)

    def allocate_frep_loop(self, loop: riscv_snitch.FRepOperation) -> None:
        """
        Allocate registers for riscv_snitch frep_outer or frep_inner loop, recursively
        calling process_operation for operations in the loop.
        """
        # Allocate values used inside the body but defined outside.
        # Their scope lasts for the whole body execution scope
        live_ins = self.live_ins_per_block[loop.body.block]
        for live_in in live_ins:
            self.allocate(live_in)

        yield_op = loop.body.block.last_op
        assert yield_op is not None, (
            "last op of riscv_snitch.frep_outer and riscv_snitch.frep_inner is guaranteed"
            " to be riscv_scf.Yield"
        )
        block_args = loop.body.block.args

        # The loop-carried variables are trickier
        # The for op operand, block arg, and yield operand must have the same type
        for block_arg, operand, yield_operand, op_result in zip(
            block_args, loop.iter_args, yield_op.operands, loop.results
        ):
            # If some allocated then assign all to that type, otherwise get new reg
            assert isinstance(block_arg.type, RISCVRegisterType)
            assert isinstance(operand.type, RISCVRegisterType)
            assert isinstance(yield_operand.type, RISCVRegisterType)
            assert isinstance(op_result.type, RISCVRegisterType)

            # Because we are walking backwards, the result of the operation may have been
            # allocated already. If it isn't it's because it's not used below.
            if not op_result.type.is_allocated:
                # We only need to check one of the four since they're constrained to be
                # the same
                self.allocate(op_result)

            shared_type = op_result.type
            block_arg.type = shared_type
            yield_operand.type = shared_type
            operand.type = shared_type

        # Operands
        for operand in loop.operands:
            self.allocate(operand)

        # Reserve the loop carried variables for allocation within the body
        for iter_arg in loop.iter_args:
            assert isinstance(iter_arg.type, IntRegisterType | FloatRegisterType)
            self.available_registers.reserve_register(iter_arg.type)

        for op in reversed(loop.body.block.ops):
            self.process_operation(op)

        # Unreserve the loop carried variables for allocation outside of the body
        for iter_arg in loop.iter_args:
            assert isinstance(iter_arg.type, IntRegisterType | FloatRegisterType)
            self.available_registers.unreserve_register(iter_arg.type)

    def allocate_func(self, func: riscv_func.FuncOp) -> None:
        if not func.body.blocks:
            # External function declaration
            return

        if len(func.body.blocks) != 1:
            raise NotImplementedError(
                f"Cannot register allocate func with {len(func.body.blocks)} blocks."
            )

        preallocated: set[RISCVRegisterType] = set()

        if self.exclude_preallocated:
            preallocated |= gather_allocated(func)

        if self.exclude_snitch_reserved and _uses_snitch_stream(func):
            preallocated |= get_snitch_reserved()

        for pa_reg in preallocated:
            if isinstance(pa_reg, IntRegisterType | FloatRegisterType):
                self.available_registers.reserve_register(pa_reg)

            if pa_reg in self.available_registers.available_int_registers:
                self.available_registers.available_int_registers.remove(pa_reg)
            if pa_reg in self.available_registers.available_float_registers:
                self.available_registers.available_float_registers.remove(pa_reg)

        block = func.body.block

        self.live_ins_per_block = live_ins_per_block(block)
        assert not self.live_ins_per_block[block]
        for op in reversed(block.ops):
            self.process_operation(op)


def _live_ins_per_block(
    block: Block, acc: dict[Block, OrderedSet[SSAValue]]
) -> OrderedSet[SSAValue]:
    res = OrderedSet[SSAValue]([])

    for op in reversed(block.ops):
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


def live_ins_per_block(block: Block) -> dict[Block, OrderedSet[SSAValue]]:
    """
    Returns a mapping from a block to the set of values used in it but defined outside of
    it.
    """
    res: dict[Block, OrderedSet[SSAValue]] = {}
    _ = _live_ins_per_block(block, res)
    return res
