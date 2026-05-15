from dataclasses import dataclass

from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_stack import OutOfRegisters
from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_func
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter


@dataclass(frozen=True)
class RISCVAllocateInfiniteRegistersPass(ModulePass):
    """
    Allocates infinite registers to physical registers in the module.

    Assumes ParallelMovOps are legalized.
    """

    name = "riscv-allocate-infinite-registers"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func_op in (i for i in op.walk() if isinstance(i, riscv_func.FuncOp)):
            register_stack = RiscvRegisterStack.get()

            # remove registers from stack that are already used in body
            # up till the first parallel move op
            for inner_op in func_op.walk():
                if isinstance(inner_op, riscv.ParallelMovOp):
                    # only consider uses for pmov
                    for reg in inner_op.operand_types:
                        if (
                            isinstance(reg, riscv.RISCVRegisterType)
                            and reg.is_allocated
                        ):
                            register_stack.exclude_register(reg)
                    break

                if isinstance(inner_op, RegisterAllocatableOperation):
                    for reg in inner_op.iter_used_registers():
                        register_stack.exclude_register(reg)

            phys_reg_by_inf_reg: dict[
                riscv.RISCVRegisterType, riscv.RISCVRegisterType
            ] = {}
            for inner_op in func_op.walk():
                if isinstance(inner_op, riscv.ParallelMovOp):
                    # Reset the register stack and search until the next pmov op
                    register_stack = RiscvRegisterStack.get()
                    # exclude allocated definitions
                    for reg in inner_op.result_types:
                        if (
                            isinstance(reg, riscv.RISCVRegisterType)
                            and reg.is_allocated
                        ):
                            register_stack.exclude_register(reg)
                    # remove registers from stack that are already used in body
                    # up till the next parallel move op
                    next_op = inner_op
                    while (next_op := next_op.next_op) is not None:
                        if isinstance(next_op, riscv.ParallelMovOp):
                            # only consider uses for pmov
                            for reg in next_op.operand_types:
                                if (
                                    isinstance(reg, riscv.RISCVRegisterType)
                                    and reg.is_allocated
                                ):
                                    register_stack.exclude_register(reg)
                            break

                        if isinstance(next_op, RegisterAllocatableOperation):
                            for reg in next_op.iter_used_registers():
                                register_stack.exclude_register(reg)
                    # Forward register mapping through blocks, but drop mapping
                    # if the phys reg has been excluded
                    for inf_reg, phys_reg in list(phys_reg_by_inf_reg.items()):
                        assert isinstance(phys_reg.index, builtin.IntAttr)
                        index = phys_reg.index.data
                        if (
                            index
                            in register_stack.available_registers[
                                phys_reg.register_pool_key()
                            ]
                        ):
                            # This mapping is safe, keep the mapping and pop the value
                            register_stack.exclude_register(phys_reg)
                        else:
                            # This mapping is no longer valid, drop the mapping
                            del phys_reg_by_inf_reg[inf_reg]

                for result in inner_op.results:
                    result_reg = result.type
                    if not isinstance(result_reg, riscv.RISCVRegisterType):
                        # Ignore non-register types
                        continue

                    if (
                        isinstance(result_reg.index, builtin.IntAttr)
                        and result_reg.index.data < 0
                    ):
                        if result_reg in phys_reg_by_inf_reg:
                            # use previously allocated phys reg for this value
                            phys_reg = phys_reg_by_inf_reg[result_reg]
                        else:
                            # allocate a new phys reg
                            phys_reg = register_stack.pop(type(result_reg))
                            phys_reg_by_inf_reg[result_reg] = phys_reg

                        Rewriter.replace_value_with_new_type(result, phys_reg)

                if isinstance(inner_op, riscv.ParallelMovOp):
                    free_regs = []
                    reg_stack = RiscvRegisterStack.get()
                    # The free registers are any registers not used by pmov
                    # because no live ranges will cross a pmov
                    for pmov_used_reg in inner_op.iter_used_registers():
                        reg_stack.exclude_register(pmov_used_reg)
                    for reg_type in (riscv.IntRegisterType, riscv.FloatRegisterType):
                        while True:
                            try:
                                free_regs.append(reg_stack.pop(reg_type))
                            except OutOfRegisters:
                                break
                    # Add free_registers to op
                    Rewriter.replace_op(
                        inner_op,
                        [
                            riscv.ParallelMovOp(
                                inner_op.inputs,
                                inner_op.result_types,
                                inner_op.input_widths,
                                free_registers=builtin.ArrayAttr(free_regs),
                            )
                        ],
                    )
