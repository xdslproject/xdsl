from dataclasses import dataclass

from xdsl.backend.riscv.register_stack import RiscvRegisterStack
from xdsl.context import Context
from xdsl.dialects import builtin, riscv, riscv_func
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter
from xdsl.utils.exceptions import PassFailedException


@dataclass(frozen=True)
class RISCVAllocateInfiniteRegistersPass(ModulePass):
    """
    Allocates infinite registers to physical registers in the module.
    """

    name = "riscv-allocate-infinite-registers"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for func_op in (i for i in op.walk() if isinstance(i, riscv_func.FuncOp)):
            register_stack = RiscvRegisterStack.get()
            phys_reg_by_inf_reg: dict[
                riscv.RISCVRegisterType, riscv.RISCVRegisterType
            ] = {}
            for inner_op in func_op.walk():
                for result in inner_op.results:
                    result_reg = result.type
                    if not isinstance(result_reg, riscv.RISCVRegisterType):
                        raise PassFailedException("Operand type not a register")

                    if (
                        isinstance(result_reg.index, builtin.IntAttr)
                        and result_reg.index.data < 0
                    ):
                        # We are an infinite reg
                        if result_reg in phys_reg_by_inf_reg:
                            # use previously allocated phys reg for this value
                            phys_reg = phys_reg_by_inf_reg[result_reg]
                        else:
                            # allocate a new phys reg
                            phys_reg = register_stack.pop(type(result_reg))
                            phys_reg_by_inf_reg[result_reg] = phys_reg

                        Rewriter.replace_value_with_new_type(result, phys_reg)
