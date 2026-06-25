import json
from collections import defaultdict
from collections.abc import Iterable

from xdsl.backend.block_naive_allocator import BlockNaiveAllocator
from xdsl.backend.register_allocatable import RegisterAllocatableOperation
from xdsl.backend.register_allocator import live_ins_per_block
from xdsl.backend.register_stack import RegisterStack
from xdsl.backend.register_type import RegisterType
from xdsl.dialects import riscv, riscv_func
from xdsl.dialects.riscv import Registers, RISCVRegisterType
from xdsl.ir import SSAValue
from xdsl.ir.post_order import PostOrderIterator
from xdsl.rewriter import InsertPoint, Rewriter
from xdsl.transforms.canonicalization_patterns.riscv import get_constant_value


def reg_types_by_name(regs: Iterable[RegisterType]) -> dict[str, set[str]]:
    """
    Groups register types by name.
    """
    res = defaultdict[str, set[str]](set)
    for reg in regs:
        res[reg.name].add(reg.register_name.data)
    return res


class RegisterAllocatorLivenessBlockNaive(BlockNaiveAllocator):
    def __init__(self, available_registers: RegisterStack) -> None:
        super().__init__(available_registers, RISCVRegisterType)

    def new_type_for_value(self, reg: SSAValue) -> RegisterType | None:
        if (
            isinstance(reg.type, self.register_base_class)
            and not reg.type.is_allocated
            and (val := get_constant_value(reg)) is not None
            and val.value.data == 0
        ):
            return Registers.ZERO
        return super().new_type_for_value(reg)

    def allocate_func(
        self, func: riscv_func.FuncOp, *, add_regalloc_stats: bool = False
    ) -> None:
        """
        Allocates values in function passed in to registers.
        The whole function must have been lowered to the relevant riscv dialects
        and it must contain no unrealized casts.
        If `add_regalloc_stats` is set to `True`, then a comment op will be inserted
        before the function op passed in with a json containing the relevant data.
        """
        if not func.body.blocks:
            # External function declaration
            return

        preallocated = RegisterAllocatableOperation.all_used_registers(func.body)
        excluded = RegisterAllocatableOperation.all_excluded_registers(func.body)

        for pa_reg in preallocated | excluded:
            self.available_registers.exclude_register(pa_reg)

        for block in PostOrderIterator(func.body.blocks[0]):
            print(block.name_hint)

            self.live_ins_per_block = live_ins_per_block(block)
            assert not self.live_ins_per_block[block]

            if add_regalloc_stats:
                preallocated_stats = reg_types_by_name(preallocated)
                excluded_stats = reg_types_by_name(excluded)
                allocated_stats = reg_types_by_name(
                    val.type
                    for op in block.walk()
                    for vals in (op.results, op.operands)
                    for val in vals
                    if isinstance(val.type, RISCVRegisterType)
                )
                stats = {
                    "preallocated_float": sorted(preallocated_stats["riscv.freg"]),
                    "preallocated_int": sorted(preallocated_stats["riscv.reg"]),
                    "excluded_float": sorted(excluded_stats["riscv.freg"]),
                    "excluded_int": sorted(excluded_stats["riscv.reg"]),
                    "allocated_float": sorted(allocated_stats["riscv.freg"]),
                    "allocated_int": sorted(allocated_stats["riscv.reg"]),
                }

                stats_str = json.dumps(stats)
                Rewriter.insert_op(
                    riscv.CommentOp(f"Regalloc stats: {stats_str}"),
                    InsertPoint.before(func),
                )

            self.allocate_block(block)

            for arg in block.args:
                self.free_value(arg)
