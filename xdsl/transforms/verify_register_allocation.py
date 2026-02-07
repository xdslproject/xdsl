from dataclasses import dataclass

from xdsl.backend.register_allocatable import HasRegisterConstraints
from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.ir import Region, SSAValue
from xdsl.ir.post_order import PostOrderIterator
from xdsl.passes import ModulePass
from xdsl.utils.exceptions import VerifyException


@dataclass(frozen=True)
class VerifyRegisterAllocationPass(ModulePass):
    """
    Verify that, assuming the dominance property is respected, the
    use of a register value as inout is its last use.
    """

    name = "verify-register-allocation"

    def _process_region(self, region: Region, alive: set[SSAValue] = set()) -> None:
        alive_card = len(alive)
        if not region.blocks:
            return
        assert region.first_block
        block_iterator = PostOrderIterator(region.first_block)
        for block in block_iterator:
            # Process one block
            for op in reversed(block.ops):
                # Process one operation
                alive.difference_update(op.results)
                if isinstance(op, HasRegisterConstraints):
                    _, _, inouts = op.get_register_constraints()
                    for in_reg, _ in inouts:
                        if in_reg in alive:
                            op.emit_error(
                                f"{in_reg.name_hint} should not be read after in/out usage",
                                VerifyException(),
                            )
                alive.update(op.operands)
                # Recursive calls on the embedded regions
                for r in op.regions:
                    self._process_region(region=r, alive=alive.copy())
            # Handle the block arguments
            alive.difference_update(block.args)
        assert alive_card == len(alive)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        self._process_region(op.body)
