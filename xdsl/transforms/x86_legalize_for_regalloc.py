from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, x86
from xdsl.ir import Operation, Region, SSAValue
from xdsl.ir.post_order import PostOrderIterator
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter


@dataclass(frozen=True)
class X86LegalizeForRegallocPass(ModulePass):
    """
    Legalize x86 code before register allocation:
    - remove copies when they are the last use of a register variable.
    """

    name = "x86-legalize-for-regalloc"

    def _process_region(
        self,
        region: Region,
        to_erase: list[Operation] = [],
        alive: set[SSAValue] = set(),
    ) -> None:
        if not region.blocks:
            return
        assert region.first_block
        block_iterator = PostOrderIterator(region.first_block)
        for block in block_iterator:
            # Process one block
            for op in reversed(block.ops):
                # Process one operation
                alive.difference_update(op.results)
                if isinstance(op, x86.DS_MovOp):
                    if op.source not in alive:
                        op.destination.replace_all_uses_with(op.source)
                        to_erase.append(op)
                alive.update(op.operands)
                # Recursive calls on the embedded regions
                for r in op.regions:
                    self._process_region(
                        region=r, alive=alive.copy(), to_erase=to_erase
                    )
            # Handle the block arguments
            alive.difference_update(block.args)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        to_erase: list[Operation] = []
        self._process_region(op.body, to_erase)
        for e in to_erase:
            Rewriter.erase_op(e)
