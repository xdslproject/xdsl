from dataclasses import dataclass, field
from typing import List, Set
from xdsl.dialects import scf, symref
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.rewriter import Rewriter

# Promoters
# =========
#
# Promoters are extremely important for desymrification of an operation - they
# dictate how an operation updates values. Generally speaking, for any operation
# in the form that uses some symbols @a, @b and @c:
#
#   dialect.op {
#     use @a, @b, @c
#   }
#
# promoter for 'dialect.op' rewrites it into
#
#   fetch @a, @b, @c
#   %new_symbol_values = dialect.op [%argument_symbol_values] {
#     // no symbols used
#   }
#   update @a, @b, @c with %new_symbol_values
#
# such that regions inside operation does not access any symbols, and any symbol
# updates are propogated outside of the operation.


@dataclass
class PromoteException(Exception):
    """
    Exception type if operation promotion is not possible or is unsuccessful.
    """
    op: Operation
    msg: str

    def __str__(self) -> str:
        return f"Unable to promote '{self.op.name}': {self.msg}."


def insert_after(op: Operation, new_op: Operation):
    """Inserts a new operation after another operation."""
    op_parent_block = op.parent_block()
    op_idx = op_parent_block.get_operation_index(op)
    op_parent_block.insert_op(new_op, op_idx + 1)


def insert_before(op: Operation, new_op: Operation):
    """Inserts a new operation before another operation."""
    op_parent_block = op.parent_block()
    op_idx = op_parent_block.get_operation_index(op)
    op_parent_block.insert_op(new_op, op_idx)


@dataclass
class Promoter:
    """Base promoter class."""

    op: Operation
    """Operation for pormotion."""

    rewriter: Rewriter = field(default_factory=Rewriter)
    """IR rewriter used to erase or replace ops."""

    def _find_symbols(self) -> Set[str]:
        """Returns a set of all unpromoted symbols."""
        symbols = set()
        for region in self.op.regions:
            for block in region.blocks:
                for op in block.ops:
                    if isinstance(op, symref.Fetch) or isinstance(op, symref.Update):
                        symbols.add(op.attributes["symbol"].data.data)
        return symbols

    def promote(self):
        # First, check if promotion is needed at all.
        symbols = self._find_symbols()
        if len(symbols) == 0:
            return

        # If we do need it, dispacth the right promoter.
        dialect_name: str = self.op.name.split('.')[0]
        op_name: str = self.op.name.split('.')[1]
        method_name = f"promote_{dialect_name}_{op_name}"

        if not hasattr(self, method_name):
            raise PromoteException(self.op, f"Cannot find promotion method '{method_name}'")

        promote = getattr(self.__class__, method_name)
        promote(self.rewriter, self.op, symbols)

    def promote_scf_if(rewriter: Rewriter, if_op: scf.If, symbols: Set[str]):
        """Promotes a sngle scf.if operation."""

        # First, check if we can do the promotion.
        num_true_blocks = len(if_op.true_region.blocks)
        num_false_blocks = len(if_op.false_region.blocks)
        if num_true_blocks > 1 or num_false_blocks > 1:
            raise PromoteException(if_op, f"found {num_true_blocks} true and {num_false_blocks} blocks, but expected 0 or 1 only")

        true_block = if_op.true_region.blocks[0]
        false_block = if_op.false_region.blocks[0]

        # This scf.if is promotable, so we start by hoisting fetches outside
        # fo this operation. Here, everything is safe to be fetched outside
        # becuse we are guaranteed to have at most one fetch and at most one
        # update and all fetch operations preceed update operations.
        promoted_fetch_ops: List[None | symref.Fetch] = []
        for symbol in symbols:
            fetch_ops_to_promote: List[symref.Fetch] = []

            # First, we have to find all fetch operations in true and false
            # blocks.
            def promote_from_block(block: Block):
                for op in block.ops:
                    if isinstance(op, symref.Fetch):
                        if symbol == op.attributes["symbol"].data.data:
                            fetch_ops_to_promote.append(op)
            promote_from_block(true_block)
            promote_from_block(false_block)

            if len(fetch_ops_to_promote) == 0:
                # This symbol was never fetched, we are done.
                promoted_fetch_ops.append(None)
            else:
                # If this symbol is fetched, hoist it outside and delete
                # fetches in both true and false blocks.
                new_fetch_op = fetch_ops_to_promote[0].clone()
                insert_before(if_op, new_fetch_op)
                for fetch_op in fetch_ops_to_promote:
                    rewriter.replace_op(fetch_op, [], [new_fetch_op.results[0]])
                promoted_fetch_ops.append(new_fetch_op)

        # At this point all possible fetches were hoisted out. We are ready to
        # transform symbol updates into yield operations.
        promoted_symbols: List[str] = []
        true_block_yield_operands: List[SSAValue] = []
        false_block_yield_operands: List[SSAValue] = []
        for i, symbol in enumerate(symbols):

            # Start by finding updates to this symbol in both true and false
            # blocks.
            def promote_from_block(block: Block) -> symref.Update:
                for op in block.ops:
                    if isinstance(op, symref.Update):
                        if symbol == op.attributes["symbol"].data.data:
                            return op
                return None

            true_block_update_op = promote_from_block(true_block)
            false_block_update_op = promote_from_block(false_block)

            # No updates to this symbol - move on.
            if true_block_update_op is None and false_block_update_op is None:
                continue

            # Record the type of the update to reconstruct a new scf.if later.
            update_ty = true_block_update_op.operands[0].typ if true_block_update_op is not None else false_block_update_op.operands[0].typ

            # Otherwise there is an update. First, we check if some of the blocks
            # haven't updated the symbol and it was also not fetched. In this case,
            # we need to introduce a fetch operation in the parent block.
            if (true_block_update_op is None or false_block_update_op is None) and promoted_fetch_ops[i] is None:
                new_fetch_op = symref.Fetch.get(symbol, update_ty)
                insert_before(if_op, new_fetch_op)
                promoted_fetch_ops[i] = new_fetch_op

            def promote_update(update_op: None | symref.Update, yield_operands: List[SSAValue]) -> SSAValue:
                if update_op is not None:
                    yield_value = update_op.operands[0]
                    rewriter.erase_op(update_op)
                else:
                    yield_value = promoted_fetch_ops[i].results[0]
                yield_operands.append(yield_value)

            # Promote updates by making them operands to scf.yield.
            promoted_symbols.append(symbol)
            promote_update(true_block_update_op, true_block_yield_operands)
            promote_update(false_block_update_op, false_block_yield_operands)

        # Next, actually construct a new scf.if operation.
        rewriter.replace_op(true_block.ops[-1], scf.Yield.get(*true_block_yield_operands))
        rewriter.replace_op(false_block.ops[-1], scf.Yield.get(*false_block_yield_operands))
        return_types = list(map(lambda op: op.typ, true_block_yield_operands))

        new_true_region = Region()
        if_op.true_region.clone_into(new_true_region)
        new_false_region = Region()
        if_op.false_region.clone_into(new_false_region)

        new_if_op = scf.If.get(if_op.cond, return_types, new_true_region, new_false_region)
        insert_after(if_op, new_if_op)
        rewriter.erase_op(if_op)

        # Lastly, ensure that symbols are updated with the results from scf.if.
        for i, symbol in enumerate(promoted_symbols):
            update_op = symref.Update.get(symbol, new_if_op.results[i])
            insert_after(new_if_op, update_op)
