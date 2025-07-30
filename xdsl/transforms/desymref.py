from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin, symref
from xdsl.frontend.pyast.utils.exceptions import FrontendProgramException
from xdsl.ir import Block, Operation, Region
from xdsl.passes import ModulePass
from xdsl.rewriter import Rewriter

# Background
# ==========
#
# We want to allow users to write non-SSA code. For example, the frontend should
# allow functions like this:
#
# ```
# def foo() -> i32:
#   a: i32 = 0
#   for i in range(100):
#     a = a + 2
#   return a
# ```
#
# The solution is to have a `symref` dialect: it uses declare, fetch and store
# operations (for those familiar with LLVM, these correspond to `alloca`, `load`
# and `store` but based on symbols instead of memory locations). Each symref
# operation is defined by the programmer with a simple mapping:
#
# 1. variable declaration, e.g. `a: i32 = 0` maps to `declare @a`
# 2. variable use, e.g. `... = a` maps to `... = fetch @a`
# 3. variable assignement, e.g. `a = ...` maps to `update @a ...`
#
# With these, it is relatively straightforward to lower any Python program to
# xDSL (taking care of scoping rules, of course). For example, the code above
# would become (in a slightly abused xDSL syntax):
#
# ```
# func foo() -> i32 {
#   declare @a
#   update @a with 0
#   for 0 to 100 {
#     t1 = fetch @a
#     t2 = add t1 2
#     update @a t2
#   }
#   t3 = fetch @a
#   return t3
# }
# ```
#
# Note that while insertion of symref operations is easy, the generated xDSL is
# not fully in SSA form. For example, blocks in region should pass values
# between each other via block arguments, instead of symref calls. Similarly,
# some operations like scf.if or affine.for should yield a value.
#
#
# Desymrefication pass
# ====================
#
# In this file, we implement desymrefication pass - it goes over generated xDSL
# and removes all symref operations. To describe how the pass works, first
# recall that in xDSL an operation can contain regions which in turn can contain
# CFGs of basic blocks. An operation passes control flow to a region and on exit
# from the region the control flow returns to the operation. Then the operation
# can transfer control flow to another region, etc.
#
# Desymrefication can be applied to regions and operations. First, consider a
# simple case when all operations in the region do not have any nested regions.
# This means that the region is simply a CFG. Otherwise, there are some
# operations that contain regions. However, we can apply desymrefication to
# these operations them first, reducing the problem to the the first case. In
# pseudocode, that would look like:
#
# ```
# desymref_region(region):
#   for block in region.blocks:
#     for op in block.ops:
#       desymref_op(op)
#     # at this point no op contains a symref operation
#     # in the nested region.
# ```
#
# Next, we describe how desymrefication actually works, on a high level.
# Every region either declares a symbol or uses it from the parent region.
# Let's consider each case separately.
#
# Case 1: Symbol is declared in this region. This is an easy case. We already
# know that symbol is only used in this region and no other nested region uses
# it. Therefore running any SSA-construction algorithm on the CFG is enough.
#
# Case 2: Symbol is not declared in this region. In this case, we only support
# non-CFGs at the moment (i.e. single block regions). We observe that in general
# every symbol can be simplified within a block to have at most one fetch (start
# of the block) at most one update (end of the block). This means that using
# specification of an operation we can promote these symbols outside.
# TODO: Add op promotion.


def has_symbol(op: Operation) -> bool:
    return isinstance(op, symref.DeclareOp | symref.FetchOp | symref.UpdateOp)


def get_symbol(op: Operation) -> str | None:
    """
    Returns a symbol attribute for an operation. If operation has no symbol
    attributes, None is returned.
    """
    if isinstance(op, symref.DeclareOp):
        return op.sym_name.data
    elif isinstance(op, symref.FetchOp | symref.UpdateOp):
        return op.symbol.root_reference.data
    else:
        return None


def get_symbols(block: Block) -> set[str]:
    """Returns a set of all symbols defined/used in a basic block."""
    symbols: set[str] = set()
    for op in block.ops:
        if has_symbol(op):
            symbol = get_symbol(op)
            assert symbol is not None
            symbols.add(symbol)
    return symbols


def lower_positional_bound(
    writes: list[symref.UpdateOp], read: symref.FetchOp
) -> Operation | None:
    """
    Returns a nearest write preceeding the `read`. If there is no such write,
    `None` is returned.

    Pre-condition: list `writes` is sorted based on operation indices.
    """
    block = read.parent_block()
    assert block is not None

    idx = block.get_operation_index(read)
    low_idx = -1
    high_idx = len(writes) - 1

    # Binary search to find the right write.
    while low_idx < high_idx:
        mid_idx = (high_idx - low_idx + 1) // 2 + low_idx
        user_idx = block.get_operation_index(writes[mid_idx])

        if user_idx < idx:
            low_idx = mid_idx
        else:
            high_idx = mid_idx - 1

    if low_idx == -1:
        return None
    return writes[low_idx]


@dataclass
class Desymrefier:
    """
    Rewrites the program by removing all reads/writes from/to symbols and symbol
    definitions.
    """

    def desymrefy(self, op: Operation):
        """
        Desymrefy an operation. This method guarantees that the operation does
        not have any symbols.
        """
        self.prepare_op(op)
        self.promote_op(op)

    def promote_op(self, op: Operation):
        """
        Promotes an operation. This method guarantees that the operation does
        not have any symbols.
        """
        # TODO: Add promoters in the next patch.
        pass

    def prepare_op(self, op: Operation):
        """
        Prepares an operation for promotion. This method guarantees that any
        symbol in any region of this operation is read at most once and written
        at most once.
        """

        # For operation with no regions we don't have to do any work.
        if len(op.regions) == 0:
            return

        # Otherwise, we have to prepare regions.
        for region in op.regions:
            self.prepare_region(region)

    def prepare_region(self, region: Region):
        """
        Prepares a region for promotion. This method guarantees that any symbol
        in the region is read at most once and written at most once.
        """
        num_blocks = len(region.blocks)
        if num_blocks == 1:
            # If there is only one block, preparing region is easier, so we
            # handle it separately.
            self.prepare_block(region.block)
        else:
            # TODO: Support regions with multiple blocks. This is not too
            # difficult but requires many more things:
            # 1. SSA-construction algorithm to prune symbol declarations. This
            #    also requires analyses passes (dominators).
            # 2. Insertion of entry/exit blocks to ensure dominance.
            raise FrontendProgramException(
                f"Running desymrefier on region with {num_blocks} > 1 blocks is "
                "not supported."
            )

    def prepare_block(self, block: Block):
        """Prepares a block for promotion."""

        # First, desymrefy nested regions.
        for op in block.ops:
            self.desymrefy(op)

        self.prune_definitions(block)
        self.prune_uses_without_definitions(block)

        # Ensure that each symbol is read/written at most once.
        symbols = get_symbols(block)
        for symbol in symbols:
            num_reads = sum(
                isinstance(op, symref.FetchOp)
                for op in block.ops
                if get_symbol(op) == symbol
            )
            num_writes = sum(
                isinstance(op, symref.UpdateOp)
                for op in block.ops
                if get_symbol(op) == symbol
            )
            if num_reads > 1 or num_writes > 1:
                raise FrontendProgramException(
                    f"Block {block} not ready for promotion: found {num_reads}"
                    f" reads and {num_writes} writes."
                )

    def prune_definitions(self, block: Block):
        """Removes all symbol definitions and their uses from the block."""

        # Find all symbol definitions in this block. If no definitions
        # found, terminate.
        while (
            len(
                definitions := [
                    op for op in block.ops if isinstance(op, symref.DeclareOp)
                ]
            )
            > 0
        ):
            # Otherwise, some definitions are still alive.
            for definition in definitions:
                symbol = get_symbol(definition)

                # Find all reads and writes for this symbol.
                reads = [
                    op
                    for op in block.ops
                    if isinstance(op, symref.FetchOp) and get_symbol(op) == symbol
                ]
                writes = [
                    op
                    for op in block.ops
                    if isinstance(op, symref.UpdateOp) and get_symbol(op) == symbol
                ]

                # Symbol is never read, so remove its definition and any writes.
                if len(reads) == 0:
                    for write in writes:
                        Rewriter.erase_op(write)
                    Rewriter.erase_op(definition)
                    continue

                # For symbols which are written once, the write dominates all
                # the uses and therefore can be trivially replaced.
                if len(writes) == 1:
                    write = writes[0]
                    for read in reads:
                        Rewriter.replace_op(read, [], [write.operands[0]])
                    Rewriter.erase_op(write)
                    Rewriter.erase_op(definition)
                    continue

                # If there are multiple reads and writes, replace every
                # read with the closest preceding write.
                for read in reads:
                    write = lower_positional_bound(writes, read)
                    if write is not None:
                        Rewriter.replace_op(read, [], [write.operands[0]])

    def _prune_unused_reads(self, block: Block):
        def is_unused_read(op: Operation) -> bool:
            return isinstance(op, symref.FetchOp) and not op.results[0].uses

        unused_reads = [op for op in block.ops if is_unused_read(op)]
        for read in unused_reads:
            Rewriter.erase_op(read)

    def prune_uses_without_definitions(self, block: Block):
        """Removes all possible symbol uses in a single block."""
        prepared_symbols: set[str] = set()

        while True:
            self._prune_unused_reads(block)

            # Find all symbols that are still in use in this block.
            symbol_worklist: set[str] = {
                symbol
                for symbol in get_symbols(block)
                if symbol not in prepared_symbols
            }
            if len(symbol_worklist) == 0:
                return

            for symbol in symbol_worklist:
                reads = [
                    op
                    for op in block.ops
                    if isinstance(op, symref.FetchOp) and get_symbol(op) == symbol
                ]
                writes = [
                    op
                    for op in block.ops
                    if isinstance(op, symref.UpdateOp) and get_symbol(op) == symbol
                ]
                assert len(reads) > 0 or len(writes) > 0

                # There are no reads, so we can only keep the last write to the
                # symbol.
                if len(reads) == 0:
                    for write in writes[:-1]:
                        Rewriter.erase_op(write)
                    prepared_symbols.add(symbol)
                    continue

                # There are no writes, so we can replace all reads with this
                # symbol.
                if len(writes) == 0:
                    for read in reads[1:]:
                        Rewriter.replace_op(read, [], [reads[0].results[0]])
                    prepared_symbols.add(symbol)
                    continue

                # sets of reads and writes are disjoint.
                last_read_idx = block.get_operation_index(reads[-1])
                first_write_idx = block.get_operation_index(writes[0])
                if last_read_idx < first_write_idx:
                    for read in reads[1:]:
                        Rewriter.replace_op(read, [], [reads[0].results[0]])
                    for write in writes[:-1]:
                        Rewriter.erase_op(write)
                    prepared_symbols.add(symbol)
                    continue

                # Otherwise, replace reads with the closest preceding write.
                for read in reads:
                    write = lower_positional_bound(writes, read)
                    if write is not None:
                        Rewriter.replace_op(read, [], [write.operands[0]])


class FrontendDesymrefyPass(ModulePass):
    name = "frontend-desymrefy"

    def apply(self, ctx: Context, op: builtin.ModuleOp):
        Desymrefier().desymrefy(op)
