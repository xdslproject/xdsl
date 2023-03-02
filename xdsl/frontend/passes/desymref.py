from dataclasses import dataclass, field
from typing import Callable, List, Set, Tuple, TypeAlias
from xdsl.dialects.builtin import IntAttr
from xdsl.frontend import symref
from xdsl.frontend.exception import FrontendProgramException
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
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
# recall that in xDSL an operation can contain regions which in turn contain
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


Definition = symref.Declare
Use: TypeAlias = symref.Fetch | symref.Update

def is_definition(op: Operation) -> bool:
    return isinstance(op, Definition)


def is_use(op: Use) -> bool:
    return isinstance(op, Use)


Read: TypeAlias = symref.Fetch
Write: TypeAlias = symref.Update


def is_read(op: Operation) -> bool:
    return isinstance(op, Read)


def is_write(op: Operation) -> bool:
    return isinstance(op, Write)


def get_symbol(op: Definition | Use) -> str:
    if is_definition(op):
        return op.sym_name.data
    else:
        return op.symbol.root_reference.data


def get_symbols(block: Block) -> Set[str]:
    symbols: Set[str] = set()
    for op in block.ops:
        if is_definition(op) or is_use(op):
            symbols.add(get_symbol(op))
    return symbols


def count_ops_by(block: Block, cond: Callable[[Operation], bool]) -> int:
    count = 0
    for op in block.ops:
        if cond(op):
            count += 1
    return count


def select_ops_by(block: Block, cond: Callable[[Operation], bool]) -> List[Operation]:
    selected: List[Operation] = []
    for op in block.ops:
        if cond(op):
            selected.append(op)
    return selected

def lower_bound(ops: List[Operation], op: Operation, numbering: Callable[[Operation], int]) -> Operation | None:
    idx = numbering(op)
    low_idx = -1
    high_idx = len(ops) - 1

    while low_idx < high_idx:
        mid_idx = (high_idx - low_idx + 1) // 2 + low_idx
        user_idx = numbering(ops[mid_idx])

        if user_idx < idx:
            low_idx = mid_idx
        else:
            high_idx = mid_idx - 1

    if low_idx == -1:
        return None
    return ops[low_idx]


@dataclass
class Symbol:
    """
    Encapsulates all information about this symbol, including its uses, etc.
    """

    definition: Definition | None = field(default=None)
    """Definition of the symbol."""

    write: Write | None = field(default=None)
    """Write to the symbol."""

    write_blocks: List[Block] = field(default_factory=list)
    """List of blocks that write to the symbol."""

    single_block: Block | None = field(default=None)
    """Set if the symbol is only used in a single block."""

    used_in_single_block: bool = field(default=True)
    """True if the symbol is only used in a single block."""

    never_read: bool = field(default=True)
    """True if the symbol is never read."""

    uses: List[Use] = field(default_factory=list)
    """All uses of this symbol, i.e. all reads and writes."""

    def add_use(self, op: Use):
        # This use reads the symbol. 
        if is_read(op) and self.never_read:
            self.never_read = False

        # Record a write to this symbol.
        block = op.parent_block()
        if is_write(op):
            self.write_blocks.append(block)
            self.write = op

        # Update the flags depending whether the uses of the symbol are within
        # the same block.
        if self.used_in_single_block:
            if self.single_block is None:
                self.single_block = block
            elif self.single_block != block:
                self.used_in_single_block = False
        self.uses.append(op)


@dataclass
class Desymrefier:
    """
    Rewrites the program by removing all reads/writes from/to symbols and symbol
    definitions.
    """

    rewriter: Rewriter
    """Rewriter to replace and erase operations."""

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
            # handle it seprately.
            self.prepare_block(region.blocks[0])
        else:
            # TODO: Support regions with multiple blocks.
            raise FrontendProgramException(
                f"Running desymrefier on region with {num_blocks} > 1 blocks is "
                "not supported.")

    def prepare_block(self, block: Block):
        """Prepares a block for promotion."""

        # First, desymrefy nested regions.
        for op in block.ops:
            self.desymrefy(op)

        self.prune_definitions(block)
        self.prune_uses_without_definitions(block)

        symbols = get_symbols(block)
        for symbol in symbols:
            num_reads = count_ops_by(block, lambda op: is_read(op) and get_symbol(op) == symbol)
            num_writes = count_ops_by(block, lambda op: is_write(op) and get_symbol(op) == symbol)
            if  num_reads > 1 or num_writes > 1:
                raise FrontendProgramException(
                    f"Block {block} not ready for promotion: found {num_reads}"
                    f" reads and {num_writes} writes.")

    def prune_definitions(self, block: Block):
        """Removes all symbol definitions and their uses from the block."""
        while True:
            # Find all symbol definitions in this block. If no definitions
            # found, terminate.
            definitions: List[Definition] = select_ops_by(block, is_definition)
            if len(definitions) == 0:
                return

            # Otherwise, some definitions are still alive.
            for definition in definitions:
                symbol = get_symbol(definition)

                # Find all reads and writes for this symbol.
                reads: List[Read] = select_ops_by(block, lambda op: is_read(op) and get_symbol(op) == symbol)
                writes: List[Write] = select_ops_by(block, lambda op: is_write(op) and get_symbol(op) == symbol)
 
                # Symbol is never read, so remove its definition and any writes.
                if len(reads) == 0:
                    for write in writes:
                        self.rewriter.erase_op(write)
                    self.rewriter.erase_op(definition)
                    continue

                # For symbols which are written once, the write dominates all
                # the uses and therefore can be trivially replaced.
                if len(writes) == 1:
                    write = writes[0]
                    for read in reads:
                        self.rewriter.replace_op(read, [], [write.operands[0]])
                    self.rewriter.erase_op(write)
                    self.rewriter.erase_op(definition)
                    continue

                # If there are multiple reads and writes, replace every
                # read with the closest preceding write.
                for read in reads:
                    write = lower_bound(writes, read, block.get_operation_index)
                    if write is not None:
                        self.rewriter.replace_op(read, [], [write.operands[0]])

    def prune_uses_without_definitions(self, block: Block):
        """Removes all possible symbol uses in a single block."""
        prepared_symbols: Set[str] = set()

        while True:
            is_unused_read: Callable[[Operation], bool] = lambda op: is_read(op) and len(op.results[0].uses) == 0
            unused_reads = select_ops_by(block, is_unused_read)
            for read in unused_reads:
                self.rewriter.erase_op(read)

            # Find all symbols that are still in use in this block.
            symbol_worklist: Set[str] = set(map(get_symbol, select_ops_by(block, lambda op: is_use(op) and get_symbol(op) == symbol and symbol not in prepared_symbols)))
            if len(symbol_worklist) == 0:
                return

            for symbol in symbol_worklist:
                reads: List[Read] = select_ops_by(block, lambda op: is_read(op) and get_symbol(op) == symbol)
                writes: List[Write] = select_ops_by(block, lambda op: is_write(op) and get_symbol(op) == symbol)

                # There are no reads, so we can only keep the last write to the
                # symbol.
                if len(reads) == 0:
                    for write in writes[:-1]:
                        self.rewriter.erase_op(write)
                    prepared_symbols.add(symbol)
                    continue

                # There are no writes, so we can replace all reads with this
                # symbol.
                if len(writes) == 0:
                    for read in reads[1:]:
                        self.rewriter.replace_op(read, [], [reads[0].results[0]])
                    prepared_symbols.add(symbol)
                    continue

                # Sets of reads and writes are disjoint.
                last_read_idx = block.get_operation_index(reads[-1])
                first_write_idx = block.get_operation_index(writes[0])
                if last_read_idx < first_write_idx:
                    for read in reads[1:]:
                        self.rewriter.replace_op(read, [], [reads[0].results[0]])
                    for write in writes[:-1]:
                        self.rewriter.erase_op(write)
                    prepared_symbols.add(symbol)
                    continue

                # Otherwise, replace reads with the closest preceding write.
                for read in reads:
                    write = lower_bound(writes, read, block.get_operation_index)
                    if write is not None:
                        self.rewriter.replace_op(read, [], [write.operands[0]])


@dataclass
class DesymrefyPass:
    """Pass which is called by the client to desymrefy xDSL code."""

    @staticmethod
    def run(op: Operation) -> bool:
        rewriter = Rewriter()
        Desymrefier(rewriter).desymrefy(op)
