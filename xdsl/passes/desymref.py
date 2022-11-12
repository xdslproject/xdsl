from dataclasses import dataclass, field
from typing import Dict, List
from xdsl.dialects import scf, symref
from xdsl.frontend.codegen.exception import prettify
from xdsl.ir import Block, Operation, Region

from xdsl.rewriter import Rewriter

# Background
# ==========
#
# We want to allow users to write non-SSA code. For example, the frontend should
# allow functions like this:
#
#   def foo() -> i32:
#     a: i32 = 0
#     for i in range(100):
#       a = a + 2
#     return a
#
# Our solution is to have a symref dialect: it uses declare, fetch and store
# operations (for those familiar with LLVM, these correspond to alloca, load and
# store but based on symbols instead of memory locations). Each symref operation
# is defined by the programmer with a simple mapping: 
# 
# 1. variable declaration, e.g. a: i32 = 0 maps to declare @a
# 2. variable use, e.g. ... = a maps to ... = fetch @a
# 3. variable assignement, e.g. a = ... maps to update @a ...
#
# With these, it is relatively straightforward to lower any Python program to xDSL.
# For example, the code above would become (in a slightly abused xDSL syntax):
#
#   func foo() -> i32 {
#     declare @a
#     update @a with 0
#     for 0 to 100 {
#       t1 = fetch @a
#       t2 = add t1 2
#       update @a t2
#     }
#     t3 = fetch @a
#     return t3
#   }
#
# Note that while insertion of symref operations is easy, the generated xDSL is not
# fully in SSA form. For example, blocks in region should pass values between each
# other via block arguments, instead of symref calls. Similarly, some operations like
# scf.if or affine.for should yield a value.
# 
# 
# Desymrefication pass
# ====================
# 
# In this file, we implement desymrefication pass - it goes over generated xDSL and
# removes all symref operations. To describe how the pass works, first recall that in
# xDSL an operation can contain regions which in turn contain CFGs of basic blocks.
# An operation pass control flow to a region and on exit from the region the control
# flow returns to the operation. Then the operation can transfer control flo to another
# region, etc.
#
# Desymrefication can be applied to regions and operations. First, consider a simple
# case when all operations in the region do not have any nested regions. This means
# that the region is simply a CFG. Otherwise, there are some operations that contain
# regions. However, we can apply desymrefy them first, reducing the problem to the
# the first case. In pseudocode, that would look like:
#
# desymref_region(region):
#   for block in region.blocks:
#     for op in block.ops:
#       desymref_op(op)
#   // at this point no op contains a symref operation
#   // in the nested region.
#
# Next, we describe how desymrefication actually works, on a high level. Every region
# either declares a symbol or uses it from the parent region. Let's consider each case
# separately.
# 
#   Case 1: Symbol is declared in this region.
#   This is an easy case. We already know that symbol is only used in this region and no
#   nested region uses it. Therefore a running any SSA-construction algorithm on the CFG
#   is enough.
#
#   Case 2: Symbol is not declared in this region.
#   # TODO: support this!
#


@dataclass
class DesymrefyException(Exception):
    """
    Exception type if something goes terribly wrong when running `desymref` pass.
    """
    msg: str

    def __str__(self) -> str:
        return f"Exception in desymrefy: {self.msg}."


@dataclass
class SymbolInfo:
    declare: symref.Declare | None = field(default=None)
    update: symref.Update | None = field(default=None)
    def_blocks: List[Block] = field(default_factory=list)
    single_block: Block | None = field(default=None)
    used_in_single_block: bool = field(default=True)

    never_fetched: bool = field(default=True)

    users: List[symref.Fetch | symref.Update] = field(default_factory=list)

    def add_user(self, op: symref.Fetch | symref.Update):
        if self.never_fetched and isinstance(op, symref.Fetch):
            self.never_fetched = False
        if isinstance(op, symref.Update):
            self.def_blocks.append(op.parent_block())
            self.update = op

        if self.used_in_single_block:
            if not self.single_block:
                self.single_block = op.parent_block()
            elif self.single_block != op.parent_block():
                self.used_in_single_block = False

        self.users.append(op)

    @staticmethod
    def from_declare(op: symref.Declare) -> 'SymbolInfo':
        return SymbolInfo(op)

    @staticmethod
    def from_fetch_or_update(op: symref.Fetch | symref.Update) -> 'SymbolInfo':
        info = SymbolInfo(None)
        info.add_user(op)
        return info


@dataclass
class Desymrefier:
    """Class responsible for rewriting xDSL and removing symref operations."""

    rewriter: Rewriter
    """Rewriter to replace and erase operations."""

    def run_on_operation(self, op: Operation):
        """Desymrefies an operation."""

        # For operation with no regions we don't have to do any work.
        if len(op.regions) == 0:
            return

        # Otherwise, there is a region containing a CFG and we have to desymrefy it.
        for region in op.regions:
            self.run_on_region(region)

        # Some regions were not fully desymrefied, so use the definition of the
        # operation to decide what to do.
        # TODO: do something here.

    def run_on_region(self, region: Region):
        """Desymrefies a region."""
        num_blocks = len(region.blocks)
        if num_blocks == 1:
            # If there is only one block, desymrefication is significantly easier.
            self.run_on_single_block(region.blocks[0])
        else:
            # TODO: support regions with multiple blocks. This is not trivial, particularly
            # when the symbol is declared in one of the parent regions.
            raise DesymrefyException(f"running desymrefier on region with {num_blocks} blocks is not supported")

    def remove_declared_symbols(self, block: Block):
        """Removes all symbol declarations in a single block."""
        while True:
            # Get all symbol declarations in this block.
            declare_ops: List[symref.Declare] = []
            for op in block.ops:
                if isinstance(op, symref.Declare):
                    declare_ops.append(op)
            
            # No declarations - we are done.
            if len(declare_ops) == 0:
                return
            
            # Otherwise, some declarations are still alive and
            # there is still some work to do.
            for declare_op in declare_ops:
                symbol = declare_op.attributes["sym_name"].data

                # Find all fetches and updates of this symbol. Keep
                # track of their indices to 
                fetch_ops: List[symref.Fetch] = []
                update_ops: List[symref.Update] = []
                for op in block.ops:
                    if isinstance(op, symref.Fetch):
                        op_symbol = op.attributes["symbol"].data.data
                        if symbol == op_symbol:
                            fetch_ops.append(op)
                    elif isinstance(op, symref.Update):
                        op_symbol = op.attributes["symbol"].data.data
                        if symbol == op_symbol:
                            update_ops.append(op)
                
                # Declared symbol can be never read, and so all updates
                # are dead.
                if len(fetch_ops) == 0:
                    for update_op in update_ops:
                        self.rewriter.erase_op(update_op)
                    self.rewriter.erase_op(declare_op)
                    continue

                # Otherwise, symbol is read and used. We can first check
                # if it was updated once, sunce then we can replace every
                # symbol fetch trwith updated value trivially (recall that
                # we are in the same block, so no CFG and the dominance
                # relations do not matter)!
                if len(update_ops) == 1: 
                    for fetch_op in fetch_ops:
                        self.rewriter.replace_op(fetch_op, [], [update_ops[0].operands[0]])
                    self.rewriter.erase_op(declare_op)
                    self.rewriter.erase_op(update_ops[0])
                    continue

                # If there are multiple fetches and updates, we repalce every
                # fetch with the closest preceding update. This is easy and can
                # be done with a binary search.
                for fetch_op in fetch_ops:
                    fetch_idx = block.get_operation_index(fetch_op)

                    # TODO: use binary search.
                    prev_update_op = None
                    for update_op in update_ops:
                        update_idx = block.get_operation_index(update_op)
                        if fetch_idx < update_idx:
                            break
                        prev_update_op = update_op

                    # Replace the result of the fetch with update's operand.
                    if prev_update_op is not None:
                        self.rewriter.replace_op(fetch_op, [], [prev_update_op.operands[0]])

    def remove_used_symbols(self, block: Block):
        """Removes all possible symbol uses in a single block."""

        # List of symbols we successfully simplified.
        ignore_symols = set()

        while True:
            # Immediately remove all unused fetches.
            for op in block.ops:
                if isinstance(op, symref.Fetch):
                    if len(op.results[0].uses) == 0:
                        self.rewriter.erase_op(op)

            # Find all symbols that are still in use in this block.
            symbols = set()
            for op in block.ops:
                if isinstance(op, symref.Fetch) or isinstance(op, symref.Update):
                    symbol = op.attributes["symbol"].data.data
                    if symbol not in ignore_symols:
                        symbols.add(symbol)

            if len(symbols) == 0:
                return

            for symbol in symbols:
                # First, get a list of fetces and updates.
                fetch_ops: List[symref.Fetch] = []
                update_ops: List[symref.Update] = []
                for op in block.ops:
                    if isinstance(op, symref.Fetch):
                        op_symbol = op.attributes["symbol"].data.data
                        if symbol == op_symbol:
                            fetch_ops.append(op)
                    elif isinstance(op, symref.Update):
                        op_symbol = op.attributes["symbol"].data.data
                        if symbol == op_symbol:
                            update_ops.append(op)
 
                # There is no fetches of this symbol. Then we can only
                # keep the last update to that symbol!
                if len(fetch_ops) == 0:
                    for update_op in update_ops[:-1]:
                        self.rewriter.erase_op(update_op)
                    ignore_symols.add(symbol)
                    continue

                # There are no updates to this symbol. We can replace
                # all fetches with this symbol
                if len(update_ops) == 0:
                    for fetch_op in fetch_ops[1:]:
                        self.rewriter.replace_op(fetch_op, [], [fetch_ops[0].results[0]])
                    ignore_symols.add(symbol)
                    continue


                # Otherwise, in general we are done if all fetches preceed all updates.
                last_fetch_idx = block.get_operation_index(fetch_ops[-1])
                first_update_idx = block.get_operation_index(update_ops[0])
                if last_fetch_idx < first_update_idx:
                    # Get rid of all but one fetches, and keep only the
                    # last update.
                    for fetch_op in fetch_ops[1:]:
                        self.rewriter.replace_op(fetch_op, [], [fetch_ops[0].results[0]])
                    for update_op in update_ops[:-1]:
                        self.rewriter.erase_op(update_op)
                    ignore_symols.add(symbol)
                    continue

                # This symbol should still be processed then, and has a micture of
                # fetches and updates. We can use the same strategy as with declared
                # symbols and replace all fetches with updated value.
                for fetch_op in fetch_ops:
                    fetch_idx = block.get_operation_index(fetch_op)

                    # TODO: use binary search.
                    prev_update_op = None
                    for update_op in update_ops:
                        update_idx = block.get_operation_index(update_op)
                        if fetch_idx < update_idx:
                            break
                        prev_update_op = update_op

                    # Replace the result of the fetch with update's operand.
                    if prev_update_op is not None:
                        self.rewriter.replace_op(fetch_op, [], [prev_update_op.operands[0]])

    def check_single_block_for_promotion(self, block: Block):
        symbols = set()
        for op in block.ops:
            if isinstance(op, symref.Fetch) or isinstance(op, symref.Update):
                symbols.add(op.attributes["symbol"].data.data)
        
        # Every symbol should be fetched and updated at most once!
        for symbol in symbols:
            fetch_cnt = 0
            update_cnt = 0
            for op in block.ops:
                if isinstance(op, symref.Fetch):
                    fetch_cnt += 1
                elif isinstance(op, symref.Update):
                    update_cnt += 1
            
            if fetch_cnt > 1 or update_cnt > 1:
                raise DesymrefyException(f"Block {prettify(block)} not ready for promotion: found {fetch_cnt} fetches and {update_cnt} updates") 


    def run_on_single_block(self, block: Block):
        """Desymrefies a single block inside a region."""

        # First, we desymrefy nested regions.
        for op in block.ops:
            self.run_on_operation(op)
        
        # Case 1: symbol is declared in this region. Then,
        # iterate until all symbols are destroyed.
        self.remove_declared_symbols(block)

        # Case 2: some symbols are not declared in this region.
        # For a single block, it is not that different!
        self.remove_used_symbols(block)

        # Sanity check.
        self.check_single_block_for_promotion(block)

@dataclass
class DesymrefyPass:
    """Pass which is called by the client to desymrefy xDSL code."""

    @staticmethod
    def run(op: Operation) -> bool:
        rewriter = Rewriter()
        Desymrefier(rewriter).run_on_operation(op)
