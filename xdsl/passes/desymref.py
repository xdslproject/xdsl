from dataclasses import dataclass, field
from typing import Dict, List
from xdsl.dialects import symref
from xdsl.ir import Block, Operation, Region

from xdsl.rewriter import Rewriter

# Desymrefication pass
# ====================
#
# We want to allow users to write non-SSA code. For example, the frontend should
# allow functions like this:
#
#   def foo() -> i32:
#     a: i32 = 0
#     for i in range(100):
#       a = use(a)
#     return a
#
# Our solution is to have a symref dialect: it uses declare, fetch and store
# operations which mimic LLVM' alloca, load and store but based on symbols
# defined by the programmer. This way, the example from above would become:
#
#   func.foo(){
#     symref.declare @a
#     symref.update @a with 0
#     affine.for 0 to 100 {
#       t1 = symref.fetch from @a
#       t2 = use(t1)
#       symref.update @a with t2
#     }
#     t3 = symref.fetch from @a
#     func.return t3
#   }
#
# Now, with desymrefication pass we can remove calls to symref and make everything
# look like normal SSA again.
#
# To describe how the pass works, first recall that in xDSL an operation can contain
# regions which in turn contain CFGs of basic blocks. Operation pass control flow to
# a region and on exit from the region the control flow returns to the operation. Then
# the operation can transfer control flo to another region, etc.
#
# An important observation here, is that for each region one the following holds:
#   1. The region defines the value using symref.declare. This implies that the value
#      never leaves the region.
#   2. The region uses the value defined in the parent region containing the parent
#      operation. Then the value is always first obtained using symref.fetch and
#      maybe updated in the end using symref.update.
# Now consider these cases in part.
#
# Case 1:
# {
#   t = symref.fetch @a
# }
#
# TODO: finish documentation.


@dataclass
class DesymrefException(Exception):
    """
    Exception type if something goes terribly wrong when running `desymref` pass.
    """
    msg: str

    def __str__(self) -> str:
        return f"Exception in desymref pass: {self.msg}."


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
    rewriter: Rewriter
    """Rewriter to replace and erase operations."""

    def run_on_operation(self, op: Operation):
        """Returns true if operation was successfully desymrefied."""

        # For operation with no regions we don't have to do any work.
        if len(op.regions) == 0:
            return

        # Otherwise, there is a region containing a CFG and we have to desymrefy it.
        for region in op.regions:
            self.run_on_region(region)

        # TODO: some regions were not desymrefied!

    def run_on_region(self, region: Region):
        # First, remove symref from regions within other operations. At the same
        # time, gather all symbols local to this region.
        info_map: Dict[str, SymbolInfo] = dict()
        for block in region.blocks:
            for op in block.ops:
                self.run_on_operation(op)

                # If this symbol is declared in this region, add it to the
                # information map.
                if isinstance(op, symref.Declare):
                    symbol = op.attributes["sym_name"].data
                    info_map[symbol] = SymbolInfo.from_declare(op)

                # Otherwise, the symbol can be fetched or updated.
                if isinstance(op, symref.Fetch) or isinstance(op, symref.Update):
                    symbol = op.attributes["symbol"].data.data
                    if symbol not in info_map:
                        info_map[symbol] = SymbolInfo.from_fetch_or_update(op)
                    else:
                        info_map[symbol].add_user(op)

        # At this point only this region has desymref operations so we
        # do not have to worry about uses in regions defined by ops.

        # Now, process each symbol in part.
        for symbol, info in info_map.items():

            # First, is symbol is declared in this region it can simply be
            # never used.
            if info.declare is not None and len(info.users) == 0:
                self.rewriter.erase_op(info.declare)
                continue

            # Second, symbol can be never fetched. Here we consider only the case
            # when symbol declared in this region. Otherwise, updates can be performed
            # in different basic blocks so it is not clear which value would stored in
            # the end.
            if info.never_fetched and info.declare is not None:
                for user in info.users:
                    self.rewriter.erase_op(user)
                self.rewriter.erase_op(info.declare)
                continue

            # Next, symbol can be updated only once. This means that if this symbol is
            # declared in this regions, its update dominates all users because in Python
            # we always initialize variables, so update immediately follows a declare.
            if info.declare is not None and len(info.def_blocks) == 1:
                for user in info.users:
                    if isinstance(user, symref.Fetch):
                        self.rewriter.replace_op(user, [], [info.update.operands[0]])
                self.rewriter.erase_op(info.update)
                self.rewriter.erase_op(info.declare)
                continue

            # Lastly, symbol can be used within a single basic block.
            if info.used_in_single_block:
                # First, split fetch and update operations, recording operation
                # indices.
                fetch_ops: List[(int, symref.Fetch)] = []
                update_ops: List[(int, symref.Update)] = []
                for index, use in enumerate(info.users):
                    if isinstance(use, symref.Fetch):
                        fetch_ops.append((index, use))
                    if isinstance(use, symref.Update):
                        update_ops.append((index, use))

                # For each fetch operation, find the closest preceding update.
                # TODO: use binary search.
                for i, fetch_op in fetch_ops:
                    preceding_update_op = None
                    for j, update_op in update_ops:
                        if i < j:
                            break
                        preceding_update_op = update_op

                # Replace the result of the fetch with update's operand.
                if preceding_update_op is not None:
                    self.rewriter.replace_op(fetch_op, [], [update_op.operands[0]])

                # Optionally remove declaration if it exists.
                if info.declare is not None:
                    self.rewriter.erase_op(info.declare)

                # All updates are dead. Remove them.
                if len(update_ops) > 0:
                    if info.declare is None:
                        # Note that actually if symbol is dclared outside of this
                        # region, last update must be kept.
                        update_ops = update_ops[:-1]
                    for _, update_op in update_ops:
                        self.rewriter.erase_op(update_op)
                continue

            # Otherwise, we have to run an actual SSA-construction algorithm.
            # TODO: copy algorithm from old implementation here and make it work
            # in this new setting.


@dataclass
class DesymrefyPass:

    @staticmethod
    def run(op: Operation) -> bool:
        Desymrefier(Rewriter()).run_on_operation(op)
