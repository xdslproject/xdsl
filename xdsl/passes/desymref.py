from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Optional, Set, Union

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import Func, FuncOp, Return
from xdsl.dialects.symref import Declare, Fetch, Update
from xdsl.ir import Block, Operation, Region, Use
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter

# TODO: remove these or have better debug support (e.g. log statistics for the pass). For
# now this suffices.
def debug_block(block: Block):
    """Useful for pretty-printing blocks."""
    file = StringIO("")
    printer = Printer(stream=file)
    for op in block.ops:
        printer.print_op(op)
    print("block {")
    print(file.getvalue().strip())
    print("}")

def debug_op(op: Operation):
    """Useful for pretty-printing operations."""
    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(op)
    print(file.getvalue().strip())


@dataclass
class DesymrefException(Exception):
    """
    Exception type if something goes terribly wrong when running `desymref` pass.
    """
    msg: str

    def __str__(self) -> str:
        return f"Exception in desymref pass: {self.msg}."


@dataclass
class DesymrefToDo(Exception):
    """
    Exception type if `desymref` pass ends up in unimplemented execution path.
    """
    msg: str

    def __str__(self) -> str:
        return f"TODO: {self.msg}."


@dataclass
class BlockGraphException(Exception):
    """
    Exception type when processing block graph.
    """
    msg: str

    def __str__(self) -> str:
        return f"Exception in control-flow graph: {self.msg}."


@dataclass
class BlockGraph:
    """
    This class represent CFG-like structure for the xDSL program. It is not
    strictly a CFG because operations have regions, which inside have blocks and
    can be nested. However, for the purpose of getting rid of symref dialect, it
    is sufficient to get all blocks of the program and construct a graph to do
    analysis on.
    """
    func: FuncOp
    """Function for which this control-flow graph is built."""

    num_blocks: int = field(init=False)
    """Number of node (blocks) in control-flow graph."""

    forward_map: Dict[Block, int] = field(init=False)
    """Forward map from block to index."""

    backward_map: List[Block] = field(init=False)
    """Backward map from index to block."""

    pred: List[List[int]] = field(init=False)
    """Predecessors map for each basic block in control-flow graph."""

    succ: List[List[int]] = field(init=False)
    """Successors map for each basic block in control-flow graph."""

    def _register_block(self, block: Block, idx: int) -> int:
        """Registers the block with index `idx` and returns the next index."""
        num_registred_blocks = len(self.backward_map)
        if num_registred_blocks != idx:
            raise BlockGraphException("Trying to create a block with idx = %i but have registered %i blocks" % (idx, num_registred_blocks))

        self.forward_map[block] = idx
        self.backward_map.append(block)
        self.succ.append([])
        self.pred.append([])
        return idx + 1

    def __post_init__(self):
        """
        Constructs control-flow graph for the blocks in the function. While this
        might not be the most-efficient approach, but at least it is relatively
        easy to understand what's going on.
        """
        self.forward_map = {}
        self.backward_map = []
        self.succ = []
        self.pred = []

        # Run BFS to find all the blocks, register them (i.e. associate with an integer),
        # and create edges.
        idx = 0
        visited = set()
        queue = [self.func]
        while queue:
            op = queue.pop(0)
            for region in op.regions:
                for block in region.blocks:
                    # Skip already visited blocks.
                    if block in visited:
                        continue

                    if block not in self.forward_map:
                        block_idx = idx
                        idx = self._register_block(block, idx)
                    else:
                        block_idx = self.forward_map[block]

                    # Find all ops that have a successor and add edges:
                    for op in block.ops:
                        if not op.successors:
                            continue

                        # There are successors, add this operation to the queue to visit later.
                        # For each successor block, create an edge to it and back. Note that this
                        # may involve initializing maps and succ/pred lists.
                        queue.append(op)
                        for succ_block in op.successors:
                            if succ_block not in self.forward_map:
                                succ_idx = idx
                                idx = self._register_block(succ_block, idx)
                            else:
                                succ_idx = self.forward_map[succ_block]

                            self.succ[block_idx].append(succ_idx)
                            self.pred[succ_idx].append(block_idx)

                    # Mark this block as visited.
                    visited.add(block)
        self.num_blocks = idx

    def _dfs(self) -> List[int]:
        """Runs a DFS and return the numbering of blocks in the graph."""
        idx = 0
        dfs_nums = [-1 for _ in range(0, self.num_blocks)]

        start = self.forward_map[self.func.regions[0].blocks[0]]
        visited = [False for _ in range(0, self.num_blocks)]

        stack = [start]
        while stack:
            v = stack.pop()

            # Number the current vertex.
            dfs_nums[v] = idx
            idx += 1
            visited[v] = True

            # Visit neighbors.
            for u in self.succ[v]:
                if not visited[u]:
                    stack.append(u)

        return dfs_nums

    def _dominators(self) -> List[Set[int]]:
        """For each node in the control-flow graph, returns a list of nodes that dominate it."""

        # Initialize to all nodes.
        domintators = [set([i for i in range(0, self.num_blocks)]) for _ in range(0, self.num_blocks)]        
        # Initialize the worlist which would be iteratively modified unless nothing changes.
        start = self.forward_map[self.func.regions[0].blocks[0]]
        worklist = [start]

        while worklist:
            # For each node and its predecessors... 
            y = worklist.pop(0)
            y_preds = self.pred[y]

            # find intersection of dominator sets of all predecessors and union with the current node.
            if not y_preds:
                x = set()
            else:
                x = domintators[y_preds[0]]
                for i in range(1, len(y_preds)):
                    x = x.intersection(domintators[y_preds[i]])    
            new = x.union(set([y]))

            # If the set of dominators has changed, ensure that successors are added to the worklist.
            if new.intersection(domintators[y]):
                domintators[y] = new
                for s in self.succ[y]:
                    worklist.append(s)
        
        # We are done :)
        return domintators

    def dom_tree(self) -> List[List[int]]:
        """Construct the Dominator Tree for the given control-flow graph on blocks."""

        tree = [[] for i in range(0, self.num_blocks)]

        # Precomputation: find dfs numbers and dominator sets.
        dfs_nums = self._dfs()
        dominator_sets = self._dominators()

        for i in range(0, self.num_blocks):
            best_idx = -1
            best_block = None

            # For every node that dominates this node, select one which is
            # closer w.r.t. dfs numbers.
            for dom in dominator_sets[i]:
                idx = dfs_nums[dom]
                if idx > best_idx and dom != i:
                    best_idx = idx
                    best_block = i
            if best_block is not None:
                tree[best_block].append(best_idx)
        return tree


@dataclass
class DeclareAnalyzer:
    """Stores all information about `symref.declare` operaton."""

    def_blocks: List[Block] = field(default_factory=list)
    """Blocks that update (write to) the symbol."""

    use_blocks: List[Block] = field(default_factory=list)
    """Blocks that fetch (read) the symbol."""

    single_update: Optional[Update] = field(default=None)
    """If declare operation is updated only once, stores the update."""

    single_block: Optional[Block] = field(default=None)
    """If the users of declare are all in one block, stores that block."""

    used_in_single_block: bool = field(default=True)
    """Flag to check if all users of declare are in a single block."""

    def _reset(self):
        """Clears all collected information."""
        self.def_blocks = []
        self.use_blocks = []
        self.single_update = None
        self.single_block = None
        self.used_in_single_block = True

    def info(self, delcare_op: Declare):
        """ Computes information about the given delcare operation."""
        self._reset()

        # Go through all users of this declare operation.
        for user in delcare_op.users():

            # Track all blocks that define the value of the declared symbol.
            if isinstance(user, Update):
                self.def_blocks.append(user.parent_block())
                self.single_update = user

            # Track all blocks that fetch the value of the declared symbol.
            if isinstance(user, Fetch):
                self.use_blocks.append(user.parent_block())
            
            # Make sure that we know if this declare is used in a single block only.
            if self.used_in_single_block:
                if not self.single_block:
                    self.single_block = user.parent_block()
                elif self.single_block != user.parent_block():
                    self.used_in_single_block = False


@dataclass
class Desymref:
    rewriter: Rewriter
    """Rewriter to replace and erase operations."""

    declares: List[Declare]
    """List of potentially removable `symref.declare` operations."""

    func: FuncOp
    """Function which we applying the pass to."""

    def _desymref_single_block(self, declare_op: Declare):
        # First, find indices of the users of this declare operation.
        update_ops: List[(int, Update)] = []
        fetch_ops: List[(int, Fetch)] = []
        for index, use in enumerate(declare_op.users()):
            if isinstance(use, Update):
                update_ops.append((index, use))
            if isinstance(use, Fetch):
                fetch_ops.append((index, use))

        # For each fetch, find the closest preceding update.
        # TODO: We can use binary search here, but for now we don't care.
        for i, fetch_op in fetch_ops:
            preceding_update_op = None
            for j, update_op in update_ops:
                if i < j:
                    break
                preceding_update_op = update_op

            if preceding_update_op is None:
                raise DesymrefException("fetch instruction doesn't have a preceding update")
            else:
                # If all ok, symply replace the result of fetch with SSA value from the update.
                self.rewriter.replace_op(fetch_op, [], [update_op.operands[0]])

        # Now, each symref.fetch has been replaced, and we can eliminate update operations.
        for _, update_op in update_ops:
            self.rewriter.erase_op(update_op)
        self.rewriter.erase_op(declare_op)

    def _desymref_single_update(self, declare_op: Declare, analyzer: DeclareAnalyzer):
        update_op = analyzer.single_update

        # For every fetch, replace with the SSA value of the update operation. Note that this is
        # always correct because a single update always dominates all uses (it is done in the
        # entry block just after declare).
        for use in declare_op.users():
            if isinstance(use, Fetch):
                self.rewriter.replace_op(use, [], [update_op.operands[0]])

        self.rewriter.erase_op(update_op)
        self.rewriter.erase_op(declare_op)

    def _get_live_in_blocks(self, declare_op: Declare, analyzer: DeclareAnalyzer, cfg: BlockGraph, def_blocks: Set[Block]):
        """Computes the blocks where the value of declare operation is live in."""
        # TODO: Refactor this!

        live_in_blocks: Set[Block] = set()

        n = len(analyzer.use_blocks)
        worklist = analyzer.use_blocks.copy()
        for i in range(0, n):
            block = worklist[i]
            if block not in def_blocks:
                continue
            # At this point this block both uses and defines!
            for op in block.ops:
                if isinstance(op, Update):
                    if op.attributes["symbol"].data.data != declare_op.attributes["sym_name"].data:
                        continue

                    # Here we have an update for the given declare.
                    i -= 1
                    n -= 1
                    last = worklist.pop()
                    worklist[i] = last
                    break

                if isinstance(op, Fetch):
                    if op.attributes["symbol"].data.data == declare_op.attributes["sym_name"].data:
                        break
        
        # Traverse up to find live blocks.
        while worklist:
            block = worklist.pop()
            if block in live_in_blocks:
                continue

            live_in_blocks.add(block)
            for pred in cfg.pred[block]:
                if pred in def_blocks:
                    continue
                worklist.append(pred)

        return live_in_blocks


    def run(self):
        analyzer = DeclareAnalyzer()

        # Before going through declare operations, make sure we have a dominator
        # tree constructed. 
        cfg = BlockGraph(self.func)
        dt = cfg.dom_tree()

        # DEBUG CFG
        # for b, i in cfg.forward_map.items():
        #     print(i)
        #     debug_block(b)
        
        # for i, idx in enumerate(cfg._dfs()):
        #     print("%i --> %i" % (i, idx))
        
        # for i, doms in enumerate(cfg._dominators()):
        #     print("%i --> %s" % (i, doms))
        
        # tree = cfg.dom_tree()
        # for i, nodes in enumerate(tree):
        #     print("%i --> %s" % (i, nodes))

        for declare_op in self.declares:
            users = declare_op.users()

            # First, delete all unused declare operations.
            if not users:
                self.rewriter.erase_op(declare_op)
                continue

            # If declared symbol is never read, it is also safe to delete it.
            # TODO: this is the responsibility of DSE, but let's do it anyway. Also efficiency
            # can be improved here potentially.
            fetch_ops = list(filter(lambda op: isinstance(op, Fetch), users))
            update_ops = list(filter(lambda op: isinstance(op, Update), users))
            if not fetch_ops:
                # All updates are dead! Remove all of them and then the declaration.
                for update_op in update_ops:
                    self.rewriter.erase_op(update_op)
                self.rewriter.erase_op(declare_op)
                continue
            
            # Gather information about this operation.
            analyzer.info(declare_op)

            # If there is only a single update to this symbol, then we can trivially replace
            # all fetches with the SSA value of the update. This is possible because in Python
            # all variables are intitialized, which means for each declare there exists at least
            # one update which dominates all uses.
            if len(analyzer.def_blocks) == 1:
                self._desymref_single_update(declare_op, analyzer)
                continue

            # If the declared symbol is only used in a single block, simply scan the block
            # top-down to eliminate any updates or fetches.
            if analyzer.used_in_single_block:
                self._desymref_single_block(declare_op)
                continue

            # TODO: maybe compute block numbers here

            # TODO: Uncomment when all block arguments are added.

            # Below we compute where the value is live in.
            # live_blocks: Set[Block] = set()
            # def_blocks: Set[Block] = set()
            # for b in analyzer.def_blocks:
            #     def_blocks.add(b)
            # self._find_live_blocks(declare_op, analyzer, cfg, def_blocks, live_blocks)
            # for b in live_blocks:
            #     debug_block(b)
            # raise DesymrefToDo("not implemented SSA-consturuction")


@dataclass
class DesymrefPass:
    """A pass which operates on a function and removes all symref operations."""

    @staticmethod
    def run(func: FuncOp) -> bool:
        changed = False

        i = 0
        while (i < 1):
            # First, find declares that we will try to remove.
            declares: List[Declare] = []
            for op in func.body.blocks[0].ops:
                if isinstance(op, Declare):
                    declares.append(op)

            # If the list of declares is empty, we are done.
            if not declares:
                break

            # Otherwise, try to remove these declares.
            Desymref(Rewriter(), declares, func).run()
            changed = True

            i += 1

        return changed
