# from lark import ParseTree, Token

# from convert_x86_to_mlir import X86Converter
from xdsl.dialects.x86 import C_JmpOp, FallthroughOp
from xdsl.dialects.x86.ops import ConditionalJumpOperation
from xdsl.dialects.x86_func import CallOp, RetOp
from xdsl.ir import Block, Region

# from xdsl.tools.convert_x86_to_mlir import X86Converter


class CFGError(Exception):
    def __init__(self, *args: object):
        super().__init__(*args)


def cfg_pprint(region: Region):
    """Given an x86 Region, inspect JUMPs and produce list of edges between blocks"""
    for i, block in enumerate(region.blocks):
        print(f"----- BLOCK {i} -----")

        for op in block.ops:
            print(f"{op.name}")

        last_op = block.last_op
        if isinstance(last_op, (C_JmpOp, FallthroughOp)):
            dest = region.blocks.index(last_op.successor)
            print(f"* UNCOND JUMP/FALLTHROUGH -> BLOCK {dest}")
        elif isinstance(last_op, ConditionalJumpOperation):
            t_dest = region.blocks.index(last_op.then_block)
            e_dest = region.blocks.index(last_op.else_block)
            print(f"* TRUE  -> BLOCK {t_dest}")
            print(f"* FALSE -> BLOCK {e_dest}")
        print()


def build_adj(
    region: Region,
) -> dict[Block, tuple[()] | tuple[Block] | tuple[Block, Block]]:
    """Build adjacency lists of blocks"""
    # NB not strictly necessary, but this simplifies the Block structures

    # Only three types of edges: no edge (ret); unconditional/fallthrough edge; and then/else edge
    adj: dict[Block, tuple[()] | tuple[Block] | tuple[Block, Block]] = {}

    for block in region.blocks:
        last_op = block.last_op
        # last op should be one of three things:
        if isinstance(last_op, RetOp):
            # ret
            adj[block] = ()
        elif isinstance(last_op, (C_JmpOp, FallthroughOp)):
            # unconditional jump/fallthrough
            adj[block] = (last_op.successor,)
        elif isinstance(last_op, ConditionalJumpOperation):
            # conditional jump
            adj[block] = (last_op.then_block, last_op.else_block)
        else:
            raise CFGError("Last op not one of (RetOp, FallthroughOp, jump)")

    return adj


def detect_loops(
    adj: dict[Block, tuple[()] | tuple[Block] | tuple[Block, Block]], start: Block
) -> dict[Block, tuple[Block, ...]]:
    """Detect backedges and return dictionary of `first block in loop` -> `blocks in loop`"""

    d: dict[Block, tuple[Block, ...]] = {}

    def rec(cur: Block, vis: tuple[Block, ...]):
        if cur in vis:
            # found loop. ie cur=x and we have vis=(..., x, a, b, c). so the loop is x->a->b->c.
            i = vis.index(cur)
            d[cur] = vis[i:]
            return
        for nx in adj[cur]:
            rec(nx, vis + (cur,))

    rec(start, ())
    return d


def group_functions(
    blocks: list[Block], label_map: dict[str, int]
) -> list[list[Block]]:
    if not blocks:
        return []

    # assume the functions only have one entry point
    # want to maintain order here, so use dict
    entry_points = {
        blocks[0]: None
    }  # blocks called by a CallOp; should be one-to-one to functions in region
    for block in blocks:
        for op in block.ops:
            if isinstance(op, CallOp):
                callee = str(op.callee)[1:]
                if callee not in label_map:
                    raise CFGError(f"callee label not recognised: {callee}")
                ep = blocks[label_map[callee]]
                if ep not in entry_points:
                    entry_points[ep] = None

    functions: list[list[Block]] = []
    visited: set[Block] = (
        set()
    )  # assume no shared blocks, so we can maintain a global visited set

    for entry in entry_points.keys():
        this_func: list[Block] = []
        stack = [entry]

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            this_func.append(cur)

            last_op = cur.last_op

            if isinstance(last_op, (C_JmpOp, FallthroughOp)):
                stack.append(last_op.successor)
            elif isinstance(last_op, ConditionalJumpOperation):
                stack.append(last_op.then_block)
                stack.append(last_op.else_block)

        functions.append(this_func)

    return functions


if __name__ == "__main__":
    # --- 1. Setup Blocks ---
    # Function: main
    b_main = Block()
    # Function: math_func
    b_math = Block()
    # Function: loop_func
    b_loop_entry = Block()
    b_loop_top = Block()
    b_loop_exit = Block()

    # --- 2. Create the Label Map ---
    # This mimics what the parser would produce
    label_map = {
        "main": 0,
        "math_func": 1,
        "loop_func": 2,
        "loop_top": 3,
        "loop_exit": 4,
    }
    all_blocks = [b_main, b_math, b_loop_entry, b_loop_top, b_loop_exit]

    # --- 3. Populate Blocks with Instructions ---

    # Block: main
    # Note: CallOp takes (callee, arguments, return_types)
    # Passing empty lists for args/returns for testing purposes
    b_main.add_op(CallOp("math_func", [], []))
    b_main.add_op(CallOp("loop_func", [], []))
    b_main.add_op(RetOp())

    # Block: math_func
    b_math.add_op(RetOp())

    # Block: loop_func (Entry)
    b_loop_entry.add_op(FallthroughOp([], b_loop_top))

    # Block: loop_top (The Loop)
    # We use a dummy jump back to itself to test the 'visited' logic
    # and a jump to exit to test conditional logic
    b_loop_top.add_op(C_JmpOp([], b_loop_top))  # Simulating a back-edge loop

    # Block: loop_exit
    b_loop_exit.add_op(RetOp())

    groups = group_functions(all_blocks, label_map)

    # --- 5. Verify Results ---
    print(f"Functions detected: {len(groups)}")
    for i, group in enumerate(groups):
        # Get the entry block's "name" by reversing the label_map
        entry_block = group[0]
        idx = all_blocks.index(entry_block)
        name = [k for k, v in label_map.items() if v == idx][0]
        print(f"Function {i} ({name}): contains {len(group)} blocks")
