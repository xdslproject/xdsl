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


# This functionality is implemented in the raiser instead.
# def detect_loops(
#     adj: dict[Block, tuple[()] | tuple[Block] | tuple[Block, Block]], start: Block
# ) -> dict[Block, tuple[Block, ...]]:
#     """Detect backedges and return dictionary of `first block in loop` -> `blocks in loop`"""

#     d: dict[Block, tuple[Block, ...]] = {}

#     def rec(cur: Block, vis: tuple[Block, ...]):
#         if cur in vis:
#             # found loop. ie cur=x and we have vis=(..., x, a, b, c). so the loop is x->a->b->c.
#             i = vis.index(cur)
#             d[cur] = vis[i:]
#             return
#         for nx in adj[cur]:
#             rec(nx, vis + (cur,))

#     rec(start, ())
#     return d


def collate_function(entry: Block, visited: set[Block]) -> list[Block]:
    """Given an entry point block, find all reachable blocks"""
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

    return this_func


def group_functions(
    blocks: list[Block], label_map: dict[str, int]
) -> list[Block | list[Block]]:
    """Given a list of blocks, group them into functions if they are reachable
    and return a sequence of blocks, preserving the order of the given list (but with functions grouped).
    NB: isinstance(result[i], Block) <=> result[i] is unreachable! (can be annotated)"""

    if not blocks:
        return []

    # assume the functions only have one entry point
    # want to maintain the order that they appear in source, so use dict with (block: block_idx);
    # and later we will sort by block_idx
    entry_points = set(
        [blocks[0]]
    )  # blocks called by a CallOp; should be one-to-one to functions in region
    for block in blocks:
        for op in block.ops:
            if isinstance(op, CallOp):
                callee = str(op.callee)[1:]
                if callee not in label_map:
                    raise CFGError(f"callee label not recognised: {callee}")
                ep = blocks[label_map[callee]]
                if ep not in entry_points:
                    entry_points.add(ep)

    grouped: list[Block | list[Block]] = []
    # assume no shared blocks, so we can maintain a global visited set
    visited: set[Block] = set()

    for block in blocks:
        if block in visited:
            continue
        if block in entry_points:
            grouped.append(collate_function(block, visited))
        else:
            grouped.append(block)
            visited.add(block)

    return grouped


if __name__ == "__main__":
    # --- Setup Blocks ---
    b_main = Block()  # Entry 0
    b_orphan_1 = Block()  # Unreachable between functions
    b_helper = Block()  # Entry 2 (Called by main)
    b_helper_cont = Block()  # Reachable from helper (multi-block function)
    b_orphan_2 = Block()  # Unreachable at the end

    # --- Label Map ---
    label_map = {"main": 0, "orphan_1": 1, "helper": 2, "helper_cont": 3, "orphan_2": 4}
    all_blocks = [b_main, b_orphan_1, b_helper, b_helper_cont, b_orphan_2]

    # --- Instructions & Logic ---

    # Block 0: main (Entry point)
    # Calls 'helper', but has no successors (it returns)
    b_main.add_op(CallOp("helper", [], []))
    b_main.add_op(RetOp())

    # Block 1: orphan_1 (Dead code)
    # Nothing points here, and it doesn't point anywhere.
    b_orphan_1.add_op(RetOp())

    # Block 2: helper (Entry point)
    # Points to Block 3
    b_helper.add_op(FallthroughOp([], b_helper_cont))

    # Block 3: helper_cont (Part of helper)
    b_helper_cont.add_op(RetOp())

    # Block 4: orphan_2 (Dead code)
    b_orphan_2.add_op(RetOp())

    # --- Run Grouping ---
    result = group_functions(all_blocks, label_map)

    # --- Verification ---
    print(f"Resulting items: {len(result)}")
    for i, item in enumerate(result):
        if isinstance(item, list):
            entry_idx = all_blocks.index(item[0])
            name = [k for k, v in label_map.items() if v == entry_idx][0]
            print(f"Index {i}: [Function Group] {name} ({len(item)} blocks)")
        else:
            orphan_idx = all_blocks.index(item)
            name = [k for k, v in label_map.items() if v == orphan_idx][0]
            print(f"Index {i}: [Unreachable Block] {name}")
