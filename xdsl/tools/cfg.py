from lark import ParseTree, Token

from xdsl.dialects.x86 import C_JmpOp, FallthroughOp
from xdsl.dialects.x86.ops import ConditionalJumpOperation
from xdsl.dialects.x86_func import RetOp
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
    # TODO: natural loops instead

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


def group_functions(blocks: list[Block]) -> dict[int, list[Block]]: ...


if __name__ == "__main__":
    jambloat = ParseTree(
        "program",
        [
            ParseTree("label", [Token("LABELNAME", "jambloat"), Token(":", ":")]),
            ParseTree(
                "instruction",
                [Token("opcode", "add"), Token("REG", "rax"), Token("REG", "rbx")],
            ),
            ParseTree(
                "instruction",
                [Token("opcode", "mov"), Token("REG", "rcx"), Token("REG", "rax")],
            ),
            ParseTree(
                "instruction",
                [Token("opcode", "cmp"), Token("REG", "rbx"), Token("REG", "rcx")],
            ),
            ParseTree(
                "instruction",
                [
                    Token("opcode", "lea"),
                    Token("REG", "rbx"),
                    ParseTree("mem", [Token("REG", "rax")]),
                ],
            ),
            ParseTree(
                "instruction",
                [
                    Token("opcode", "mov"),
                    Token("REG", "rsp"),
                    ParseTree(
                        "mem", [Token("REG", "rbx"), Token("+", "+"), Token("IMM", "3")]
                    ),
                ],
            ),
            ParseTree("label", [Token("LABELNAME", "jambloat2"), Token(":", ":")]),
            ParseTree(
                "instruction",
                [
                    Token("opcode", "add"),
                    Token("REG", "r8"),
                    ParseTree(
                        "mem",
                        [Token("REG", "rsp"), Token("-", "-"), Token("IMM", "10")],
                    ),
                ],
            ),
            ParseTree(
                "instruction",
                [
                    Token("opcode", "cmp"),
                    ParseTree(
                        "mem",
                        [Token("REG", "rsp"), Token("-", "-"), Token("IMM", "10")],
                    ),
                    Token("IMM", "100"),
                ],
            ),
            ParseTree("instruction", [Token("opcode", "inc"), Token("REG", "rsp")]),
            ParseTree("instruction", [Token("opcode", "push"), Token("REG", "rbp")]),
            ParseTree("instruction", [Token("opcode", "pop"), Token("REG", "r15")]),
            ParseTree("instruction", [Token("opcode", "push"), Token("REG", "rsp")]),
            ParseTree(
                "instruction",
                [Token("opcode", "neg"), ParseTree("mem", [Token("REG", "rax")])],
            ),
            ParseTree(
                "instruction",
                [
                    Token("opcode", "push"),
                    ParseTree(
                        "mem",
                        [Token("REG", "r15"), Token("-", "-"), Token("IMM", "10")],
                    ),
                ],
            ),
            ParseTree(
                "instruction", [Token("opcode", "jne"), Token("LABELNAME", "jambloat3")]
            ),
            ParseTree(
                "instruction",
                [
                    Token("opcode", "pop"),
                    ParseTree(
                        "mem",
                        [Token("REG", "rbp"), Token("+", "+"), Token("IMM", "10")],
                    ),
                ],
            ),
            ParseTree(
                "instruction", [Token("opcode", "jmp"), Token("LABELNAME", "jambloat2")]
            ),
            ParseTree("label", [Token("LABELNAME", "jambloat3"), Token(":", ":")]),
            ParseTree("instruction", [Token("opcode", "ret")]),
        ],
    )

    # converter = X86Converter()
    # res = converter.convert(jambloat)
    # cfg_pprint(res)
    # d = detect_loops(build_adj(res), res.first_block)
    # for k, v in d.items():
    #     s = "LOOP HEAD = " + str(res.get_block_index(k)) + ": "
    #     for vv in v:
    #         s += str(res.get_block_index(vv)) + " "
    #     print(s)
