from collections.abc import Iterator
from dataclasses import dataclass

from typing_extensions import Self

from xdsl.ir import Block, Operation
from xdsl.traits import IsTerminator


@dataclass(init=False)
class PostOrderIterator(Iterator[Block]):
    """
    Iterates through blocks in a region by a depth first search in post order.
    Each block's successors are processed before the block itself (unless they
    have already been encounted).

    Blocks that are not reachable from the starting block will not appear in the
    iteration.
    """

    stack: list[tuple[Block, bool]]
    seen: set[Block]

    def __init__(self, block: Block) -> None:
        self.stack = [(block, False)]
        self.seen = {block}

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Block:
        if not self.stack:
            raise StopIteration
        (block, visited) = self.stack.pop()
        while not visited:
            self.stack.append((block, True))
            term = block.last_op
            if isinstance(term, Operation) and term.has_trait(IsTerminator()):
                self.stack.extend(
                    (x, False) for x in reversed(term.successors) if x not in self.seen
                )
                self.seen.update(term.successors)
            # stack cannot be empty here
            (block, visited) = self.stack.pop()
        return block
