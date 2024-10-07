from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from xdsl.ir import Block, Operation
from xdsl.traits import IsTerminator


@dataclass(init=False)
class PostOrderIterator(Iterator[Block]):
    stack: deque[Block]
    seen: set[Block]

    def __init__(self, block: Block) -> None:
        self.stack = deque((block,))
        self.seen = {block}

    def __iter__(self) -> Iterator[Block]:
        return self

    def __next__(self) -> Block:
        if not self.stack:
            raise StopIteration
        block = self.stack.popleft()
        term = block.last_op
        if isinstance(term, Operation) and term.has_trait(IsTerminator()):
            self.stack.extend(x for x in term.successors if x not in self.seen)
            self.seen.update(term.successors)
        return block
