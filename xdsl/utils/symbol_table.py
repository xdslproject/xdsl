"""
Helper methods and classes to reason about operations that refer to other operations.
"""

from collections.abc import Iterator

from xdsl import traits
from xdsl.ir import Operation


def walk_symbol_table(op: Operation) -> Iterator[Operation]:
    """
    Walk all of the operations nested under, and including, the given operation, without
    traversing into any nested symbol tables.
    """
    yield op
    if op.has_trait(traits.SymbolTable):
        return

    regions = list(op.regions)

    while regions:
        region = regions.pop()
        for block in region.blocks:
            for op in block.ops:
                yield op

                if not op.has_trait(traits.SymbolTable):
                    regions.extend(op.regions)
