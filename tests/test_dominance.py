import pytest

from xdsl.context import Context
from xdsl.dialects import get_all_dialects
from xdsl.irdl.dominance import strictly_dominates, strictly_postdominates
from xdsl.parser import Parser

ctx = Context()
ctx.register_dialect("test", get_all_dialects()["test"])


# Create a block graph as in https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332

op = Parser(
    ctx,
    """
"test.op"() ({
^bb1():
  "test.op"()[^bb2] : () -> ()
^bb2():
  "test.op"()[^bb3, ^bb4, ^bb6] : () -> ()
^bb3():
  "test.op"()[^bb5] : () -> ()
^bb4():
  "test.op"()[^bb5] : () -> ()
^bb5():
  "test.op"()[^bb2] : () -> ()
^bb6():
  "test.op"() : () -> ()
}) : () -> ()
""",
).parse_op()

blocks = op.regions[0].blocks


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (1, 1, False),
        (1, 2, True),
        (1, 3, True),
        (1, 4, True),
        (1, 5, True),
        (1, 6, True),
        (2, 2, False),
        (2, 3, True),
        (2, 4, True),
        (2, 5, True),
        (2, 6, True),
        (3, 3, False),
        (3, 4, False),
        (3, 5, False),
        (3, 6, False),
        (4, 4, False),
        (4, 5, False),
        (4, 6, False),
        (5, 5, False),
        (5, 6, False),
        (6, 6, False),
    ],
)
def test_region_strictly_dominates_block(a: int, b: int, expected: bool):
    """
    Test in-region block dominance.
    """
    assert strictly_dominates(blocks[a - 1], blocks[b - 1]) == expected


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        # Block 6 (exit) post-dominates everything
        (6, 6, False),  # not strictly
        (6, 1, True),
        (6, 2, True),
        (6, 3, True),
        (6, 4, True),
        (6, 5, True),
        # Block 2 post-dominance (hub before exit)
        (2, 2, False),
        (2, 1, True),
        (2, 3, True),
        (2, 4, True),
        (2, 5, True),
        (2, 6, False),
        (1, 1, False),
        (1, 2, False),
        (1, 3, False),
        (1, 4, False),
        (1, 5, False),
        (1, 6, False),
        (3, 3, False),
        (3, 1, False),
        (3, 2, False),
        (3, 4, False),
        (3, 5, False),
        (3, 6, False),
        (4, 4, False),
        (4, 1, False),
        (4, 2, False),
        (4, 3, False),
        (4, 5, False),
        (4, 6, False),
        (5, 5, False),
        (5, 1, False),
        (5, 2, False),
        (5, 3, True),
        (5, 4, True),
        (5, 6, False),
    ],
)
def test_region_postdominates_block(a: int, b: int, expected: bool):
    """
    Test in-region block post-dominance.

    For the graph:
    1 -> 2 -> {3,4,6}
    3 -> 5 -> 2 (cycle)
    4 -> 5 -> 2 (cycle)
    6 (exit)

    Post-dominance analysis:
    - 6 post-dominates everything
    - 2 post-dominates everything but 6
    - 5 post-dominates 3 and 4
    """
    assert strictly_postdominates(blocks[a - 1], blocks[b - 1]) == expected
