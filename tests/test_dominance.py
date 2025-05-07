import pytest

from xdsl.context import Context
from xdsl.dialects import get_all_dialects
from xdsl.irdl.dominance import strictly_dominates
from xdsl.parser import Parser

ctx = Context()
ctx.register_dialect("test", get_all_dialects()["test"])


# Create a block graph as in https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332

op = Parser(
    ctx,
    """
"test.op"() ({
^1():
  "test.op"()[^2] : () -> ()
^2():
  "test.op"()[^3, ^4, ^6] : () -> ()
^3():
  "test.op"()[^5] : () -> ()
^4():
  "test.op"()[^5] : () -> ()
^5():
  "test.op"()[^2] : () -> ()
^6():
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
