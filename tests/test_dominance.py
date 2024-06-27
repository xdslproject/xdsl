from typing import cast

import pytest

from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects
from xdsl.ir import Operation
from xdsl.irdl.dominance import properly_dominates
from xdsl.parser import Parser

ctx = MLContext()
ctx.register_dialect("test", get_all_dialects()["test"])

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
    (
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
    ),
)
def test_region_properly_dominates_block(a: int, b: int, expected: bool):
    """
    Test in-region block dominance.
    """
    # Create blocks as in https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332

    assert properly_dominates(blocks[a - 1], blocks[b - 1]) == expected


nested = Parser(
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
  "test.op"()[^2] ({
  ^7():
    "test.op"() : () -> ()
  }) : () -> ()
^6():
  "test.op"() : () -> ()
}) : () -> ()
""",
).parse_op()

blocks = nested.regions[0].blocks
blocks += cast(Operation, blocks[5 - 1].first_op).regions[0].blocks


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    (
        (1, 1, False),
        (1, 2, True),
        (1, 3, True),
        (1, 4, True),
        (1, 5, True),
        (1, 6, True),
        (1, 7, True),
        (2, 2, False),
        (2, 3, True),
        (2, 4, True),
        (2, 5, True),
        (2, 6, True),
        (2, 7, True),
        (3, 3, False),
        (3, 4, False),
        (3, 5, False),
        (3, 6, False),
        (3, 7, False),
        (4, 4, False),
        (4, 5, False),
        (4, 6, False),
        (4, 7, False),
        (5, 5, False),
        (5, 6, False),
        (5, 7, True),
        (6, 6, False),
        (6, 7, False),
        (7, 7, False),
    ),
)
def test_nested_properly_dominates_block(a: int, b: int, expected: bool):
    """
    Test in-region block dominance.
    """
    # Create blocks as in https://en.wikipedia.org/w/index.php?title=Dominator_(graph_theory)&oldid=1189814332

    assert properly_dominates(blocks[a - 1], blocks[b - 1]) == expected
