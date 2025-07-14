import pytest

from xdsl.transforms.memref_stream_interleave import factors


@pytest.mark.parametrize(
    "num,expected_factors",
    [
        (-1, ()),
        (0, ()),
        (1, (1,)),
        (2, (1, 2)),
        (3, (1, 3)),
        (4, (1, 2, 4)),
        (6, (1, 2, 3, 6)),
        (24, (1, 2, 3, 4, 6, 8, 12, 24)),
    ],
)
def test_factors_parametrized(num: int, expected_factors: tuple[int, ...]):
    assert factors(num) == expected_factors
