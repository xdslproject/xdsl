import pytest

from xdsl.transforms.memref_stream_interleave import (
    factors,
    interleave_index_and_factor,
)


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


@pytest.mark.parametrize(
    "indices_and_factors,expected_res",
    [
        ((), None),
        (((0, 1),), (0, 1)),
        (((0, 1), (1, 1)), (1, 1)),
        (((0, 1), (1, 2), (1, 3)), (1, 3)),
        (((0, 1), (1, 4), (1, 5)), (1, 5)),
        (((0, 1), (0, 3), (1, 4), (1, 11), (1, 44)), (1, 4)),
    ],
)
def test_index_and_factor(
    indices_and_factors: tuple[tuple[int, int], ...],
    expected_res: tuple[int, int] | None,
):
    assert interleave_index_and_factor(indices_and_factors, 4) == expected_res
