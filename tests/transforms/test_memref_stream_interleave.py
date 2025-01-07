import pytest

from xdsl.transforms.memref_stream_interleave import interleave_index_and_factor


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
