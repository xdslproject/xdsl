import pytest

from xdsl.interpreters.comparisons import signed_less_than, unsigned_less_than

BITWIDTH = 2
SIGNED_UPPER_BOUND = 1 << (BITWIDTH - 1)
SIGNED_LOWER_BOUND = -SIGNED_UPPER_BOUND
UNSIGNED_TO_SIGNED = list(range(SIGNED_LOWER_BOUND, SIGNED_UPPER_BOUND))


def test_bitwidth_2_values():
    """
    Above calculations are correct for bitwidth 2.
    """
    assert UNSIGNED_TO_SIGNED == [-2, -1, 0, 1]


@pytest.mark.parametrize("u_lhs", range(len(UNSIGNED_TO_SIGNED)))
@pytest.mark.parametrize("u_rhs", range(len(UNSIGNED_TO_SIGNED)))
def test_signed_unsigned_comparison(u_lhs: int, u_rhs: int):
    """
    For all possible bit patterns of a given bitwidth, signed and unsigned comparisons
    return expected values.
    """
    s_lhs = UNSIGNED_TO_SIGNED[u_lhs]
    s_rhs = UNSIGNED_TO_SIGNED[u_rhs]
    assert (u_lhs < u_rhs) == unsigned_less_than(s_lhs, s_rhs)
    assert (u_lhs < u_rhs) == unsigned_less_than(u_lhs, u_rhs)
    assert (s_lhs < s_rhs) == signed_less_than(s_lhs, s_rhs, BITWIDTH)
    assert (s_lhs < s_rhs) == signed_less_than(u_lhs, u_rhs, BITWIDTH)
