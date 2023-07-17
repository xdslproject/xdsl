import pytest

from xdsl.interpreters.comparisons import (
    signed_less_than,
    signed_lower_bound,
    signed_upper_bound,
    unsigned_less_than,
    unsigned_upper_bound,
)

BITWIDTH = 2
UNSIGNED_UPPER_BOUND = unsigned_upper_bound(BITWIDTH)
SIGNED_UPPER_BOUND = signed_upper_bound(BITWIDTH)
SIGNED_LOWER_BOUND = signed_lower_bound(BITWIDTH)

UNSIGNED_RANGE = range(UNSIGNED_UPPER_BOUND)
SIGNED_RANGE = range(SIGNED_LOWER_BOUND, SIGNED_UPPER_BOUND)


def unsigned_to_signed(u: int) -> int:
    return u + SIGNED_LOWER_BOUND


def test_bitwidth_2_values():
    """
    Above calculations are correct for bitwidth 2.
    """
    assert len(UNSIGNED_RANGE) == len(SIGNED_RANGE)
    for i, u in zip(SIGNED_RANGE, UNSIGNED_RANGE):
        assert i == unsigned_to_signed(u)


@pytest.mark.parametrize("u_lhs", UNSIGNED_RANGE)
@pytest.mark.parametrize("u_rhs", UNSIGNED_RANGE)
def test_signed_unsigned_comparison(u_lhs: int, u_rhs: int):
    """
    For all possible bit patterns of a given bitwidth, signed and unsigned comparisons
    return expected values.
    """
    s_lhs = unsigned_to_signed(u_lhs)
    s_rhs = unsigned_to_signed(u_rhs)
    assert (u_lhs < u_rhs) == unsigned_less_than(s_lhs, s_rhs)
    assert (u_lhs < u_rhs) == unsigned_less_than(u_lhs, u_rhs)
    assert (s_lhs < s_rhs) == signed_less_than(s_lhs, s_rhs, BITWIDTH)
    assert (s_lhs < s_rhs) == signed_less_than(u_lhs, u_rhs, BITWIDTH)
