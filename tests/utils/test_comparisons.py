from xdsl.utils.comparisons import (
    signed_lower_bound,
    signed_upper_bound,
    to_signed,
    to_unsigned,
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
    assert list(SIGNED_RANGE) == [unsigned_to_signed(u) for u in UNSIGNED_RANGE]


def test_conversion():
    assert to_unsigned(SIGNED_LOWER_BOUND, BITWIDTH) == SIGNED_UPPER_BOUND
    assert to_unsigned(-1, BITWIDTH) == UNSIGNED_UPPER_BOUND - 1
    assert to_unsigned(0, BITWIDTH) == 0
    assert to_unsigned(1, BITWIDTH) == 1
    assert to_unsigned(SIGNED_UPPER_BOUND, BITWIDTH) == SIGNED_UPPER_BOUND
    assert to_unsigned(UNSIGNED_UPPER_BOUND, BITWIDTH) == 0

    assert to_signed(SIGNED_LOWER_BOUND, BITWIDTH) == SIGNED_LOWER_BOUND
    assert to_signed(-1, BITWIDTH) == -1
    assert to_signed(0, BITWIDTH) == 0
    assert to_signed(1, BITWIDTH) == 1
    assert to_signed(SIGNED_UPPER_BOUND, BITWIDTH) == SIGNED_LOWER_BOUND
    assert to_signed(UNSIGNED_UPPER_BOUND, BITWIDTH) == 0
