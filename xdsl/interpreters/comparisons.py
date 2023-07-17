"""
    Signed numbers are stored as Two's complement, meaning that the highest bit is used as
    the sign. Here's a table of values for a three-bit two's complement integer type:

    |------|----------|--------|
    | bits | unsigned | signed |
    |------|----------|--------|
    |  000 |     0    |   +0   |
    |  001 |     1    |   +1   |
    |  010 |     2    |   +2   |
    |  011 |     3    |   +3   |
    |  100 |     4    |   -4   |
    |  101 |     5    |   -3   |
    |  110 |     6    |   -2   |
    |  111 |     7    |   -1   |
    |------|----------|--------|

    https://en.wikipedia.org/wiki/Two%27s_complement

    We follow LLVM and MLIR in having a concept of signless integers:

    https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics

    The main idea is to not have the signedness be a property of the type of the value,
    and rather be a property of the operation. That means that a signless value can be
    interpreted as either a signed or unsigned value at runtime, depending on the
    operation that acts on it.

    During interpretation, this gets a little tricky, as the same bit pattern can be
    interpreted as two runtime values, meaning that comparing signless values is a little
    involved. For example, a signless value of 5 is equal to a signless value of -3, since
    their bit representations are the same.
"""


def unsigned_upper_bound(bitwidth: int) -> int:
    return 1 << bitwidth


def signed_lower_bound(bitwidth: int) -> int:
    return -(1 << (bitwidth - 1))


def signed_upper_bound(bitwidth: int) -> int:
    return 1 << (bitwidth - 1)


def to_unsigned(signless: int, bitwidth: int) -> int:
    """
    Transforms values in range [MIN_SIGNED, MAX_UNSIGNED] to range [0, MAX_UNSIGNED].
    """
    # Normalise to unsigned range by adding the unsigned range and taking the remainder
    m = unsigned_upper_bound(bitwidth)
    return (signless + m) % m


def signless_equal(signless_lhs: int, signless_rhs: int, bitwidth: int) -> bool:
    """
    If the values are in the range [0, MAX_SIGNED) then normal comparison will work.
    If not, we have to normalise to either the signed or unsigned range.
    """
    return to_unsigned(signless_lhs, bitwidth) == to_unsigned(signless_rhs, bitwidth)


def unsigned_less_than(signless_lhs: int, signless_rhs: int) -> bool:
    """
    When signs differ, the sign bit is treated as the highest value bit, meaning that the
    negative number will become the larger positive number.
    """
    lhs_is_negative = signless_lhs < 0
    rhs_is_negative = signless_rhs < 0
    if lhs_is_negative == rhs_is_negative:
        # Same signedness, normal comparison will work
        return signless_lhs < signless_rhs
    else:
        # Negative number will be greater when bitcast to unsigned
        return lhs_is_negative


def signed_less_than(lhs: int, rhs: int, bitwidth: int) -> bool:
    """
    The highest bit of the unsigned representation will be treated as the sign bit.
    """
    lhs_is_negative = lhs >= 1 << bitwidth - 1
    rhs_is_negative = rhs >= 1 << bitwidth - 1
    if lhs_is_negative == rhs_is_negative:
        # Same signedness, normal comparison will work
        return lhs < rhs
    else:
        # Negative number is smaller
        return rhs_is_negative
