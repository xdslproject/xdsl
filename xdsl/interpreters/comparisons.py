"""
    Signed numbers are stored as two's complement, meaning that the highest bit is used as
    the sign. Here's a table of values for a three-bit two's complement integer type:

    |------|----------|--------|
    | bits | unsigned | signed |
    |------|----------|--------|
    |  000 |     0    |   +0   |
    |  001 |     1    |   +1   |
    |  010 |     2    |   +2   |
    |  011 |     3    |   +3   |
    |  100 |     4    |   -1   |
    |  101 |     5    |   -2   |
    |  110 |     6    |   -3   |
    |  111 |     7    |   -4   |
    |------|----------|--------|

    When the signs of operands are equal, signed comparison works same as unsigned.
    When the signs differ, the comparison is more involved, detailed in each of the
    functions below.

    Here is an explanation of signedness semantics:

    https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics
"""


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
