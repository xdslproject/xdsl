"""
Signed numbers are stored as Two's complement, meaning that the highest bit is used as
the sign. Here's a table of values for a three-bit two's complement integer type:

|------|----------|--------|----------|
| bits | unsigned | signed | signless |
|------|----------|--------|----------|
|  000 |     0    |   +0   |    +0    |
|  001 |     1    |   +1   |    +1    |
|  010 |     2    |   +2   |    +2    |
|  011 |     3    |   +3   |    +3    |
|  100 |     4    |   -4   | +4 or -4 |
|  101 |     5    |   -3   | +5 or -3 |
|  110 |     6    |   -2   | +6 or -2 |
|  111 |     7    |   -1   | +7 or -1 |
|------|----------|--------|----------|

See [wikipedia](https://en.wikipedia.org/wiki/Two%27s_complement).

We follow LLVM and MLIR in having a concept of signless integers.

See external [documentation](https://mlir.llvm.org/docs/Rationale/Rationale/#integer-signedness-semantics).

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
    """
    The maximum representable value + 1.
    """
    return 1 << bitwidth


def signed_lower_bound(bitwidth: int) -> int:
    """
    The minimum representable value.
    """
    return -((1 << bitwidth) >> 1)


def signed_upper_bound(bitwidth: int) -> int:
    """
    The maximum representable value + 1.
    """
    return 1 << max(bitwidth - 1, 0)


def unsigned_value_range(bitwidth: int) -> tuple[int, int]:
    """
    For a given bitwidth, returns a tuple `(min, max)`, such that unsigned integers of
    this bitwidth are in the range [`min`, `max`).
    """
    return 0, unsigned_upper_bound(bitwidth)


def signed_value_range(bitwidth: int) -> tuple[int, int]:
    """
    For a given bitwidth, returns a tuple `(min, max)`, such that signed integers of
    this bitwidth are in the range [`min`, `max`).
    """
    min_value = signed_lower_bound(bitwidth)
    max_value = signed_upper_bound(bitwidth)

    return min_value, max_value


def signless_value_range(bitwidth: int) -> tuple[int, int]:
    """
    For a given bitwidth, returns a tuple `(min, max)`, such that signless integers of
    this bitwidth are in the range [`min`, `max`).

    Signless integers are semantically just bit patterns, and don't represent an
    integer until being converted to signed or unsigned explicitly, so the representable
    range is the union of the signed and unsigned representable ranges.
    """
    min_value = signed_lower_bound(bitwidth)
    max_value = unsigned_upper_bound(bitwidth)

    return min_value, max_value


def to_unsigned(signless: int, bitwidth: int) -> int:
    """
    Transforms values in range `[MIN_SIGNED, MAX_UNSIGNED)` to range `[0,
    MAX_UNSIGNED)`.
    """
    # Normalise to unsigned range by adding the unsigned range and taking the remainder
    modulus = unsigned_upper_bound(bitwidth)
    return (signless + modulus) % modulus


def to_signed(signless: int, bitwidth: int) -> int:
    """
    Transforms values in range `[MIN_SIGNED, MAX_UNSIGNED)` to range `[MIN_SIGNED,
    MAX_SIGNED)`.
    """
    # Normalise to unsigned range by adding the unsigned range and taking the remainder
    modulus = unsigned_upper_bound(bitwidth)
    half_modulus = modulus >> 1
    return (signless + half_modulus) % modulus - half_modulus
