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

from xdsl.dialects.builtin import Signedness


def unsigned_upper_bound(bitwidth: int) -> int:
    """
    The maximum representable value + 1.
    """
    _, ub = Signedness.UNSIGNED.value_range(bitwidth)
    return ub


def signed_lower_bound(bitwidth: int) -> int:
    """
    The minimum representable value.
    """
    lb, _ = Signedness.SIGNED.value_range(bitwidth)
    return lb


def signed_upper_bound(bitwidth: int) -> int:
    """
    The maximum representable value + 1.
    """
    _, ub = Signedness.SIGNED.value_range(bitwidth)
    return ub


def to_unsigned(signless: int, bitwidth: int) -> int:
    """
    Transforms values in range [MIN_SIGNED, MAX_UNSIGNED] to range [0, MAX_UNSIGNED].
    """
    # Normalise to unsigned range by adding the unsigned range and taking the remainder
    modulus = unsigned_upper_bound(bitwidth)
    return (signless + modulus) % modulus


def to_signed(signless: int, bitwidth: int) -> int:
    """
    Transforms values in range [MIN_SIGNED, MAX_UNSIGNED] to range [MIN_SIGNED, MAX_SIGNED].
    """
    # Normalise to unsigned range by adding the unsigned range and taking the remainder
    modulus = unsigned_upper_bound(bitwidth)
    half_modulus = modulus >> 1
    return (signless + half_modulus) % modulus - half_modulus
