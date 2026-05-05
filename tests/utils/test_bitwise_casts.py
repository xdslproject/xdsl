import struct

import pytest

from xdsl.utils.bitwise_casts import (
    convert_bf16_to_f32,
    convert_f32_to_bf16,
    convert_f32_to_u32,
    convert_u32_to_f32,
    is_power_of_two,
)


# http://bartaz.github.io/ieee754-visualization/
@pytest.mark.parametrize(
    "i, f",
    [
        (0b00000000000000000000000000000000, 0.0),
        (0b10000000000000000000000000000000, -0.0),
        (0b00111111100000000000000000000000, 1.0),
        (0b01000000000000000000000000000000, 2.0),
    ],
)
def test_float_bitwise_casts(i: int, f: float):
    assert convert_f32_to_u32(f) == i
    assert struct.pack(">f", convert_u32_to_f32(i)) == struct.pack(">f", f)


@pytest.mark.parametrize(
    "i, f",
    [
        (0x0000, 0.0),
        (0x8000, -0.0),
        (0x3F80, 1.0),
        (0xBF80, -1.0),
        (0x4000, 2.0),
        (0x7F80, float("inf")),
        (0xFF80, float("-inf")),
    ],
)
def test_bf16_bitwise_casts(i: int, f: float):
    assert convert_f32_to_bf16(f) == i
    # Round-trip via struct so -0.0 and 0.0 compare distinctly.
    assert struct.pack(">f", convert_bf16_to_f32(i)) == struct.pack(">f", f)


def test_bf16_nan_stays_nan():
    nan_bits = convert_f32_to_bf16(float("nan"))
    assert (nan_bits & 0x7F80) == 0x7F80  # exponent all-ones
    assert (nan_bits & 0x007F) != 0  # mantissa non-zero
    assert nan_bits & 0x0040  # quiet bit set


def test_bf16_round_to_nearest_even():
    # Halfway between two bf16 values; ties go to even (mantissa LSB 0).
    halfway = 1.0 + 2.0**-8
    assert convert_f32_to_bf16(halfway) == 0x3F80
    just_above = 1.0 + 2.0**-8 + 2.0**-20
    assert convert_f32_to_bf16(just_above) == 0x3F81


def test_bf16_lossy_roundtrip():
    rt = convert_bf16_to_f32(convert_f32_to_bf16(0.1))
    assert rt != 0.1
    assert abs(rt - 0.1) < 2**-6


@pytest.mark.parametrize(
    "i, p",
    [
        (-2, False),
        (-1, False),
        (0, False),
        (1, True),
        (2, True),
        (3, False),
        (4, True),
        (5, False),
    ],
)
def test_is_power_of_two(i: int, p: bool):
    assert is_power_of_two(i) == p
