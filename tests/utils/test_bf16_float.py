import struct

import pytest

from xdsl.utils.bf16_float import BF16Float


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
def test_bf16_round_trip(i: int, f: float):
    encoded = BF16Float.from_value(f)
    assert encoded.raw == BF16Float.from_value(i).raw
    # Round-trip via struct so -0.0 and 0.0 compare distinctly.
    assert struct.pack(">f", float(encoded)) == struct.pack(">f", f)


def test_bf16_nan_stays_nan():
    encoded = BF16Float.from_value(float("nan"))
    bits = int.from_bytes(encoded.raw, "little")
    assert (bits & 0x7F80) == 0x7F80  # exponent all-ones
    assert (bits & 0x007F) != 0  # mantissa non-zero
    assert bits & 0x0040  # quiet bit set


def test_bf16_round_to_nearest_even():
    # Halfway between two bf16 values; ties go to even (mantissa LSB 0).
    halfway = 1.0 + 2.0**-8
    assert BF16Float.from_value(halfway).raw == BF16Float.from_value(0x3F80).raw
    just_above = 1.0 + 2.0**-8 + 2.0**-20
    assert BF16Float.from_value(just_above).raw == BF16Float.from_value(0x3F81).raw


def test_bf16_lossy_roundtrip():
    rt = float(BF16Float.from_value(0.1))
    assert rt != 0.1
    assert abs(rt - 0.1) < 2**-6


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, b"\x00\x00"),
        (-0.0, b"\x00\x80"),
        (1.0, b"\x80\x3f"),
        (-1.0, b"\x80\xbf"),
        (2.0, b"\x00\x40"),
        (float("inf"), b"\x80\x7f"),
        (float("-inf"), b"\x80\xff"),
    ],
)
def test_bf16_encode(value: float, expected: bytes):
    assert BF16Float.encode(value) == expected


def test_bf16_encode_nan_is_quiet():
    raw = BF16Float.encode(float("nan"))
    bits = int.from_bytes(raw, "little")
    assert (bits & 0x7F80) == 0x7F80  # exponent all-ones
    assert (bits & 0x007F) != 0  # mantissa non-zero
    assert bits & 0x0040  # quiet bit set


def test_bf16_encode_round_to_nearest_even():
    halfway = 1.0 + 2.0**-8
    assert BF16Float.encode(halfway) == (0x3F80).to_bytes(2, "little")
    just_above = 1.0 + 2.0**-8 + 2.0**-20
    assert BF16Float.encode(just_above) == (0x3F81).to_bytes(2, "little")


def test_bf16_size_bytes():
    assert BF16Float.size_bytes == 2
    assert len(BF16Float.from_value(1.0).raw) == 2


def test_bf16_rejects_wrong_byte_count():
    with pytest.raises(ValueError, match="expects 2 bytes"):
        BF16Float(b"\x00")
    with pytest.raises(ValueError, match="expects 2 bytes"):
        BF16Float(b"\x00\x00\x00")
