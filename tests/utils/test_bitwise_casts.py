import struct

import pytest

from xdsl.utils.bitwise_casts import convert_f32_to_u32, convert_u32_to_f32


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
