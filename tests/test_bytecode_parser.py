import pytest

from xdsl.bytecode_parser import BytecodeParser


@pytest.mark.parametrize(
    "input, result",
    (
        (0b00000001, 0),
        (0b00000011, 0),
        (0b00000111, -1),
        (0b00000101, 1),
        (0b11111101, 0b111111),
        (0b11111111, -0b111111),
    ),
)
def test_single_bit_varint(input: int, result: int):
    assert BytecodeParser(input.to_bytes(1, "little")).parse_varint() == result


@pytest.mark.parametrize(
    "input, result",
    (
        ((0b00000010, 0), 0),
        ((0b00000010, 1), 0),
        ((0b00000010, 0b11), -1),
        ((0b11111110, 0b11111110), 8191),
        (
            (
                0b00000000,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111110,
            ),
            2**63 - 1,
        ),
        (
            (
                0b00000000,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
                0b11111111,
            ),
            -(2**63 - 1),
        ),
    ),
)
def test_multi_bit_varint(input: tuple[int], result: int):
    assert BytecodeParser(bytes(input)).parse_varint() == result
