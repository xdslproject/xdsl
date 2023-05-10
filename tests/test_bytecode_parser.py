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
    assert BytecodeParser(input.to_bytes(1, "big")).parse_varint() == result
    assert BytecodeParser(input.to_bytes(1, "big")).parse_varint() == result
    assert BytecodeParser(input.to_bytes(1, "big")).parse_varint() == result
    assert BytecodeParser(input.to_bytes(1, "big")).parse_varint() == result
    assert BytecodeParser(input.to_bytes(1, "big")).parse_varint() == result
    assert BytecodeParser(input.to_bytes(1, "big")).parse_varint() == result
