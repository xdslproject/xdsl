from dataclasses import dataclass


@dataclass
class BytecodeParser:
    _bytes: bytes
    _pos: int = 0

    def _pop_bytes(self, len: int = 1) -> bytes:
        b = self._bytes[self._pos : self._pos + len]
        self._pos += len
        return b

    def _peek_bytes(self, len: int = 1) -> bytes:
        return self._bytes[self._pos : self._pos + len]

    def parse_varint(self):
        """
        Parse a varint as per the spec:
        https://mlir.llvm.org/docs/BytecodeFormat/#variable-width-integers
        """
        first_byte = self._pop_bytes()[0]

        additional_bytes = 0
        while first_byte & (1 << additional_bytes) == 0 and additional_bytes < 8:
            additional_bytes += 1

        leading_bits = first_byte >> (additional_bytes + 1)

        value_bits = leading_bits
        for byte in self._pop_bytes(additional_bytes):
            value_bits = (value_bits << 8) + byte

        sign = -1 if value_bits & 1 else 1
        value = value_bits >> 1
        return sign * value
