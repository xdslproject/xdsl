"""
Typed wrapper around the raw 2 bytes of a bfloat16 value.

bf16 has no native Python representation. ``BF16Float`` owns the format
codec (round-to-nearest-even, NaN-quiet preservation; matches LLVM
APFloat) and lets the rest of xDSL pass bf16 values around as a typed
object instead of a bare ``int`` bit pattern.
"""

from __future__ import annotations

import struct
from typing import ClassVar

from typing_extensions import Self


class BF16Float:
    """
    Typed wrapper around the raw little-endian bytes of a bfloat16 value.

    Construct from raw bytes (``BF16Float(b"\\x80\\x3f")``) or from a
    Python numeric (``BF16Float.from_value(1.0)``). Decode to a Python
    float with ``float(self)``. Equality and hashing are bytes-based:
    ``+0.0`` vs ``-0.0`` and distinct NaN payloads compare unequal.
    """

    size_bytes: ClassVar[int] = 2

    __slots__ = ("raw",)

    raw: bytes
    """
    The two raw bytes of the bf16 bit pattern, stored little-endian (low
    byte first). For value 1.0 (bit pattern ``0x3F80``), ``raw`` is
    ``b"\\x80\\x3f"``. ``hex()`` reverses to natural reading order.
    """

    def __init__(self, raw: bytes) -> None:
        if len(raw) != self.size_bytes:
            raise ValueError(
                f"BF16Float expects {self.size_bytes} bytes, got {len(raw)}"
            )
        self.raw = raw

    @classmethod
    def from_value(cls, value: float | int) -> Self:
        """
        Build from a Python numeric value.

        ``int`` is interpreted as the raw bit pattern (little-endian),
        ``float`` is encoded through the bf16 codec.
        """
        if isinstance(value, int):
            return cls(value.to_bytes(cls.size_bytes, "little"))
        return cls(_encode_bf16(value))

    def hex(self) -> str:
        """``0x``-prefixed lowercase hex, in natural (big-endian) reading order."""
        return f"0x{self.raw[::-1].hex()}"

    def __float__(self) -> float:
        # bf16 is the high 16 bits of f32 with the low 16 truncated; the
        # inverse is to zero-extend with two low bytes in little-endian.
        return struct.unpack("<f", b"\x00\x00" + self.raw)[0]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BF16Float) and self.raw == other.raw

    def __hash__(self) -> int:
        return hash((BF16Float, self.raw))

    def __repr__(self) -> str:
        return f"BF16Float({self.hex()})"

    def __str__(self) -> str:
        return str(float(self))


def _encode_bf16(value: float) -> bytes:
    """
    Encode a Python float (interpreted as IEEE 754 binary32 after Python's
    f64 -> f32 narrowing) as bf16 bytes. Round-to-nearest-even,
    quiet-NaN preservation; matches LLVM APFloat semantics.
    """
    f32_bits = struct.unpack("<I", struct.pack("<f", value))[0]
    # NaN must remain a NaN after truncation; force the quiet bit on so a
    # signaling NaN with mantissa entirely in the truncated bits doesn't
    # become inf.
    if (f32_bits & 0x7FFFFFFF) > 0x7F800000:
        bits = ((f32_bits >> 16) | 0x0040) & 0xFFFF
    else:
        rounding_bias = 0x7FFF + ((f32_bits >> 16) & 1)
        bits = ((f32_bits + rounding_bias) >> 16) & 0xFFFF
    return bits.to_bytes(BF16Float.size_bytes, "little")
