"""
A collection of helpers for reinterpreting bits.
Used in lowering and interpreting low-level dialects.
"""

import ctypes
import struct


def convert_f16_to_u16(value: float) -> int:
    """
    Convert an IEEE 754 float to a raw unsigned integer representation.
    """
    # using struct library as ctypes does not support half-precision floats
    return struct.unpack("<H", struct.pack("<e", value))[0]


def convert_u16_to_f16(value: int) -> int:
    """
    Convert an IEEE 754 float to a raw unsigned integer representation.
    """
    # using struct library as ctypes does not support half-precision floats
    return struct.unpack("<e", struct.pack("<H", value))[0]


def convert_f32_to_u32(value: float) -> int:
    """
    Convert an IEEE 754 float to a raw unsigned integer representation.
    """
    raw_float = ctypes.c_float(value)
    raw_int = ctypes.c_uint32.from_buffer(raw_float).value
    return raw_int


def convert_u32_to_f32(value: int) -> float:
    """
    Convert a raw 32-bit unsigned integer to IEEE 754 float representation.
    """
    raw_int = ctypes.c_uint32(value)
    raw_float = ctypes.c_float.from_buffer(raw_int).value
    return raw_float


def convert_f64_to_u64(value: float) -> int:
    """
    Convert an IEEE 754 float to a raw unsigned integer representation.
    """
    raw_float = ctypes.c_double(value)
    raw_int = ctypes.c_uint64.from_buffer(raw_float).value
    return raw_int


def convert_u64_to_f64(value: int) -> float:
    """
    Convert a raw 32-bit unsigned integer to IEEE 754 float representation.
    """
    raw_int = ctypes.c_uint64(value)
    raw_float = ctypes.c_double.from_buffer(raw_int).value
    return raw_float


def convert_f32_to_bf16(value: float) -> int:
    """
    Convert a Python float (interpreted as IEEE 754 binary32 after Python's
    f64 -> f32 narrowing) to its bfloat16 bit pattern.

    Uses round-to-nearest-even, matching LLVM APFloat semantics. NaNs are
    preserved as quiet NaNs.
    """
    f32_bits = struct.unpack("<I", struct.pack("<f", value))[0]
    # NaN must remain a NaN after truncation; force the quiet bit on so a
    # signaling NaN with mantissa entirely in the truncated bits doesn't
    # become inf.
    if (f32_bits & 0x7FFFFFFF) > 0x7F800000:
        return ((f32_bits >> 16) | 0x0040) & 0xFFFF
    rounding_bias = 0x7FFF + ((f32_bits >> 16) & 1)
    return ((f32_bits + rounding_bias) >> 16) & 0xFFFF


def convert_bf16_to_f32(value: int) -> float:
    """
    Convert a bfloat16 bit pattern to a Python float by zero-extending the
    low 16 bits and reinterpreting as IEEE 754 binary32.
    """
    f32_bits = (value & 0xFFFF) << 16
    return struct.unpack("<f", struct.pack("<I", f32_bits))[0]


def is_power_of_two(value: int) -> bool:
    """
    Return True if an integer is a power of two.
    Powers of two have only one bit set to one
    """
    return (value > 0) and (value.bit_count() == 1)
