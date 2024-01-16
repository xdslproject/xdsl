"""
A collection of helpers for reinterpreting bits.
Used in lowering and interpreting low-level dialects.
"""

import ctypes


def convert_f32_to_u32(value: float) -> int:
    """
    Convert an IEEE 754 float to a raw unsigned integer representation.
    """
    raw_float = ctypes.c_float(value)
    raw_int = ctypes.c_uint32.from_address(ctypes.addressof(raw_float)).value
    return raw_int


def convert_u32_to_f32(value: int) -> float:
    """
    Convert a raw 32-bit unsigned integer to IEEE 754 float representation.
    """
    raw_int = ctypes.c_uint32(value)
    raw_float = ctypes.c_float.from_address(ctypes.addressof(raw_int)).value
    return raw_float


def is_power_of_two(value: int) -> bool:
    """
    Return True if an integer is a power of two.
    Powers of two have only one bit set to one
    """
    return (value > 0) and (value.bit_count() == 1)
