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
