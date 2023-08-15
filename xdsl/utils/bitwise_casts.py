"""
A collection of helpers for reinterpreting bits.
Used in lowering and interpreting low-level dialects.
"""

import ctypes


def convert_float_to_int(value: float) -> int:
    """
    Convert an IEEE 754 float to a raw integer representation.
    """
    raw_float = ctypes.c_float(value)
    raw_int = ctypes.c_int.from_address(ctypes.addressof(raw_float)).value
    return raw_int


def convert_i32_to_float(value: int) -> float:
    """
    Convert a raw 32-bit integer to IEEE 754 float representation.
    """
    raw_int = ctypes.c_int32(value)
    raw_float = ctypes.c_float.from_address(ctypes.addressof(raw_int)).value
    return raw_float
