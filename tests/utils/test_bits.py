from math import isinf, isnan
from random import randrange

import pytest

from xdsl.utils.bits import (
    as_str,
    from_f16,
    from_f32,
    from_f64,
    from_i8,
    from_i16,
    from_i32,
    from_i64,
    from_u8,
    from_u16,
    from_u32,
    from_u64,
    to_f16,
    to_f32,
    to_f64,
    to_i8,
    to_i16,
    to_i32,
    to_i64,
    to_u8,
    to_u16,
    to_u32,
    to_u64,
)


@pytest.mark.parametrize(
    "i",
    (randrange(1 << 8) for _ in range(256)),
)
def test_8bits(i: int):
    b = from_u8(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i8(to_i8(b)) == b
    assert from_u8(to_u8(b)) == b
    b = from_u16(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i16(to_i16(b)) == b
    assert from_u16(to_u16(b)) == b
    assert isnan(v := to_f16(b)) or isinf(v) or from_f16(v) == b
    b = from_u32(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i32(to_i32(b)) == b
    assert from_u32(to_u32(b)) == b
    assert isnan(v := to_f32(b)) or isinf(v) or from_f32(v) == b
    b = from_u64(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i64(to_i64(b)) == b
    assert from_u64(to_u64(b)) == b
    assert isnan(v := to_f64(b)) or isinf(v) or from_f64(v) == b


@pytest.mark.parametrize(
    "i",
    (randrange(1 << 8, 1 << 16) for _ in range(256)),
)
def test_16bits(i: int):
    b = from_u16(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i16(to_i16(b)) == b
    assert from_u16(to_u16(b)) == b
    assert isnan(v := to_f16(b)) or isinf(v) or from_f16(v) == b
    b = from_u32(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i32(to_i32(b)) == b
    assert from_u32(to_u32(b)) == b
    assert isnan(v := to_f32(b)) or isinf(v) or from_f32(v) == b
    b = from_u64(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i64(to_i64(b)) == b
    assert from_u64(to_u64(b)) == b
    assert isnan(v := to_f64(b)) or isinf(v) or from_f64(v) == b


@pytest.mark.parametrize(
    "i",
    (randrange(1 << 16, 1 << 32) for _ in range(256)),
)
def test_32bits(i: int):
    b = from_u32(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i32(to_i32(b)) == b
    assert from_u32(to_u32(b)) == b
    assert isnan(v := to_f32(b)) or isinf(v) or from_f32(v) == b
    b = from_u64(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i64(to_i64(b)) == b
    assert from_u64(to_u64(b)) == b
    assert isnan(v := to_f64(b)) or isinf(v) or from_f64(v) == b


@pytest.mark.parametrize(
    "i",
    (randrange(1 << 32, 1 << 64) for _ in range(256)),
)
def test_64bits(i: int):
    b = from_u64(i)
    assert as_str(b).endswith(bin(int(b.hex(), 16)).removeprefix("0b"))
    assert from_i64(to_i64(b)) == b
    assert from_u64(to_u64(b)) == b
    assert isnan(v := to_f64(b)) or isinf(v) or from_f64(v) == b
