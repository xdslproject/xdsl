from struct import pack, unpack


def from_i8(value: int) -> bytes:
    return pack(">b", value)


def from_i16(value: int) -> bytes:
    return pack(">h", value)


def from_i32(value: int) -> bytes:
    return pack(">l", value)


def from_i64(value: int) -> bytes:
    return pack(">q", value)


def from_u8(value: int) -> bytes:
    return pack(">B", value)


def from_u16(value: int) -> bytes:
    return pack(">H", value)


def from_u32(value: int) -> bytes:
    return pack(">L", value)


def from_u64(value: int) -> bytes:
    return pack(">Q", value)


def from_f16(value: float) -> bytes:
    return pack(">e", value)


def from_f32(value: float) -> bytes:
    return pack(">f", value)


def from_f64(value: float) -> bytes:
    return pack(">d", value)


def to_i8(value: bytes) -> int:
    return unpack(">b", value)[0]


def to_i16(value: bytes) -> int:
    return unpack(">h", value)[0]


def to_i32(value: bytes) -> int:
    return unpack(">l", value)[0]


def to_i64(value: bytes) -> int:
    return unpack(">q", value)[0]


def to_u8(value: bytes) -> int:
    return unpack(">B", value)[0]


def to_u16(value: bytes) -> int:
    return unpack(">H", value)[0]


def to_u32(value: bytes) -> int:
    return unpack(">L", value)[0]


def to_u64(value: bytes) -> int:
    return unpack(">Q", value)[0]


def to_f16(value: bytes) -> float:
    return unpack(">e", value)[0]


def to_f32(value: bytes) -> float:
    return unpack(">f", value)[0]


def to_f64(value: bytes) -> float:
    return unpack(">d", value)[0]


def as_str(value: bytes) -> str:
    return f"{int(value.hex(), 16):0{8 * len(value)}b}"
