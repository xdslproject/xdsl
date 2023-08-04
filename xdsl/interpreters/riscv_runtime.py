"""
This adds support for a `libc` style runtime of functions, similar to riscemus libc
"""
from math import ceil
from typing import Any

from xdsl.interpreters.memref import MemrefValue
from xdsl.interpreters.riscv import Buffer


def _malloc(args: tuple[Any, ...], name: str) -> tuple[Any, ...]:
    assert len(args) == 1
    assert isinstance(args[0], int)
    size = args[0]
    if size % 4 != 0:
        # malloc a bit too much if not word-aligned
        size = ceil(size / 4) * 4

    # set values to 1 to signify uninitialized memory
    return (Buffer([MemrefValue.Initialized] * (size // 4)),)


def _calloc(args: tuple[Any, ...], name: str) -> tuple[Any, ...]:
    assert len(args) == 2
    assert isinstance(args[0], int)
    assert isinstance(args[1], int)
    num = args[0]
    size = args[1]

    num_bytes = num * size

    if num_bytes % 4 != 0:
        # malloc a bit too much if not word-aligned
        num_bytes = ceil(num_bytes / 4) * 4

    return (Buffer([0] * (num_bytes // 4)),)


def _free(args: tuple[Any, ...], name: str) -> tuple[Any, ...]:
    assert len(args) == 1
    assert isinstance(args[0], Buffer)
    buff: Buffer = args[0]

    for i in range(len(buff.data)):
        buff.data[i] = MemrefValue.Freed
    return tuple()


def _putchar(args: tuple[Any, ...], name: str) -> tuple[Any, ...]:
    assert len(args) == 1
    char = args[0]
    if isinstance(char, int):
        print(bytes([char]).decode("ascii"), end="")
    else:
        print(char, end="")
    return tuple()


RUNTIME = {
    "malloc": _malloc,
    "calloc": _calloc,
    "free": _free,
    "putchar": _putchar,
}
