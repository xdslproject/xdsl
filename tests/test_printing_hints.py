"""
Test type_repr, which is the function used to print type hints.
"""

from typing import Any

import pytest

from xdsl.utils.hints import type_repr


class A:
    pass


@pytest.mark.parametrize(
    "type, expected",
    [
        (int, "int"),
        (str, "str"),
        (list[int], "list[int]"),
        (list[list[int]], "list[list[int]]"),
        (..., "..."),
        (A, "A"),
        (list[A], "list[A]"),
        (A | int, "A|int"),
        (list[A] | int, "list[A]|int"),
        (None, "None"),
        (int | None, "int|None"),
    ],
)
def test_type_repr(type: Any, expected: str):
    assert type_repr(type) == expected
