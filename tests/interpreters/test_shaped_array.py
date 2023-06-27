from typing import cast
from xdsl.interpreters.shaped_array import ShapedArray


def test_shaped_array_offset():
    array = ShapedArray([0, 1, 2, 3, 4, 5], [2, 3])

    assert array.load((0, 0)) == 0
    assert array.load((0, 1)) == 1
    assert array.load((0, 2)) == 2
    assert array.load((1, 0)) == 3
    assert array.load((1, 1)) == 4
    assert array.load((1, 2)) == 5


def test_shaped_array_printing():
    assert f"{ShapedArray(cast(list[int], []), [0])}" == "[]"
    assert f"{ShapedArray([1], [1])}" == "[1]"
    assert f"{ShapedArray([1], [1, 1])}" == "[[1]]"
    assert f"{ShapedArray([1], [1, 1, 1])}" == "[[[1]]]"
    assert f"{ShapedArray([1, 2, 3, 4, 5, 6], [2, 3])}" == "[[1, 2, 3], [4, 5, 6]]"
    assert f"{ShapedArray([1, 2, 3, 4, 5, 6], [3, 2])}" == "[[1, 2], [3, 4], [5, 6]]"
