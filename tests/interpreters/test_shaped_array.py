from typing import cast
from xdsl.interpreters.shaped_array import ShapedArray


def test_zeros():
    array = ShapedArray([0, 0, 0, 0, 0, 0], [2, 3])
    assert array == ShapedArray(0, [2, 3])


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


def test_shaped_array_indices():
    array = ShapedArray([0, 1, 2, 3, 4, 5], [2, 3])
    indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    assert indices == list(array.indices())


def test_transposed():
    source = ShapedArray([0, 1, 2, 3, 4, 5], [2, 3])
    destination = ShapedArray([0, 3, 1, 4, 2, 5], [3, 2])

    assert source.transposed(0, 1) == destination
