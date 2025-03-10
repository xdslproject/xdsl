from xdsl.dialects.builtin import i32
from xdsl.interpreters.shaped_array import ShapedArray
from xdsl.interpreters.utils.ptr import TypedPtr


def test_shaped_array_type():
    array = ShapedArray(TypedPtr.new_int32([1, 2, 3]), [3])
    assert array.element_type == i32


def test_shaped_array_offset():
    array = ShapedArray(TypedPtr.new_int32([0, 1, 2, 3, 4, 5]), [2, 3])

    assert array.load((0, 0)) == 0
    assert array.load((0, 1)) == 1
    assert array.load((0, 2)) == 2
    assert array.load((1, 0)) == 3
    assert array.load((1, 1)) == 4
    assert array.load((1, 2)) == 5


def test_shaped_array_printing():
    assert f"{ShapedArray(TypedPtr.new_int32([]), [0])}" == "[]"
    assert f"{ShapedArray(TypedPtr.new_int32([1]), [1])}" == "[1]"
    assert f"{ShapedArray(TypedPtr.new_int32([1]), [1, 1])}" == "[[1]]"
    assert f"{ShapedArray(TypedPtr.new_int32([1]), [1, 1, 1])}" == "[[[1]]]"
    assert (
        f"{ShapedArray(TypedPtr.new_int32([1, 2, 3, 4, 5, 6]), [2, 3])}"
        == "[[1, 2, 3], [4, 5, 6]]"
    )
    assert (
        f"{ShapedArray(TypedPtr.new_int32([1, 2, 3, 4, 5, 6]), [3, 2])}"
        == "[[1, 2], [3, 4], [5, 6]]"
    )


def test_shaped_array_indices():
    array = ShapedArray(TypedPtr.new_int32([0, 1, 2, 3, 4, 5]), [2, 3])
    indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    assert indices == list(array.indices())


def test_transposed():
    source = ShapedArray(TypedPtr.new_int32([0, 1, 2, 3, 4, 5]), [2, 3])
    destination = ShapedArray(TypedPtr.new_int32([0, 3, 1, 4, 2, 5]), [3, 2])

    assert source.transposed(0, 1) == destination
