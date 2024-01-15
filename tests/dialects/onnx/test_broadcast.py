import pytest

from xdsl.dialects.onnx import (
    multidirectional_broadcast_shape,
    unidirectional_broadcast_shape,
)

m_test_cases: list[tuple[tuple[list[int], list[int]], list[int]]] = [
    (([2, 3, 4, 5], []), [2, 3, 4, 5]),
    (([2, 3, 4, 5], [5]), [2, 3, 4, 5]),
    (([4, 5], [2, 3, 4, 5]), [2, 3, 4, 5]),
    (([1, 4, 5], [2, 3, 1, 5]), [2, 3, 4, 5]),
    (([3, 4, 5], [2, 1, 1, 1]), [2, 3, 4, 5]),
]

u_test_cases: list[tuple[tuple[list[int], list[int]], list[int]]] = [
    (([2, 3, 4, 5], []), [2, 3, 4, 5]),
    (([2, 3, 4, 5], [5]), [2, 3, 4, 5]),
    (([2, 3, 4, 5], [2, 1, 1, 5]), [2, 3, 4, 5]),
    (([2, 3, 4, 5], [1, 3, 1, 5]), [2, 3, 4, 5]),
]

m_fail_test_cases: list[tuple[tuple[list[int], list[int]], list[int | None]]] = [
    (([2, 3, 4, 5], [6, 1]), [None]),
    (([1, 3, 4, 1], [5]), [None]),
    (([4, 5, 6], [2, 3, 4, 5]), [None]),
    (([1, 4, 5], [2, 3, 1, 5]), [None]),
    (([3, 4, 5], [2, 1, 1, 1]), [None]),
]

u_fail_test_cases: list[tuple[tuple[list[int], list[int]], list[int | None]]] = [
    (([], [2, 3, 4, 5]), [None]),
    (([5], [2, 3, 4, 5]), [None]),
    (([2, 1, 1, 5], [2, 3, 4, 5]), [None]),
    (([2, 3, 5], [1, 3, 1, 5]), [None]),
]


# Multidirectional Broadcasting Tests
@pytest.mark.parametrize("input_shapes, expected_result", m_test_cases)
def multi_test_broadcast_shape(
    input_shapes: tuple[list[int], list[int]], expected_result: list[int]
):
    lhs, rhs = input_shapes
    result = multidirectional_broadcast_shape(lhs, rhs)
    assert result == expected_result


# Unidirectional Broadcasting Tests
@pytest.mark.parametrize("input_shapes, expected_result", u_test_cases)
def uni_test_broadcast_shape(
    input_shapes: tuple[list[int], list[int]], expected_result: list[int]
):
    lhs, rhs = input_shapes
    result = unidirectional_broadcast_shape(lhs, rhs)
    assert result == expected_result


# Multidirectional Broadcasting Tests (Failing Cases)
@pytest.mark.parametrize("input_shapes, expected_result", m_fail_test_cases)
def multi_fail_test_broadcast_shape(
    input_shapes: tuple[list[int], list[int]], expected_result: list[int] | None
):
    lhs, rhs = input_shapes
    result = multidirectional_broadcast_shape(lhs, rhs)
    assert result == expected_result


# Unidirectional Broadcasting Tests (Failing Cases)
@pytest.mark.parametrize("input_shapes, expected_result", u_fail_test_cases)
def uni_fail_test_broadcast_shape(
    input_shapes: tuple[list[int], list[int]], expected_result: list[int] | None
):
    lhs, rhs = input_shapes
    result = unidirectional_broadcast_shape(lhs, rhs)
    assert result == expected_result
