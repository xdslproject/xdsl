import re
from collections.abc import Callable, Sequence

import pytest
from typing_extensions import TypeVar

from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineExpr,
    AffineMap,
    SimpleAffineExprFlattener,
)


def test_simple_map():
    # x, y
    x = AffineExpr.dimension(0)
    y = AffineExpr.dimension(1)

    # map1: (x, y) -> (x + y, y)
    map1 = AffineMap(2, 0, (x + y, y))
    assert map1.eval([1, 2], []) == (3, 2)
    assert map1.eval([3, 4], []) == (7, 4)
    assert map1.eval([5, 6], []) == (11, 6)

    # map2: (x, y) -> (2x + 3y)
    map2 = AffineMap(2, 0, (2 * x + 3 * y,))
    assert map2.eval([1, 2], []) == (8,)
    assert map2.eval([3, 4], []) == (18,)
    assert map2.eval([5, 6], []) == (28,)

    # map3: (x, y) -> (x + y, 2x + 3y)
    map3 = AffineMap(2, 0, (x + y, 2 * x + 3 * y))
    assert map3.eval([1, 2], []) == (3, 8)
    assert map3.eval([3, 4], []) == (7, 18)
    assert map3.eval([5, 6], []) == (11, 28)


def test_quasiaffine_map():
    # x
    x = AffineExpr.dimension(0)
    # N
    N = AffineExpr.symbol(0)

    # map1: (x)[N] -> (x floordiv 2)
    map1 = AffineMap(1, 1, (x // 2,))
    assert map1.eval([1], [10]) == (0,)
    assert map1.eval([2], [10]) == (1,)
    assert map1.eval([3], [10]) == (1,)
    assert map1.eval([4], [13]) == (2,)
    assert map1.eval([5], [10]) == (2,)
    assert map1.eval([6], [11]) == (3,)

    # map2: (x)[N] -> (-(x ceildiv 2) + N)
    map2 = AffineMap(1, 1, (-(x.ceil_div(2)) + N,))
    assert map2.eval([1], [10]) == (9,)
    assert map2.eval([2], [10]) == (9,)
    assert map2.eval([3], [10]) == (8,)
    assert map2.eval([4], [13]) == (11,)
    assert map2.eval([5], [10]) == (7,)
    assert map2.eval([6], [11]) == (8,)

    # map3: (x)[N] -> (x mod 2 - N)
    map3 = AffineMap(1, 1, ((x % 2) - N,))
    assert map3.eval([1], [10]) == (-9,)
    assert map3.eval([2], [10]) == (-10,)
    assert map3.eval([3], [10]) == (-9,)
    assert map3.eval([4], [13]) == (-13,)
    assert map3.eval([5], [10]) == (-9,)
    assert map3.eval([6], [11]) == (-11,)


def test_composition_simple():
    # map1 = (x, y) -> (x - y)
    map1 = AffineMap(2, 0, (AffineExpr.dimension(0) - AffineExpr.dimension(1),))
    # map2 = (x, y) -> (y, x)
    map2 = AffineMap(
        2,
        0,
        (AffineExpr.dimension(1), AffineExpr.dimension(0)),
    )
    # Compose
    # map3 = (x, y) -> (y - x)
    map3 = map1.compose(map2)

    assert map3.eval([1, 2], []) == (1,)
    assert map3.eval([3, 4], []) == (1,)
    assert map3.eval([5, 6], []) == (1,)
    assert map3.eval([20, 10], []) == (-10,)


def test_composition():
    # map1: (x, y) -> (x floordiv 2, 2 * x + 3 * y)
    map1 = AffineMap(
        2,
        0,
        (
            AffineExpr.dimension(0) // 2,
            2 * AffineExpr.dimension(0) + 3 * AffineExpr.dimension(1),
        ),
    )
    # map2: (x, y) -> (-x, -y)
    map2 = AffineMap(2, 0, (-AffineExpr.dimension(0), -AffineExpr.dimension(1)))
    # Compose
    # map3: (x, y) -> (-x floordiv 2, -2 * x - 3 * y)
    map3 = map1.compose(map2)

    assert map3.eval([1, 2], []) == (-1, -8)
    assert map3.eval([3, 4], []) == (-2, -18)
    assert map3.eval([5, 6], []) == (-3, -28)


def test_compose_expr():
    d = [AffineExpr.dimension(i) for i in range(3)]
    s = [AffineExpr.symbol(i) for i in range(2)]

    expr = d[0] + d[2]
    map = AffineMap.from_callable(
        lambda d0, d1, d2, s0, s1: (d0 + s1, d1 + s0, d0 + d1 + d2),
        dim_symbol_split=(3, 2),
    )
    expected = (d[0] + s[1]) + (d[0] + d[1] + d[2])
    assert expr.compose(map) == expected


def test_compose_expr_recursive_simplification():
    d = [AffineExpr.dimension(i) for i in range(4)]

    # Tests simplifications that require recursion
    add1 = (d[0] + d[1]) + (d[2] + d[3])
    assert add1.compose(
        AffineMap.from_callable(lambda d0, d1: (1, 2, 3, 4))
    ) == AffineExpr.constant(10)

    add2 = d[0] + d[1] + d[2] + d[3]
    assert add2.compose(
        AffineMap.from_callable(lambda d0, d1: (1, 2, 3, 4))
    ) == AffineExpr.constant(10)


def test_compose_map():
    map1 = AffineMap.from_callable(
        lambda d0, d1, s0, s1: (d0 + 1 + s1, d1 - 1 - s0), dim_symbol_split=(2, 2)
    )
    map2 = AffineMap.from_callable(
        lambda d0, s0: (d0 + s0, d0 - s0), dim_symbol_split=(1, 1)
    )
    map3 = AffineMap.from_callable(
        lambda d0, s0, s1, s2: ((d0 + s2) + 1 + s1, (d0 - s2) - 1 - s0),
        dim_symbol_split=(1, 3),
    )

    assert map1.compose(map2) == map3


def test_helpers():
    m = AffineMap.constant_map(0)
    assert m == AffineMap(0, 0, (AffineExpr.constant(0),))
    m = AffineMap.point_map(0, 1)
    assert m == AffineMap(0, 0, (AffineExpr.constant(0), AffineExpr.constant(1)))
    m = AffineMap.identity(2)
    assert m == AffineMap(2, 0, (AffineExpr.dimension(0), AffineExpr.dimension(1)))
    m = AffineMap.identity(0, 2)
    assert m == AffineMap(0, 2, (AffineExpr.symbol(0), AffineExpr.symbol(1)))
    m = AffineMap.identity(2, 2)
    assert m == AffineMap(
        2,
        2,
        (
            AffineExpr.dimension(0),
            AffineExpr.dimension(1),
            AffineExpr.symbol(0),
            AffineExpr.symbol(1),
        ),
    )
    m = AffineMap.transpose_map()
    assert m == AffineMap(2, 0, (AffineExpr.dimension(1), AffineExpr.dimension(0)))
    m = AffineMap.empty()
    assert m == AffineMap(0, 0, ())


def test_from_callable():
    assert AffineMap.from_callable(lambda: (1,)) == AffineMap.constant_map(1)
    assert AffineMap.from_callable(lambda: (0, 1)) == AffineMap.point_map(0, 1)
    assert AffineMap.from_callable(lambda i, j: (i, j)) == AffineMap.identity(2)
    assert AffineMap.from_callable(lambda i, j: (j, i)) == AffineMap.transpose_map()
    assert AffineMap.from_callable(lambda: ()) == AffineMap.empty()

    assert AffineMap.from_callable(
        lambda i, j, p, q: (p + i, q + j), dim_symbol_split=(2, 2)
    ) == AffineMap(
        2,
        2,
        (
            AffineBinaryOpExpr(
                AffineBinaryOpKind.Add, AffineExpr.symbol(0), AffineExpr.dimension(0)
            ),
            AffineBinaryOpExpr(
                AffineBinaryOpKind.Add, AffineExpr.symbol(1), AffineExpr.dimension(1)
            ),
        ),
    )


def test_from_callable_fail():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Argument count mismatch in AffineMap.from_callable: 1 != 1 + 1"
        ),
    ):
        AffineMap.from_callable(lambda i: (i,), dim_symbol_split=(1, 1))


def test_inverse_permutation():
    assert AffineMap.empty().inverse_permutation() == AffineMap.empty()
    assert AffineMap.from_callable(
        lambda d0, d1, d2: (d1, d1, d0, d2, d1, d2, d1, d0)
    ).inverse_permutation() == AffineMap(
        8, 0, tuple(AffineExpr.dimension(d) for d in (2, 0, 3))
    )


@pytest.mark.parametrize(
    "input_map, expected_map",
    [
        (AffineMap.empty(), AffineMap.empty()),
        (
            AffineMap.from_callable(lambda d0, d1, d2: (d2, d1, d0)),
            AffineMap.from_callable(lambda d0, d1, d2: (d2, d1, d0)),
        ),
        (
            AffineMap.from_callable(lambda d0, d1, d2: (d1, d0)),
            AffineMap.from_callable(lambda d0, d1: (d1, d0, 0)),
        ),
        (
            AffineMap.from_callable(lambda d0, d1, d2: (d1, 0, d0)),
            AffineMap.from_callable(lambda d0, d1, d2: (d2, d0, 0)),
        ),
    ],
)
def test_inverse_and_broadcast_projected_permutation(
    input_map: AffineMap, expected_map: AffineMap
):
    inverse = input_map.inverse_and_broadcast_projected_permutation()
    assert inverse == expected_map, f"{input_map} {expected_map}"
    if inverse.is_projected_permutation():
        assert inverse.inverse_and_broadcast_projected_permutation() == input_map


t = True
f = False


@pytest.mark.parametrize(
    "input_map, drop_dims_mask, expected_map",
    [
        (
            AffineMap.from_callable(lambda d0, d1, d2: (d1, d2)),
            [t, f, f],
            AffineMap.from_callable(lambda d0, d1: (d0, d1)),
        ),
        (
            AffineMap.from_callable(lambda d0, d1, d2: (d2, d2)),
            [f, t, f],
            AffineMap.from_callable(lambda d0, d1: (d1, d1)),
        ),
    ],
)
def test_drop_dims(
    input_map: AffineMap, drop_dims_mask: Sequence[bool], expected_map: AffineMap
):
    assert input_map.drop_dims(drop_dims_mask) == expected_map


@pytest.mark.parametrize(
    "input_map, drop_results_mask, expected_map",
    [
        (
            AffineMap.from_callable(lambda d0, d1, d2: (d1, d2)),
            [t, f],
            AffineMap.from_callable(lambda d0, d1, d2: (d2,)),
        ),
        (
            AffineMap.from_callable(lambda d0, d1, d2: (d1, d2)),
            [f, t],
            AffineMap.from_callable(lambda d0, d1, d2: (d1,)),
        ),
        (
            AffineMap.from_callable(
                lambda d0, d1, s0: (d0 + s0, d1), dim_symbol_split=(2, 1)
            ),
            [t, f],
            AffineMap.from_callable(lambda d0, d1, s0: (d1,), dim_symbol_split=(2, 1)),
        ),
        (
            AffineMap.from_callable(
                lambda d0, d1, s0: (d0 + s0, d1), dim_symbol_split=(2, 1)
            ),
            [f, t],
            AffineMap.from_callable(
                lambda d0, d1, s0: (d0 + s0,), dim_symbol_split=(2, 1)
            ),
        ),
    ],
)
def test_drop_results(
    input_map: AffineMap, drop_results_mask: Sequence[bool], expected_map: AffineMap
):
    assert input_map.drop_results(drop_results_mask) == expected_map


def test_affine_expr_affine_expr_binary_simplification():
    one = AffineExpr.constant(1)
    two = AffineExpr.constant(2)
    three = AffineExpr.constant(3)
    five = AffineExpr.constant(5)
    six = AffineExpr.constant(6)

    # Should return AffineConstExpr when both lhs and rhs are AffineConstantExpr
    assert AffineExpr.binary(AffineBinaryOpKind.Add, one, one) == two
    assert AffineExpr.binary(AffineBinaryOpKind.Mul, two, three) == six
    assert AffineExpr.binary(AffineBinaryOpKind.Mod, five, two) == one
    assert AffineExpr.binary(AffineBinaryOpKind.FloorDiv, five, two) == two
    assert AffineExpr.binary(AffineBinaryOpKind.CeilDiv, five, two) == three


def test_affine_expr_used_dims():
    assert AffineExpr.dimension(1).used_dims() == {1}
    assert (AffineExpr.dimension(2) + AffineExpr.dimension(3)).used_dims() == {2, 3}
    assert AffineExpr.symbol(4).used_dims() == set()
    assert AffineExpr.constant(5).used_dims() == set()


@pytest.mark.parametrize(
    "affine_map, expected_used, expected_unused",
    [
        (
            AffineMap.from_callable(lambda i, j: (i, j)),
            {0, 1},
            set[int](),
        ),
        (
            AffineMap.from_callable(lambda i, j, _: (i + j,)),
            {0, 1},
            {2},
        ),
        (
            AffineMap.from_callable(lambda i, _, k: (i, k)),
            {0, 2},
            {1},
        ),
    ],
)
def test_affine_map_used_and_unused_dims(
    affine_map: AffineMap, expected_used: set[int], expected_unused: set[int]
):
    assert affine_map.used_dims() == expected_used
    assert affine_map.unused_dims() == expected_unused


@pytest.mark.parametrize(
    "affine_map, expected_used",
    [
        (
            AffineMap.from_callable(lambda i, j: (i, j)),
            (True, True),
        ),
        (
            AffineMap.from_callable(lambda i, j, _: (i + j,)),
            (True, True, False),
        ),
        (
            AffineMap.from_callable(lambda i, _, k: (i, k)),
            (True, False, True),
        ),
        (
            AffineMap.from_callable(lambda _, __: ()),
            (False, False),
        ),
    ],
)
def test_affine_map_used_dims_bit_vector(
    affine_map: "AffineMap", expected_used: tuple[bool, ...]
) -> None:
    expected_unused = tuple(not d for d in expected_used)
    assert affine_map.used_dims_bit_vector() == expected_used
    assert affine_map.unused_dims_bit_vector() == expected_unused


def test_minor_identity():
    assert AffineMap.empty().is_minor_identity()
    assert AffineMap.identity(3).is_minor_identity()
    assert AffineMap.minor_identity(3, 2).is_minor_identity()
    assert AffineMap.minor_identity(5, 3).is_minor_identity()

    # Test the actual structure of minor identity maps
    minor_id_3_2 = AffineMap.minor_identity(3, 2)
    assert minor_id_3_2 == AffineMap.from_callable(lambda _, d1, d2: (d1, d2))

    minor_id_5_3 = AffineMap.minor_identity(5, 3)
    assert minor_id_5_3 == AffineMap.from_callable(
        lambda _d0, _d1, d2, d3, d4: (d2, d3, d4)
    )
    # Test non-minor identity maps
    non_minor_id = AffineMap(2, 0, (AffineExpr.dimension(0), AffineExpr.dimension(0)))
    assert not non_minor_id.is_minor_identity()

    # Map with symbols is not a minor identity
    map_with_symbols = AffineMap(
        2, 1, (AffineExpr.dimension(0), AffineExpr.dimension(1))
    )
    assert not map_with_symbols.is_minor_identity()

    # Map with non-consecutive dimensions is not a minor identity
    non_consecutive = AffineMap(
        3, 0, (AffineExpr.dimension(0), AffineExpr.dimension(2))
    )
    assert not non_consecutive.is_minor_identity()

    with pytest.raises(
        ValueError,
        match="Dimension mismatch, expected dims 2 to be greater than or equal to "
        "results 3",
    ):
        AffineMap.minor_identity(2, 3)


def test_is_projected_permutation():
    assert AffineMap(0, 0, ()).is_projected_permutation()
    assert not AffineMap(0, 1, ()).is_projected_permutation()

    assert AffineMap.from_callable(lambda d0, d1: (d0, d1)).is_projected_permutation()
    assert not AffineMap.from_callable(
        lambda d0, d1: (d0, d0)
    ).is_projected_permutation()
    assert not AffineMap.from_callable(lambda d0: (d0, d0)).is_projected_permutation()

    assert AffineMap.from_callable(
        lambda d0, d1, d2: (d1, d0)
    ).is_projected_permutation()
    assert not AffineMap.from_callable(
        lambda d0, d1, d2: (d1, 0, d0)
    ).is_projected_permutation()
    assert AffineMap.from_callable(
        lambda d0, d1, d2: (d1, 0, d0)
    ).is_projected_permutation(allow_zero_in_results=True)


_T = TypeVar("_T")


@pytest.mark.parametrize(
    "permutation_map, inputs, expected",
    [
        (
            AffineMap.from_callable(lambda d0, d1, d2: (d1, d0)),
            (10, 20, 30),
            (20, 10),
        ),
        (
            AffineMap(
                8, 0, tuple(AffineExpr.dimension(d) for d in (4, 1, 6, 0, 3, 2, 5, 7))
            ),
            "policemr",
            ("c", "o", "m", "p", "i", "l", "e", "r"),
        ),
    ],
)
def test_apply_permutation_map(
    permutation_map: AffineMap, inputs: Sequence[_T], expected: tuple[_T, ...]
):
    assert permutation_map.apply_permutation(inputs) == expected


d0 = AffineExpr.dimension(0)
d1 = AffineExpr.dimension(1)
s0 = AffineExpr.symbol(0)
c2 = AffineExpr.constant(2)
c3 = AffineExpr.constant(3)


@pytest.mark.parametrize(
    "expr, expected",
    [
        # Pure affine: only addition, multiplication by constant, dimensions, symbols, constants
        (d0 + d1, True),  # d0 + d1 is pure affine
        (c2 * d0 + c3 * d1, True),  # 2 * d0 + 3 * d1 is pure affine
        (d0 * 2 + d1 * 3, True),  # d0 * 2 + d1 * 3 is pure affine
        (d0 + s0, True),  # d0 + s0 is pure affine
        (d0 + 5, True),  # d0 + 5 is pure affine
        (d0 // 2, True),  # d0 // 2 is pure affine (floordiv by constant)
        ((d0 + d1) // 3, True),  # (d0 + d1) // 3 is pure affine
        (d0 % 4, True),  # d0 % 4 is pure affine (mod by constant)
        ((d0 + d1) % 5, True),  # (d0 + d1) % 5 is pure affine
        (d0.ceil_div(7), True),  # d0.ceil_div(7) is pure affine (ceildiv by constant)
        ((d0 + d1).ceil_div(2), True),  # (d0 + d1).ceil_div(2) is pure affine
        # Not pure affine: multiplication by symbol
        (AffineBinaryOpExpr(AffineBinaryOpKind.Mul, d0, s0), False),
        # Not pure affine: multiplication by dimension
        (AffineBinaryOpExpr(AffineBinaryOpKind.Mul, d0, d1), False),
        # Not pure affine: floordiv by symbol
        (AffineBinaryOpExpr(AffineBinaryOpKind.FloorDiv, d0, s0), False),
        # Not pure affine: mod by symbol
        (AffineBinaryOpExpr(AffineBinaryOpKind.Mod, d0, s0), False),
        # Not pure affine: ceildiv by symbol
        (AffineBinaryOpExpr(AffineBinaryOpKind.CeilDiv, d0, s0), False),
    ],
)
def test_is_pure_affine(expr: AffineExpr, expected: bool):
    assert expr.is_pure_affine() is expected


def test_traversal():
    #    6
    #  4 + 5
    # 0+1 2+3
    nodes = [AffineExpr.dimension(i) for i in range(4)]
    nodes.append(nodes[0] + nodes[1])
    nodes.append(nodes[2] + nodes[3])
    nodes.append(nodes[4] + nodes[5])

    root = nodes[-1]

    dfs = tuple(root.dfs())
    assert len(dfs) == 7
    dfs_indices = [6, 4, 0, 1, 5, 2, 3]
    for i, node in zip(dfs_indices, dfs):
        assert nodes[i] is node

    post_order = tuple(root.post_order())
    post_order_indices = [0, 1, 4, 2, 3, 5, 6]
    assert len(post_order) == 7
    for i, node in zip(post_order_indices, post_order):
        assert nodes[i] is node


d0, d1, d2 = (AffineExpr.dimension(i) for i in range(3))
s0, s1, s2 = (AffineExpr.symbol(i) for i in range(3))


def test_from_flat_form():
    # Empty case
    assert AffineExpr.from_flat_form([0], 0, 0, []) == AffineExpr.constant(0)
    # One dimension
    assert AffineExpr.from_flat_form([1, 0], 1, 0, []) == d0
    # One symbol
    assert AffineExpr.from_flat_form([1, 0], 0, 1, []) == s0
    # Factors
    assert (
        AffineExpr.from_flat_form(
            [
                0,  # d0 * 0
                1,  # d1 * 1
                2,  # d2 * 2
                0,  # s0 * 0
                1,  # s1 * 1
                2,  # s2 * 2
                0,  # d0 * 0
                2,  # s0 * 2
                1,  # c1
            ],
            3,
            3,
            [d0, s0],
        )
        == d1 + d2 * 2 + s1 + s2 * 2 + s0 * 2 + 1
    )


@pytest.mark.parametrize(
    "expr,expected",
    [
        # Simple cases: constants, dimensions, symbols
        (AffineExpr.constant(0), AffineExpr.constant(0)),
        (AffineExpr.constant(42), AffineExpr.constant(42)),
        (d0, d0),
        (d1, d1),
        (s0, s0),
        (s2, s2),
        # Already-simplified sums
        (d0 + 3, d0 + 3),
        (2 * d1 + s0, 2 * d1 + s0),
        # Expressions with redundant zeros
        (d0 * 0, AffineExpr.constant(0)),
        (0 * s1, AffineExpr.constant(0)),
        (AffineExpr.constant(0) + d1, d1),
        (d0 + 0, d0),
        (0 + d2, d2),
        # Combining constants
        (
            AffineBinaryOpExpr(
                AffineBinaryOpKind.Add, AffineExpr.constant(2), AffineExpr.constant(3)
            ),
            AffineExpr.constant(5),
        ),
        (
            AffineBinaryOpExpr(
                AffineBinaryOpKind.Mul,
                AffineBinaryOpExpr(
                    AffineBinaryOpKind.Add,
                    AffineExpr.constant(1),
                    AffineExpr.constant(2),
                ),
                AffineExpr.constant(3),
            ),
            AffineExpr.constant(9),
        ),
        ((d0 + 2) + 3, d0 + 5),
        # Multiplying by 1 or 0
        (d1 * 1, d1),
        (1 * s0, s0),
        ((d1 + 2) * 1, d1 + 2),
        ((d2 + 7) * 0, AffineExpr.constant(0)),
        # Grouping like terms (if supported by simplify)
        (d0 + d0, 2 * d0),
        (2 * d0 + 3 * d0, 5 * d0),
        (d0 + s1 + d0, 2 * d0 + s1),
        # More complex: terms cancel
        (d0 - d0, AffineExpr.constant(0)),
        (2 * d1 - d1, d1),
        ((d0 + 2) - 2, d0),
        # Deeply nested expressions
        ((d0 + (AffineExpr.constant(1) + 2)), d0 + 3),
        (((d0 + 2) + (d0 + 3)), 2 * d0 + 5),
        # Exact multiples of floordiv are cancelled out
        ((2 * d0 + 4 * s1) // 2, d0 + 2 * s1),
        # Exact multiples of ceildiv are cancelled out
        ((2 * d0 + 4 * s1).ceil_div(2), d0 + 2 * s1),
        # Greatest common denominators of floordiv are cancelled out
        ((2 * d0 + 4 * s1) // 6, (d0 + 2 * s1) // 3),
        # Greatest common denominators of ceildiv are cancelled out
        ((2 * d0 + 4 * s1).ceil_div(6), (d0 + 2 * s1).ceil_div(3)),
        # Division is aggregated
        (d0 // 2 + d0 // 2, d0 // 2 * 2),
        # Factors and modulo cancel
        ((2 * d0 + 4 * s1) % 2, AffineExpr.constant(0)),
        # Modulo is converted into the manual computation (x % k == x - (x // k * k))
        ((2 * d0 + 4 * s1) % 3, (2 * d0 + 4 * s1) + ((2 * d0 + 4 * s1) // 3 * -3)),
        # This sometimes cancels out
        (d0 % 3 + d0 // 3 * 3, d0),
        # GCD cancel in the fraction
        ((2 * d0 + 4 * s1) % 6, (2 * d0 + 4 * s1) + ((d0 + 2 * s1) // 3 * -6)),
        # Modulos with same expressions are aggregated
        (d0 % 2 - d0 % 2, AffineExpr.constant(0)),
    ],
)
def test_affine_expr_simplify(expr: AffineExpr, expected: AffineExpr):
    assert str(expr.simplify(3, 3)) == str(expected)
    assert expr.simplify(3, 3) == expected
    # Eval on some numbers to catch errors in simplification test
    assert expr.eval((1, 2, 3), (5, 7, 11)) == expected.eval((1, 2, 3), (5, 7, 11))
    assert expr.eval((11, 7, 5), (3, 2, 1)) == expected.eval((11, 7, 5), (3, 2, 1))


@pytest.mark.parametrize(
    "expr",
    [
        (lambda: AffineBinaryOpExpr(AffineBinaryOpKind.Mul, d0, s0)),
        (lambda: AffineBinaryOpExpr(AffineBinaryOpKind.FloorDiv, d0, s0)),
        (lambda: AffineBinaryOpExpr(AffineBinaryOpKind.Mod, d0, s0)),
    ],
)
def test_affine_expr_simplify_semi_affine_raises(expr: Callable[[], AffineExpr]):
    with pytest.raises(
        NotImplementedError,
        match="Simplification of semi-affine expressions is not implemented yet.",
    ):
        expr().simplify(num_dims=3, num_symbols=3)
    with pytest.raises(
        NotImplementedError,
        match="Semi-affine map flattening not implemented",
    ):
        SimpleAffineExprFlattener(3, 3).simplify(expr())


def test_simplify_division_negative_divisor():
    with pytest.raises(ValueError, match="RHS of division must be positive, got -2"):
        (AffineExpr.dimension(0) // -2).simplify(3, 3)
