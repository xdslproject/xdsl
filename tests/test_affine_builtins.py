import re

import pytest

from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineExpr, AffineMap


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


def test_compress_dims():
    # (d0, d1, d2) -> (d1, d2) with [0,1,1] gives (d0, d1) -> (d0, d1)
    # (d0, d1, d2) -> (d2, d2) with [1,0,1] gives (d0, d1) -> (d1, d1)
    t = True
    f = False
    assert AffineMap.from_callable(lambda d0, d1, d2: (d1, d2)).compress_dims(
        [f, t, t]
    ) == AffineMap.from_callable(lambda d0, d1: (d0, d1))
    assert AffineMap.from_callable(lambda d0, d1, d2: (d2, d2)).compress_dims(
        [t, f, t]
    ) == AffineMap.from_callable(lambda d0, d1: (d1, d1))


def test_used_dims():
    assert AffineExpr.dimension(1).used_dims() == {1}
    assert (AffineExpr.dimension(2) + AffineExpr.dimension(3)).used_dims() == {2, 3}
    assert AffineExpr.symbol(4).used_dims() == set()
    assert AffineExpr.constant(5).used_dims() == set()
