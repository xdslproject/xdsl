from xdsl.ir import AffineExpr


def test_affine_expr():
    # d0, s0
    a = AffineExpr.dimension(0)
    b = AffineExpr.symbol(0)

    # (5 * d0) + s0 + 1
    c = (a * 5) + b + 1

    # TODO: Assert properly here
    assert c == c
