from xdsl.affine_ir import AffineExpr, AffineMap


def test_simple_map():
    # x, y
    x = AffineExpr.dimension(0)
    y = AffineExpr.dimension(1)

    # map1: (x, y) -> (x + y, y)
    map1 = AffineMap(2, 0, [x + y, y])
    assert map1.eval([1, 2], []) == [3, 2]
    assert map1.eval([3, 4], []) == [7, 4]
    assert map1.eval([5, 6], []) == [11, 6]

    # map2: (x, y) -> (2x + 3y)
    map2 = AffineMap(2, 0, [2 * x + 3 * y])
    assert map2.eval([1, 2], []) == [8]
    assert map2.eval([3, 4], []) == [18]
    assert map2.eval([5, 6], []) == [28]

    # map3: (x, y) -> (x + y, 2x + 3y)
    map3 = AffineMap(2, 0, [x + y, 2 * x + 3 * y])
    assert map3.eval([1, 2], []) == [3, 8]
    assert map3.eval([3, 4], []) == [7, 18]
    assert map3.eval([5, 6], []) == [11, 28]


def test_quasiaffine_map():
    # x
    x = AffineExpr.dimension(0)
    # N
    N = AffineExpr.symbol(0)

    # map1: (x)[N] -> (x floordiv 2)
    map1 = AffineMap(1, 1, [x.floor_div(2)])
    assert map1.eval([1], [10]) == [0]
    assert map1.eval([2], [10]) == [1]
    assert map1.eval([3], [10]) == [1]
    assert map1.eval([4], [13]) == [2]
    assert map1.eval([5], [10]) == [2]
    assert map1.eval([6], [11]) == [3]

    # map2: (x)[N] -> (-(x ceildiv 2) + N)
    map2 = AffineMap(1, 1, [-(x.ceil_div(2)) + N])
    assert map2.eval([1], [10]) == [9]
    assert map2.eval([2], [10]) == [9]
    assert map2.eval([3], [10]) == [8]
    assert map2.eval([4], [13]) == [11]
    assert map2.eval([5], [10]) == [7]
    assert map2.eval([6], [11]) == [8]

    # map3: (x)[N] -> (x mod 2 - N)
    map3 = AffineMap(1, 1, [(x % 2) - N])
    assert map3.eval([1], [10]) == [-9]
    assert map3.eval([2], [10]) == [-10]
    assert map3.eval([3], [10]) == [-9]
    assert map3.eval([4], [13]) == [-13]
    assert map3.eval([5], [10]) == [-9]
    assert map3.eval([6], [11]) == [-11]
