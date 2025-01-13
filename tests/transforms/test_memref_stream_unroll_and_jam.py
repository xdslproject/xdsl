from xdsl.transforms.memref_stream_unroll_and_jam import factors


def test_factors():
    assert factors(-1) == ()
    assert factors(0) == ()
    assert factors(1) == (1,)
    assert factors(2) == (1, 2)
    assert factors(3) == (1, 3)
    assert factors(4) == (1, 2, 4)
    assert factors(6) == (1, 2, 3, 6)
    assert factors(24) == (1, 2, 3, 4, 6, 8, 12, 24)
