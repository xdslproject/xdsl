from xdsl.interpreters.utils import ptr


def test_ptr():
    assert ptr.float32.format == "<f"
    assert ptr.float32.size == 4
    assert ptr.float64.format == "<d"
    assert ptr.float64.size == 8
    assert ptr.int32.format == "<i"
    assert ptr.int32.size == 4
    assert ptr.int64.format == "<q"
    assert ptr.int64.size == 8
