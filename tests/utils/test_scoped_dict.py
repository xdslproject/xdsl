import pytest

from xdsl.utils.scoped_dict import ScopedDict


def test_simple():
    table = ScopedDict[int, int]()

    table[1] = 2

    assert table[1] == 2

    table[2] = 3

    assert table[2] == 3

    with pytest.raises(ValueError, match="Cannot overwrite value 3 for key 2"):
        table[2] = 4

    with pytest.raises(KeyError):
        table[3]

    inner = ScopedDict(table, name="inner")

    inner[2] = 4

    assert inner[2] == 4
    assert table[2] == 3

    inner[3] = 5

    assert 3 not in table
    assert 3 in inner
    assert 4 not in inner
