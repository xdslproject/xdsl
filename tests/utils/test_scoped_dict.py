import pytest

from xdsl.utils.scoped_dict import ScopedDict


def test_simple():
    table = ScopedDict[int, int]()

    table[1] = 2

    assert table[1] == 2
    assert table.local_scope == {1: 2}

    table[2] = 3

    assert table[2] == 3
    assert table.local_scope == {1: 2, 2: 3}

    table[2] = 4
    assert table.local_scope == {1: 2, 2: 4}

    with pytest.raises(KeyError):
        table[3]

    inner = ScopedDict(table, name="inner")

    inner[2] = 5

    assert inner[2] == 5
    assert table[2] == 4
    assert table.local_scope == {1: 2, 2: 4}
    assert inner.local_scope == {2: 5}

    inner[3] = 6

    assert 3 not in table
    assert 3 in inner
    assert 4 not in inner
    assert table.local_scope == {1: 2, 2: 4}
    assert inner.local_scope == {2: 5, 3: 6}


def test_get():
    parent = ScopedDict(local_scope={"a": 1, "b": 2})
    child = ScopedDict(parent, local_scope={"a": 3, "c": 4})

    assert child.get("a") == 3
    assert child.get("b") == 2
    assert child.get("c") == 4
    assert child.get("d") is None

    assert child.get("a", 5) == 3
    assert child.get("d", 5) == 5


def test_contains():
    grand_parent = ScopedDict(local_scope={"a": 0, "b": 1})
    parent = ScopedDict(grand_parent, local_scope={"b": 2, "a": 3})
    child = ScopedDict(parent, local_scope={"a": 4, "c": 5})

    assert "a" in grand_parent
    assert "b" in grand_parent
    assert "c" not in grand_parent

    assert "a" in parent
    assert "b" in parent
    assert "c" not in parent

    assert "a" in child
    assert "b" in child
    assert "c" in child


def test_len():
    grand_parent = ScopedDict(local_scope={"a": 0, "b": 1})
    parent = ScopedDict(grand_parent, local_scope={"b": 2, "a": 3})
    child = ScopedDict(parent, local_scope={"a": 4, "c": 5})

    assert len(grand_parent) == 2
    assert len(parent) == 2
    assert len(child) == 3


def test_iter():
    grand_parent = ScopedDict(local_scope={"a": 0, "b": 1})
    parent = ScopedDict(grand_parent, local_scope={"b": 2, "a": 3})
    child = ScopedDict(parent, local_scope={"a": 4, "c": 5})

    assert (*grand_parent,) == ("a", "b")
    assert (*parent,) == ("b", "a")
    assert (*child,) == ("a", "c", "b")
