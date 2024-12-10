import pytest

from xdsl.utils.disjoint_set import DisjointSet, IntDisjointSet


def test_disjoint_set_init():
    ds = IntDisjointSet(size=5)
    assert ds.value_count() == 5
    assert ds.set_count() == 5
    # Each element should start in its own set
    for i in range(5):
        assert ds.find(i) == i


def test_disjoint_set_add():
    ds = IntDisjointSet(size=2)
    assert ds.value_count() == 2

    new_val = ds.add()
    assert new_val == 2
    assert ds.value_count() == 3
    assert ds.find(new_val) == new_val


def test_disjoint_set_find_invalid():
    ds = IntDisjointSet(size=3)
    with pytest.raises(KeyError):
        ds.find(3)
    with pytest.raises(KeyError):
        ds.find(-1)


def test_disjoint_set_union():
    ds = IntDisjointSet(size=4)

    # Union 0 and 1
    assert ds.union(0, 1)
    root = ds.find(0)
    assert ds.find(1) == root
    assert ds.set_count() == 3

    # Union 2 and 3
    assert ds.union(2, 3)
    root2 = ds.find(2)
    assert ds.find(3) == root2
    assert ds.set_count() == 2

    # Union already connected elements
    assert not ds.union(0, 1)

    # Union two sets
    assert ds.union(1, 2)
    final_root = ds.find(0)
    assert ds.find(1) == final_root
    assert ds.find(2) == final_root
    assert ds.find(3) == final_root
    assert ds.set_count() == 1


def test_disjoint_set_path_compression():
    ds = IntDisjointSet(size=4)

    # Create a chain: 3->2->1->0
    ds._parent = [0, 0, 1, 2]  # pyright: ignore[reportPrivateUsage]
    ds._count = [4, 3, 2, 1]  # pyright: ignore[reportPrivateUsage]

    # Find should compress the path
    root = ds.find(3)
    # After compression, all nodes should point directly to root
    assert ds._parent[3] == root  # pyright: ignore[reportPrivateUsage]
    assert ds._parent[2] == root  # pyright: ignore[reportPrivateUsage]
    assert ds._parent[1] == root  # pyright: ignore[reportPrivateUsage]
    assert ds._parent[0] == root  # pyright: ignore[reportPrivateUsage]


def test_generic_disjoint_set():
    ds = DisjointSet(["a", "b", "c", "d"])

    # Union a and b
    assert ds.union("a", "b")
    root = ds.find("a")
    assert ds.find("b") == root

    # Union c and d
    assert ds.union("c", "d")
    root2 = ds.find("c")
    assert ds.find("d") == root2

    # Union already connected elements
    assert not ds.union("a", "b")

    # Union two sets
    assert ds.union("b", "c")
    final_root = ds.find("a")
    assert ds.find("b") == final_root
    assert ds.find("c") == final_root
    assert ds.find("d") == final_root


def test_generic_disjoint_set_add():
    ds = DisjointSet(["a", "b"])
    ds.add("c")
    ds.add("d")

    assert ds.union("a", "c")
    root = ds.find("a")
    assert ds.find("c") == root

    assert ds.union("b", "d")
    root2 = ds.find("b")
    assert ds.find("d") == root2


def test_generic_disjoint_set_find_invalid():
    ds = DisjointSet(["a", "b", "c"])
    with pytest.raises(KeyError):
        ds.find("d")
