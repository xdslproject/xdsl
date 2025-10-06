import pytest

from xdsl.utils.disjoint_set import DisjointSet, IntDisjointSet


def test_disjoint_set_init():
    ds = IntDisjointSet(size=5)
    assert ds.value_count() == 5
    # Each element should start in its own set
    for i in range(5):
        assert ds[i] == i


def test_disjoint_set_add():
    ds = IntDisjointSet(size=2)
    assert ds.value_count() == 2

    new_val = ds.add()
    assert new_val == 2
    assert ds.value_count() == 3
    assert ds[new_val] == new_val


def test_disjoint_set_find_invalid():
    ds = IntDisjointSet(size=3)
    with pytest.raises(KeyError):
        ds[3]
    with pytest.raises(KeyError):
        ds[-1]


def test_disjoint_set_union():
    ds = IntDisjointSet(size=4)

    # Union 0 and 1
    assert ds.union(0, 1)
    root = ds[0]
    assert ds[1] == root
    assert ds.connected(0, 1)
    assert not ds.connected(0, 2)

    # Union 2 and 3
    assert ds.union(2, 3)
    root2 = ds[2]
    assert ds[3] == root2
    assert ds.connected(2, 3)
    assert not ds.connected(1, 2)

    # Union already connected elements
    assert not ds.union(0, 1)
    assert ds.connected(0, 1)

    # Union two sets
    assert ds.union(1, 2)
    final_root = ds[0]
    assert ds[1] == final_root
    assert ds[2] == final_root
    assert ds[3] == final_root
    # After unioning all elements, they should all be connected
    assert ds.connected(0, 1)
    assert ds.connected(1, 2)
    assert ds.connected(2, 3)
    assert ds.connected(0, 3)


def test_disjoint_set_path_compression():
    ds = IntDisjointSet(size=4)

    # Create a chain: 3->2->1->0
    ds._parent = [0, 0, 1, 2]  # pyright: ignore[reportPrivateUsage]
    ds._count = [4, 3, 2, 1]  # pyright: ignore[reportPrivateUsage]

    # Find should compress the path
    root = ds[3]
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
    assert ds.connected("a", "b")
    assert not ds.connected("a", "c")

    # Union c and d
    assert ds.union("c", "d")
    root2 = ds.find("c")
    assert ds.find("d") == root2
    assert ds.connected("c", "d")
    assert not ds.connected("b", "c")

    # Union already connected elements
    assert not ds.union("a", "b")
    assert ds.connected("a", "b")

    # Union two sets
    assert ds.union("b", "c")
    final_root = ds.find("a")
    assert ds.find("b") == final_root
    assert ds.find("c") == final_root
    assert ds.find("d") == final_root
    # After unioning all elements, they should all be connected
    assert ds.connected("a", "b")
    assert ds.connected("b", "c")
    assert ds.connected("c", "d")
    assert ds.connected("a", "d")


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


def test_generic_disjoint_set_union_by_size():
    ds: DisjointSet[str] = DisjointSet(["a", "b", "c", "d"])
    ds.union("a", "b")
    ds.union("a", "c")
    # The set containing "a" is larger than the one containing "c"
    # so the canonical representative should be "a".
    assert ds.find("a") == "a"
    assert ds.find("c") == "a"

    ds.union("d", "c")
    assert ds.find("d") == "a"
    assert ds.find("c") == "a"


def test_int_disjoint_set_rooted_union():
    ds = IntDisjointSet(size=6)

    # Test successful rooted union
    assert ds.union_left(0, 1)  # Union with 0 as root
    assert ds[1] == 0
    assert ds.connected(0, 1)

    # Test rooted union with multiple elements
    assert ds.union_left(0, 2)  # Add 2 to the set rooted at 0
    assert ds[2] == 0
    assert ds.connected(0, 2)
    assert ds.connected(1, 2)

    # Test rooted union when sets are already connected
    assert not ds.union_left(0, 1)  # Should return False
    assert ds[1] == 0  # Still connected

    # Test error when representative is not a root
    ds.union_left(1, 3)  # 1 is not a root, 0 is its root
    assert ds[3] == 0
    assert ds.connected(1, 3)

    # Test that the specified representative becomes the root
    assert ds.union_left(4, 5)  # 4 becomes root of its own set with 5
    assert ds[4] == 4
    assert ds[5] == 4

    # Now union the two sets with 0 as the specified root
    assert ds.union_left(0, 4)
    assert ds[4] == 0  # 4's set is now under 0
    assert ds[5] == 0  # 5 also points to 0
    assert ds.connected(0, 5)


def test_generic_disjoint_set_rooted_union():
    ds = DisjointSet(["a", "b", "c", "d", "e", "f"])

    # Test successful rooted union
    assert ds.union_left("a", "b")  # Union with "a" as root
    assert ds.find("b") == "a"
    assert ds.connected("a", "b")

    # Test rooted union with multiple elements
    assert ds.union_left("a", "c")  # Add "c" to the set rooted at "a"
    assert ds.find("c") == "a"
    assert ds.connected("a", "c")
    assert ds.connected("b", "c")

    # Test rooted union when sets are already connected
    assert not ds.union_left("a", "b")  # Should return False
    assert ds.find("b") == "a"  # Still connected

    # Test error when representative is not a root
    ds.union_left("b", "d")  # "b" is not a root, "a" is its root
    assert ds.find("d") == "a"
    assert ds.connected("b", "d")

    # Test that the specified representative becomes the root
    assert ds.union_left("e", "f")  # "e" becomes root of its own set with "f"
    assert ds.find("e") == "e"
    assert ds.find("f") == "e"

    # Now union the two sets with "a" as the specified root
    assert ds.union_left("a", "e")
    assert ds.find("e") == "a"  # "e"'s set is now under "a"
    assert ds.find("f") == "a"  # "f" also points to "a"
    assert ds.connected("a", "f")

    # Test with KeyError for non-existent values
    with pytest.raises(KeyError):
        ds.union_left("a", "nonexistent")

    with pytest.raises(KeyError):
        ds.union_left("nonexistent", "a")
