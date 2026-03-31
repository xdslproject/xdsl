"""
Generic implementation of a disjoint set data structure.

See external [documentation](https://en.wikipedia.org/wiki/Disjoint-set_data_structure).
"""

from collections.abc import Hashable, Sequence
from typing import Generic

from typing_extensions import TypeVar


class IntDisjointSet:
    """
    Represents a collection of disjoint sets of integers.
    The integers stored are always in the range [0,n), where n is the number of elements
    in this structure.

    This implementation uses path compression and union by size for efficiency.
    The amortized time complexity for operations is nearly constant.
    """

    _parent: list[int]
    """
    Index of the parent node. If the node is its own parent then it is a root node.
    """
    _count: list[int]
    """
    If the node is a root node, the corresponding value is the count of elements in the
    set. For non-root nodes, these counts may be stale and should not be used.
    """

    def __init__(self, *, size: int) -> None:
        """
        Initialize disjoint sets with elements [0,size).
        Each element starts in its own singleton set.
        """
        self._parent = list(range(size))
        self._count = [1] * size

    def value_count(self) -> int:
        """Number of nodes in this structure."""
        return len(self._parent)

    def add(self) -> int:
        """
        Add a new element to this set as a singleton.
        Returns the added value, which will be equal to the previous size.
        """
        res = len(self._parent)
        self._parent.append(res)
        self._count.append(1)
        return res

    def __getitem__(self, value: int) -> int:
        """
        Returns the root/representative value of this set.
        Uses path compression - updates parent pointers to point directly to the root
        as we traverse up the tree, improving amortized performance.
        """
        if value < 0 or len(self._parent) <= value:
            raise KeyError(f"Index {value} not found")

        # Find the root
        root = value
        while self._parent[root] != root:
            root = self._parent[root]

        # Path compression - point all nodes on path to root
        current = value
        while current != root:
            next_parent = self._parent[current]
            self._parent[current] = root
            current = next_parent

        return root

    def union_left(self, lhs: int, rhs: int) -> bool:
        """
        Merge the sets containing the two given values, with `rhs`'s tree being attached to `lhs`'s tree.
        Returns True if the sets were merged, False if they were already the same set.

        In contrast to `union`, this does not do union by size - the `rhs` set is always
        attached to the `lhs` set. This is useful when we want to control
        which element becomes the representative of the merged set.
        """
        lhs = self[lhs]
        rhs = self[rhs]
        if lhs == rhs:
            return False

        lhs_count = self._count[lhs]
        rhs_count = self._count[rhs]
        self._parent[rhs] = lhs
        self._count[lhs] = lhs_count + rhs_count
        # Note: We don't need to update _count[rhs] since it's no longer a root
        return True

    def union(self, lhs: int, rhs: int) -> bool:
        """
        Merges the sets containing lhs and rhs if they are different.
        Returns True if the sets were merged, False if they were already the same set.

        Uses union by size - the smaller tree is attached to the larger tree's root
        to maintain balance. This ensures the maximum tree height is O(log n).
        """
        lhs_root = self[lhs]
        rhs_root = self[rhs]
        if lhs_root == rhs_root:
            return False

        lhs_count = self._count[lhs_root]
        rhs_count = self._count[rhs_root]
        # Choose the root of the larger tree as the new parent
        new_parent, new_child = (
            (lhs_root, rhs_root) if lhs_count >= rhs_count else (rhs_root, lhs_root)
        )
        self._parent[new_child] = new_parent
        self._count[new_parent] = lhs_count + rhs_count
        # Note: We don't need to update _count[new_child] since it's no longer a root
        return True

    def connected(self, lhs: int, rhs: int) -> bool:
        return self[lhs] == self[rhs]


_T = TypeVar("_T", bound=Hashable)


class DisjointSet(Generic[_T]):
    """
    A disjoint-set data structure that works with arbitrary hashable values.
    Internally uses IntDisjointSet by mapping values to integer indices.
    """

    _base: IntDisjointSet
    _values: list[_T]
    _index_by_value: dict[_T, int]

    def __init__(self, values: Sequence[_T] = ()):
        """
        Initialize a DisjointSet with the given sequence of values.
        Each value starts in its own singleton set.

        Args:
            values: Initial sequence of values to add to the disjoint set
        """
        self._values = list(values)
        self._index_by_value = {v: i for i, v in enumerate(self._values)}
        self._base = IntDisjointSet(size=len(self._values))

    def __len__(self):
        return len(self._values)

    def add(self, value: _T):
        """
        Add a new value to the disjoint set in its own singleton set.

        Args:
            value: The value to add
        """
        index = self._base.add()
        self._values.append(value)
        self._index_by_value[value] = index

    def find(self, value: _T) -> _T:
        """
        Find the representative value for the set containing the given value.

        Returns the representative value for the set.

        Raises:
            KeyError: If the value is not in the disjoint set
        """
        index = self._base[self._index_by_value[value]]
        return self._values[index]

    def union_left(self, lhs: _T, rhs: _T) -> bool:
        """
        Merge the sets containing the two given values, with `rhs`'s tree being attached to `lhs`'s tree.
        Returns True if the sets were merged, False if they were already the same set.

        In contrast to `union`, this does not do union by size - the `rhs` set is always
        attached to the `lhs` set. This is useful when we want to control
        which element becomes the representative of the merged set.

        Raises:
            KeyError: If either value is not in the disjoint set
        """
        return self._base.union_left(
            self._index_by_value[lhs],
            self._index_by_value[rhs],
        )

    def union(self, lhs: _T, rhs: _T) -> bool:
        """
        Merge the sets containing the two given values if they are different.

        Returns `True` if the sets were merged, `False` if they were already the same
        set.

        Raises:
            KeyError: If either value is not in the disjoint set
        """
        return self._base.union(self._index_by_value[lhs], self._index_by_value[rhs])

    def connected(self, lhs: _T, rhs: _T) -> bool:
        """
        Returns `True` if the values are in the same set.

        Raises:
            KeyError: If either value is not in the disjoint set
        """
        return self._base.connected(
            self._index_by_value[lhs], self._index_by_value[rhs]
        )
