from collections.abc import Hashable


class Hasher:
    """
    A helper class that accumulates hash values over time.
    """

    hash: int

    def __init__(self, *, seed: int = 0):
        self.hash = seed

    def combine_hash(self, other_hash: int) -> None:
        self.hash ^= other_hash

    def combine(self, other: Hashable) -> None:
        self.combine_hash(hash(other))
