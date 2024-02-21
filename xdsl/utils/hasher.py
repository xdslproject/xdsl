from collections.abc import Hashable


class Hasher:
    """
    A helper class that accumulates hash values over time. The value depends on the order
    in which hash values are mixed into `self`.
    """

    hash: int

    def __init__(self, *, seed: int = 0):
        self.hash = seed

    def combine_hash(self, other_hash: int) -> None:
        """
        boost's hash combine function

        https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
        """
        self.hash ^= other_hash + 0x9E3779B9 + (self.hash << 6) + (self.hash >> 2)

    def combine(self, other: Hashable) -> None:
        self.combine_hash(hash(other))
