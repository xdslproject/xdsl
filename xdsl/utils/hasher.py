from collections.abc import Hashable


class Hasher:
    """
    A helper class that accumulates hash values over time.
    """

    hash: int

    def __init__(self, *, seed: int = 0):
        self.hash = seed

    def combine(self, other: Hashable) -> None:
        self.hash = hash((self.hash, other))
