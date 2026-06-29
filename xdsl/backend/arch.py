"""
Helper superclass for convenience methods to do with functionality of a target
architecture.

"""

import abc


class Arch(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """
        The name that this architecture/microarchitecture can be referred by, should be
        unique per backend.
        """
