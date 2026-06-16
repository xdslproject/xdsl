"""
Helper superclass for convenience methods to do with functionality of a target
architecture.

"""

import abc

from typing_extensions import Self


class Arch(abc.ABC):
    @abc.abstractmethod
    @classmethod
    def arch_for_name(cls, name: str | None) -> Self:
        """
        Returns an instance of Arch for a given name, or default if none is provided.
        """
