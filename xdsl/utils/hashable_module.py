from collections.abc import Hashable
from dataclasses import dataclass

from xdsl.dialects.builtin import ModuleOp
from xdsl.utils.hasher import Hasher


@dataclass(frozen=True)
class HashableModule(Hashable):
    """
    A class wrapping a ModuleOp, that forwards to structural equality checking for `==`.
    The module in this class should not be mutated.
    """

    module: ModuleOp

    def __eq__(self, other: object) -> bool:
        return isinstance(
            other, HashableModule
        ) and self.module.is_structurally_equivalent(other.module)

    def __hash__(self) -> int:
        """
        The hash of the module is a hash of the ordered combination of operation names.
        As most transformations on IR modify at least one operation, this should be
        enough to minimise collisions.
        """
        hasher = Hasher()
        for op in self.module.walk():
            hasher.combine(op.name)
        return hasher.hash
