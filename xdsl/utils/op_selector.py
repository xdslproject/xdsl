from __future__ import annotations

from itertools import islice
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from xdsl.dialects.builtin import ModuleOp


class OpSelector(NamedTuple):
    """
    A specifier of an operation in a module.
    Useful when a stable reference is needed across copies of a module.
    """

    idx: int
    """
    The index of the operation in a module, when walking top to bottom as printed in the
    IR.
    """
    op_name: str
    """
    The name of the expected operation, used to check that the op found is the expected
    one.
    """

    def get_op(self, module_op: ModuleOp):
        """
        Returns the matching op, raising IndexError if the index is out of bounds and
        ValueError if the name does not match.
        """
        op = next(islice(module_op.walk(), self.idx, None), None)
        if op is None:
            raise IndexError(f"Matching index {self.idx} out of range.")
        if op.name != self.op_name:
            raise ValueError(
                f"Unexpected op {op.name} at index {self.idx}, expected {self.op_name}."
            )
        return op
