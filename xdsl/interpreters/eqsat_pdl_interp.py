from __future__ import annotations

from dataclasses import dataclass, field

from xdsl.dialects import eqsat
from xdsl.dialects.builtin import ModuleOp
from xdsl.interpreter import (
    register_impls,
)
from xdsl.interpreters.pdl_interp import PDLInterpFunctions
from xdsl.transforms.common_subexpression_elimination import KnownOps
from xdsl.utils.disjoint_set import DisjointSet


@register_impls
@dataclass
class EqsatPDLInterpFunctions(PDLInterpFunctions):
    known_ops: KnownOps = field(default_factory=KnownOps)
    eclass_union_find: DisjointSet[eqsat.EClassOp] = field(
        default_factory=lambda: DisjointSet[eqsat.EClassOp]()
    )

    def populate_known_ops(self, module: ModuleOp) -> None:
        """
        Populates the known_ops dictionary by traversing the module.

        Args:
            module: The module to traverse
        """
        # Walk through all operations in the module
        for op in module.walk():
            # Skip EClassOp instances
            if not isinstance(op, eqsat.EClassOp):
                self.known_ops[op] = op
            else:
                self.eclass_union_find.add(op)
