from dataclasses import dataclass
from enum import Enum

from xdsl.ir.affine import AffineExpr


class AffineConstraintKind(Enum):
    ge = ">="
    le = "<="
    eq = "=="


@dataclass(frozen=True)
class AffineConstraintExpr:
    kind: AffineConstraintKind
    lhs: AffineExpr
    rhs: AffineExpr

    def __init__(
        self,
        kind: AffineConstraintKind,
        lhs: AffineExpr,
        rhs: AffineExpr,
        *,
        canonicalize: bool = True,
    ):
        if canonicalize:
            lhs = lhs - rhs
            rhs = AffineExpr.constant(0)

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lhs", lhs)
        object.__setattr__(self, "rhs", rhs)

    def __str__(self) -> str:
        return f"{self.lhs} {self.kind.value} {self.rhs}"


@dataclass(frozen=True)
class AffineSet:
    """
    AffineMap represents a map from a set of dimensions and symbols to a
    multi-dimensional affine expression.
    """

    num_dims: int
    num_symbols: int
    constraints: tuple[AffineConstraintExpr, ...]

    def __str__(self) -> str:
        # Create comma seperated list of dims.
        dims = ", ".join("d" + str(i) for i in range(self.num_dims))
        # Create comma seperated list of symbols.
        syms = ["s" + str(i) for i in range(self.num_symbols)]
        syms = ", ".join(syms)
        # Create comma seperated list of results.
        constraints = ", ".join(str(cnstr) for cnstr in self.constraints)
        if self.num_symbols == 0:
            return f"({dims}) : ({constraints})"
        return f"({dims})[{syms}] : ({constraints})"
