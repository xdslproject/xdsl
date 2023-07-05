from __future__ import annotations
from dataclasses import dataclass

from .affine_expr import AffineExpr


@dataclass
class AffineMap:
    """
    AffineMap represents a map from a set of dimensions and symbols to a
    multi-dimensional affine expression.
    """

    num_dims: int
    num_symbols: int
    results: list[AffineExpr]

    @property
    def is_constant(self) -> bool:
        """
        Returns True if it is possible to guarantee that the map is constant. There may be
        expressions that are constant in practice, but too complex to reason about, so
        this is a pessimistic estimate.
        """
        return not self.num_symbols

    @staticmethod
    def constant_map(value: int) -> AffineMap:
        return AffineMap(0, 0, [AffineExpr.constant(value)])

    @staticmethod
    def from_constants(*values: int) -> AffineMap:
        return AffineMap(0, 0, [AffineExpr.constant(value) for value in values])

    @staticmethod
    def identity(rank: int) -> AffineMap:
        return AffineMap(rank, 0, [AffineExpr.dimension(dim) for dim in range(rank)])

    @staticmethod
    def empty() -> AffineMap:
        return AffineMap(0, 0, [])

    def eval(self, dims: list[int], symbols: list[int]) -> list[int]:
        """Evaluate the AffineMap given the values of dimensions and symbols."""
        assert len(dims) == self.num_dims
        assert len(symbols) == self.num_symbols
        return [expr.eval(dims, symbols) for expr in self.results]

    def __str__(self) -> str:
        # Create comma seperated list of dims.
        dims = ["d" + str(i) for i in range(self.num_dims)]
        dims = ", ".join(dims)
        # Create comma seperated list of symbols.
        syms = ["s" + str(i) for i in range(self.num_symbols)]
        syms = ", ".join(syms)
        # Create comma seperated list of results.
        results = ", ".join(str(expr) for expr in self.results)
        if self.num_symbols == 0:
            return f"({dims}) -> ({results})"
        return f"({dims})[{syms}] -> ({results})"
