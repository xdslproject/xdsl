from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir.affine import AffineDimExpr, AffineExpr


@dataclass
class AffineMap:
    """
    AffineMap represents a map from a set of dimensions and symbols to a
    multi-dimensional affine expression.
    """

    num_dims: int
    num_symbols: int
    results: list[AffineExpr]

    @staticmethod
    def constant_map(value: int) -> AffineMap:
        return AffineMap(0, 0, [AffineExpr.constant(value)])

    @staticmethod
    def point_map(*values: int) -> AffineMap:
        return AffineMap(0, 0, [AffineExpr.constant(value) for value in values])

    @staticmethod
    def identity(rank: int) -> AffineMap:
        return AffineMap(rank, 0, [AffineExpr.dimension(dim) for dim in range(rank)])

    @staticmethod
    def empty() -> AffineMap:
        return AffineMap(0, 0, [])

    def compose(self, map: AffineMap) -> AffineMap:
        """Compose the AffineMap with the given AffineMap."""
        if self.num_dims != map.num_dims:
            raise ValueError(
                f"Cannot compose AffineMaps with different numbers of dimensions: "
                f"{self.num_dims} and {map.num_dims}"
            )

        results = [expr.compose(map) for expr in self.results]
        return AffineMap(
            num_dims=self.num_dims,
            num_symbols=map.num_symbols,
            results=results,
        )

    def inverse_permutation(self) -> AffineMap | None:
        """
        Returns a map of codomain to domain dimensions such that the first
        codomain dimension for a particular domain dimension is selected.
        Returns an empty map if the input map is empty. Returns null map (not
        empty map) if the map is not invertible (i.e. the map does not contain
        a subset that is a permutation of full domain rank).

        Prerequisites: The map should have no symbols.

        Example:
           (d0, d1, d2) -> (d1, d1, d0, d2, d1, d2, d1, d0)
                             0       2   3
        returns:
           (d0, d1, d2, d3, d4, d5, d6, d7) -> (d2, d0, d3)
        """
        if self.num_symbols != 0:
            raise ValueError(
                f"Cannot invert AffineMap with symbols: {self.num_symbols}"
            )
        found_dims = [-1] * self.num_dims

        for i, expr in enumerate(self.results):
            match expr:
                case AffineDimExpr():
                    found_dims[expr.position] = i
                case _:
                    continue

        if -1 in found_dims:
            return None

        results = [self.results[i] for i in found_dims]
        return AffineMap(
            num_dims=len(self.results),
            num_symbols=0,
            results=results,
        )

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
