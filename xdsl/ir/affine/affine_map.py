from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from inspect import getfullargspec

from xdsl.ir.affine import AffineDimExpr, AffineExpr

AffineExprBuilderT = AffineExpr | int

AffineMapBuilderT = (
    Callable[[], tuple[AffineExprBuilderT, ...]]
    | Callable[[AffineExpr], tuple[AffineExprBuilderT, ...]]
    | Callable[[AffineExpr, AffineExpr], tuple[AffineExprBuilderT, ...]]
    | Callable[[AffineExpr, AffineExpr, AffineExpr], tuple[AffineExprBuilderT, ...]]
    | Callable[
        [AffineExpr, AffineExpr, AffineExpr, AffineExpr], tuple[AffineExprBuilderT, ...]
    ]
    | Callable[
        [AffineExpr, AffineExpr, AffineExpr, AffineExpr, AffineExpr],
        tuple[AffineExprBuilderT, ...],
    ]
)


@dataclass(frozen=True)
class AffineMap:
    """
    AffineMap represents a map from a set of dimensions and symbols to a
    multi-dimensional affine expression.
    """

    num_dims: int
    num_symbols: int
    results: tuple[AffineExpr, ...]

    @staticmethod
    def constant_map(value: int) -> AffineMap:
        return AffineMap(0, 0, (AffineExpr.constant(value),))

    @staticmethod
    def point_map(*values: int) -> AffineMap:
        return AffineMap(0, 0, tuple(AffineExpr.constant(value) for value in values))

    @staticmethod
    def identity(rank: int) -> AffineMap:
        return AffineMap(
            rank, 0, tuple(AffineExpr.dimension(dim) for dim in range(rank))
        )

    @staticmethod
    def transpose_map() -> AffineMap:
        """
        Returns the map transposing a 2D matrix: `(i, j) -> (j, i)`.
        """
        return AffineMap(2, 0, (AffineExpr.dimension(1), AffineExpr.dimension(0)))

    @staticmethod
    def empty() -> AffineMap:
        return AffineMap(0, 0, ())

    @staticmethod
    def from_callable(
        func: AffineMapBuilderT, *, dim_symbol_split: tuple[int, int] | None = None
    ) -> AffineMap:
        """
        Creates an `AffineMap` by calling the function provided. If `dim_symbol_split` is
        not provided or `None`, then all parameters are treated as dimension expressions.
        If `dim_symbol_split` is provided, `func` is expected to have the same number of
        arguments as the sum of elements of `dim_symbol_split`.

        3D Identity:
        ```
        AffineMap.from_callable(lambda i, j, k: (i, j, k))
        ```
        Constant:
        ```
        AffineMap.from_callable(lambda i, j: (0, 0))
        ```
        Mix of dimensions and symbols:
        ```
        AffineMap.from_callable(lambda i, p: (p, i), dim_symbol_split=(1,1))
        ```
        """
        sig = getfullargspec(func)
        num_args = len(sig.args)
        if dim_symbol_split is None:
            num_dims = num_args
            num_symbols = 0
        else:
            num_dims, num_symbols = dim_symbol_split
            if num_args != num_dims + num_symbols:
                raise ValueError(
                    f"Argument count mismatch in AffineMap.from_callable: {num_args} != "
                    f"{num_dims} + {num_symbols}"
                )
        dim_exprs = [AffineExpr.dimension(dim) for dim in range(num_dims)]
        sym_exprs = [AffineExpr.symbol(sym) for sym in range(num_symbols)]
        result_exprs = func(*dim_exprs, *sym_exprs)
        results_tuple = tuple(
            AffineExpr.constant(r) if isinstance(r, int) else r for r in result_exprs
        )
        return AffineMap(num_dims, num_symbols, results_tuple)

    def replace_dims_and_symbols(
        self,
        new_dims: Sequence[AffineExpr],
        new_symbols: Sequence[AffineExpr],
        result_num_dims: int,
        result_num_symbols: int,
    ) -> AffineMap:
        """
        This method substitutes any uses of dimensions and symbols (e.g. dim#0 with dimReplacements[0]) in subexpressions and returns the modified expression mapping.  Because this can be used to eliminate dims and symbols, the client needs to specify the number of dims and symbols in the result.

        The returned map always has the same number of results.
        """

        return AffineMap(
            result_num_dims,
            result_num_symbols,
            tuple(
                expr.replace_dims_and_symbols(new_dims, new_symbols)
                for expr in self.results
            ),
        )

    def compose(self, other: AffineMap) -> AffineMap:
        """
        Returns the `AffineMap` resulting from composing `self` with `other`.

        The resulting `AffineMap` has as many `AffineDimExpr` as `other` and as many `AffineSymbolExpr` as the concatenation of `self` and `other` (in which case the symbols of `self` come first).

        Prerequisites: The maps are composable, i.e. that the number of `AffineDimExpr` of `self` matches the number of results of `map`.

        Example:
        ```
        map1: (d0, d1)[s0, s1] -> (d0 + 1 + s1, d1 - 1 - s0)
        map2: (d0)[s0] -> (d0 + s0, d0 - s0)
        map1.compose(map2): (d0)[s0, s1, s2] -> (d0 + s1 + s2 + 1, d0 - s0 - s2 - 1)
        ```
        """
        if self.num_dims != len(other.results):
            raise ValueError(
                "Cannot compose AffineMaps with mismatching dimensions and results: "
                "self.num_dims != len(map.results) "
                f"({self.num_dims} != {len(other.results)})"
            )

        num_dims = other.num_dims
        num_symbols = self.num_symbols + other.num_symbols

        new_dims = tuple(AffineExpr.dimension(d) for d in range(num_dims))
        new_symbols = tuple(
            AffineExpr.symbol(s) for s in range(self.num_symbols, num_symbols)
        )

        new_map = other.replace_dims_and_symbols(
            new_dims, new_symbols, num_dims, num_symbols
        )

        results = tuple(expr.compose(new_map) for expr in self.results)
        return AffineMap(
            num_dims=num_dims,
            num_symbols=num_symbols,
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

        results = tuple(self.results[i] for i in found_dims)
        return AffineMap(
            num_dims=len(self.results),
            num_symbols=0,
            results=results,
        )

    def eval(self, dims: Sequence[int], symbols: Sequence[int]) -> list[int]:
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
