from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from typing_extensions import assert_never

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
    from xdsl.ir.affine import AffineMap


@dataclass(frozen=True)
class AffineExpr:
    """
    An AffineExpr models an affine expression, which is a linear combination of
    dimensions with integer coefficients. For example, 2 * d0 + 3 * d1 is an
    affine expression, where d0, d1 are dimensions. An AffineExpr can be
    parameterized by symbols. AffineExpr also allows further extensions of an
    affine expression. Quasi-affine expressions, i.e. Integer division and
    modulo with a constant are allowed. For example, 2 * d0 + 3 * d1 + 4
    floordiv 5 is a quasi-affine expression. Semi-affine expressions i.e.
    Integer division and modulo with a symbol are also allowed. For example, 2
    * d0 + 3 * d1 + 4 floordiv s0 is a semi-affine expression.
    """

    @staticmethod
    def constant(value: int) -> AffineExpr:
        return AffineConstantExpr(value)

    @staticmethod
    def dimension(position: int) -> AffineExpr:
        return AffineDimExpr(position)

    @staticmethod
    def symbol(position: int) -> AffineExpr:
        return AffineSymExpr(position)

    @staticmethod
    def binary(
        kind: AffineBinaryOpKind,
        lhs: AffineExpr,
        rhs: AffineExpr,
    ) -> AffineExpr:
        """
        Builds a binary expression of the given kind using the operator function associated with that kind.
        As a consequence, binary expressions are simplified during construction.
        This may lead to the resulting expression not being of the type `AffineBinaryOpExpr`, but of the type that binary op is
        simplified to.
        This simplification does not occur when an `AffineBinaryOpExpr` is directly created using its constructor.

        Example:
        An expression of kind `AffineBinaryKind.Add` is built using the `AffineExpr.__add__` function.
        If both `rhs` and `lhs` are `AffineConstantExprs` this function returns an `AffineConstantExpr` of value `rhs` + `lhs`.
        """

        match kind:
            case AffineBinaryOpKind.Add:
                return lhs + rhs
            case AffineBinaryOpKind.Mul:
                return lhs * rhs
            case AffineBinaryOpKind.Mod:
                return lhs % rhs
            case AffineBinaryOpKind.FloorDiv:
                return lhs // rhs
            case AffineBinaryOpKind.CeilDiv:
                return lhs.ceil_div(rhs)
            case _:
                assert_never(kind)

    def compose(self, map: AffineMap) -> AffineExpr:
        """
        Compose with an AffineMap.

        Returns the composition of this AffineExpr with map.

        Prerequisites: this and map are composable, i.e. that the number of
        AffineDimExpr of this is smaller than the number of results of map.
        If a result of a map does not have a corresponding AffineDimExpr, that result
        simply does not appear in the produced AffineExpr.

        Example:
        ```
        expr: d0 + d2
        map: (d0, d1, d2)[s0, s1] -> (d0 + s1, d1 + s0, d0 + d1 + d2)
        returned expr: d0 * 2 + d1 + d2 + s1
        ```
        """
        return self.replace_dims_and_symbols(map.results, ())

    def replace_dims_and_symbols(
        self, new_dims: Sequence[AffineExpr], new_symbols: Sequence[AffineExpr]
    ) -> AffineExpr:
        """Replace the symbols and indices in this map with the ones provided."""
        if isinstance(self, AffineConstantExpr):
            return self

        if isinstance(self, AffineDimExpr):
            if self.position >= len(new_dims):
                return self
            return new_dims[self.position]

        if isinstance(self, AffineSymExpr):
            if self.position >= len(new_symbols):
                return self
            return new_symbols[self.position]

        if isinstance(self, AffineBinaryOpExpr):
            lhs = self.lhs.replace_dims_and_symbols(new_dims, new_symbols)
            rhs = self.rhs.replace_dims_and_symbols(new_dims, new_symbols)

            return AffineExpr.binary(
                lhs=lhs,
                rhs=rhs,
                kind=self.kind,
            )

        raise ValueError("Unreachable")

    def eval(self, dims: Sequence[int], symbols: Sequence[int]) -> int:
        """Evaluate the affine expression with the given dimension and symbol values."""
        if isinstance(self, AffineConstantExpr):
            return self.value

        if isinstance(self, AffineDimExpr):
            return dims[self.position]
        if isinstance(self, AffineSymExpr):
            return symbols[self.position]

        if isinstance(self, AffineBinaryOpExpr):
            lhs = self.lhs.eval(dims, symbols)
            rhs = self.rhs.eval(dims, symbols)

            if self.kind == AffineBinaryOpKind.Add:
                return lhs + rhs
            elif self.kind == AffineBinaryOpKind.Mul:
                return lhs * rhs
            elif self.kind == AffineBinaryOpKind.Mod:
                return lhs % rhs
            elif self.kind == AffineBinaryOpKind.FloorDiv:
                return lhs // rhs
            elif self.kind == AffineBinaryOpKind.CeilDiv:
                return -(-lhs // rhs)

        raise ValueError("Unreachable")

    def _try_fold_constant(
        self, other: AffineExpr, kind: AffineBinaryOpKind
    ) -> AffineExpr | None:
        if not isinstance(self, AffineConstantExpr):
            return None
        if not isinstance(other, AffineConstantExpr):
            return None

        match kind:
            case AffineBinaryOpKind.Add:
                return AffineExpr.constant(self.value + other.value)
            case AffineBinaryOpKind.Mul:
                return AffineExpr.constant(self.value * other.value)
            case AffineBinaryOpKind.Mod:
                return AffineExpr.constant(self.value % other.value)
            case AffineBinaryOpKind.FloorDiv:
                return AffineExpr.constant(self.value // other.value)
            case AffineBinaryOpKind.CeilDiv:
                return AffineExpr.constant(-(-self.value // other.value))

    def _simplify_add(self, other: AffineExpr) -> AffineExpr | None:
        """Simplify addition. Constant is assumed to be on RHS."""
        # Fold constants.
        if fold := self._try_fold_constant(other, AffineBinaryOpKind.Add):
            return fold
        # Ignore addition with 0.
        if isinstance(other, AffineConstantExpr) and other.value == 0:
            return self
        # Fold (expr + constant) + constant.
        if isinstance(self, AffineBinaryOpExpr) and self.kind == AffineBinaryOpKind.Add:
            if fold := self.rhs._try_fold_constant(other, AffineBinaryOpKind.Add):
                return self.lhs + fold
        return None

    def __add__(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)
        # Canonicalize the expression so that the constant is always on the RHS.
        if isinstance(self, AffineConstantExpr):
            self, other = other, self
        if simplified := self._simplify_add(other):
            return simplified
        return AffineBinaryOpExpr(AffineBinaryOpKind.Add, self, other)

    def __radd__(self, other: AffineExpr | int) -> AffineExpr:
        return self.__add__(other)

    def __neg__(self) -> AffineExpr:
        if isinstance(self, AffineConstantExpr):
            return AffineExpr.constant(-self.value)
        return self * -1

    def __sub__(self, other: AffineExpr | int) -> AffineExpr:
        return self + (-1 * other)

    def __rsub__(self, other: AffineExpr | int) -> AffineExpr:
        return self.__sub__(other)

    def _simplify_mul(self, other: AffineExpr) -> AffineExpr | None:
        """Simplify multiplication. Constant is assumed to be on RHS."""
        # Fold constant.
        if fold := self._try_fold_constant(other, AffineBinaryOpKind.Mul):
            return fold
        # Ignore multiplication by 1.
        if isinstance(other, AffineConstantExpr) and other.value == 1:
            return self
        # Fold (expr * constant) * constant.
        if isinstance(self, AffineBinaryOpExpr) and self.kind == AffineBinaryOpKind.Mul:
            if fold := self.rhs._try_fold_constant(other, AffineBinaryOpKind.Mul):
                return self.lhs * fold
        # Fold (expr + expr) * constant.
        if (
            isinstance(self, AffineBinaryOpExpr)
            and self.kind == AffineBinaryOpKind.Add
            and isinstance(other, AffineConstantExpr)
        ):
            return self.lhs * other + self.rhs * other
        return None

    def __mul__(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)
        # Canonicalize the expression so that the constant is always on the RHS.
        if isinstance(self, AffineConstantExpr):
            self, other = other, self
        if simplified := self._simplify_mul(other):
            return simplified
        if not isinstance(other, AffineConstantExpr):
            # TODO (#1087): MLIR also supports multiplication by symbols also, making
            # maps semi-affine. Currently, we do not implement semi-affine maps.
            raise NotImplementedError(
                f"Multiplication with non-constant (semi-affine) is not supported yet self: {self} other: {other}"
            )
        return AffineBinaryOpExpr(AffineBinaryOpKind.Mul, self, other)

    def __rmul__(self, other: AffineExpr | int) -> AffineExpr:
        return self.__mul__(other)

    def __floordiv__(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)

        # Fold constants.
        if fold := self._try_fold_constant(other, AffineBinaryOpKind.FloorDiv):
            return fold

        if not isinstance(other, AffineConstantExpr):
            # TODO (#1087): MLIR also supports floor-division by symbols also, making
            # maps semi-affine. Currently, we do not implement semi-affine maps.
            raise NotImplementedError(
                "Floor division with non-constant (semi-affine) is not supported yet"
            )
        # TODO (#1086): Simplify floor division here before returning.
        return AffineBinaryOpExpr(AffineBinaryOpKind.FloorDiv, self, other)

    def ceil_div(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)

        # Fold constants.
        if fold := self._try_fold_constant(other, AffineBinaryOpKind.CeilDiv):
            return fold

        if not isinstance(other, AffineConstantExpr):
            # TODO (#1087): MLIR also supports ceil-division by symbols also, making
            # maps semi-affine. Currently, we do not implement semi-affine maps.
            raise NotImplementedError(
                "Ceil division with non-constant (semi-affine) is not supported yet"
            )
        # TODO (#1086): Simplify ceil division here before returning.
        return AffineBinaryOpExpr(AffineBinaryOpKind.CeilDiv, self, other)

    def __mod__(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)

        # Fold constants.
        if fold := self._try_fold_constant(other, AffineBinaryOpKind.Mod):
            return fold

        if not isinstance(other, AffineConstantExpr):
            # TODO (#1087): MLIR also supports Mod by symbols also, making maps
            # semi-affine. Currently, we do not implement semi-affine maps.
            raise NotImplementedError(
                "Mod with non-constant (semi-affine) is not supported yet"
            )
        # TODO (#1086): Simplify modulo here before returning.
        return AffineBinaryOpExpr(AffineBinaryOpKind.Mod, self, other)

    @abstractmethod
    def dfs(self) -> Iterator[AffineExpr]:
        """
        Iterates nodes in depth-first order.

        See external [documentation](https://en.wikipedia.org/wiki/Depth-first_search).
        """
        yield self

    def used_dims(self) -> set[int]:
        return {expr.position for expr in self.dfs() if isinstance(expr, AffineDimExpr)}


class AffineBinaryOpKind(Enum):
    """Enum for the kind of storage node used in AffineExpr."""

    Add = auto()
    Mul = auto()
    Mod = auto()
    FloorDiv = auto()
    CeilDiv = auto()

    def get_token(self) -> str:
        match self:
            case AffineBinaryOpKind.Add:
                return "+"
            case AffineBinaryOpKind.Mul:
                return "*"
            case AffineBinaryOpKind.Mod:
                return "mod"
            case AffineBinaryOpKind.FloorDiv:
                return "floordiv"
            case AffineBinaryOpKind.CeilDiv:
                return "ceildiv"


@dataclass(frozen=True)
class AffineBinaryOpExpr(AffineExpr):
    """An affine expression storage node representing a binary operation."""

    kind: AffineBinaryOpKind
    lhs: AffineExpr
    rhs: AffineExpr

    def __str__(self) -> str:
        return f"({self.lhs} {self.kind.get_token()} {self.rhs})"

    def dfs(self) -> Iterator[AffineExpr]:
        yield self
        yield from self.lhs.dfs()
        yield from self.rhs.dfs()


@dataclass(frozen=True)
class AffineDimExpr(AffineExpr):
    """An affine expression storage node representing a dimension."""

    position: int

    def __str__(self) -> str:
        return f"d{self.position}"


@dataclass(frozen=True)
class AffineSymExpr(AffineExpr):
    """An affine expression storage node representing a symbol."""

    position: int

    def __str__(self) -> str:
        return f"s{self.position}"


@dataclass(frozen=True)
class AffineConstantExpr(AffineExpr):
    """An affine expression storage node representing a constant."""

    value: int

    def __str__(self) -> str:
        return f"{self.value}"
