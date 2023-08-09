from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
    from xdsl.ir.affine import AffineMap


@dataclass()
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

    def compose(self, map: AffineMap) -> AffineExpr:
        """Compose the affine expression with the given affine map."""
        if map.num_symbols != 0:
            raise NotImplementedError("AffineMap with symbol not supported yet")

        if isinstance(self, AffineConstantExpr):
            return self

        if isinstance(self, AffineDimExpr) or isinstance(self, AffineSymExpr):
            return map.results[self.position]

        if isinstance(self, AffineBinaryOpExpr):
            lhs = self.lhs.compose(map)
            rhs = self.rhs.compose(map)

            return AffineBinaryOpExpr(
                lhs=lhs,
                rhs=rhs,
                kind=self.kind,
            )

        raise ValueError("Unreachable")

    def eval(self, dims: list[int], symbols: list[int]) -> int:
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

    def floor_div(self, other: AffineExpr | int) -> AffineExpr:
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


@dataclass
class AffineBinaryOpExpr(AffineExpr):
    """An affine expression storage node representing a binary operation."""

    kind: AffineBinaryOpKind
    lhs: AffineExpr
    rhs: AffineExpr

    def __str__(self) -> str:
        return f"({self.lhs} {self.kind.get_token()} {self.rhs})"


@dataclass
class AffineDimExpr(AffineExpr):
    """An affine expression storage node representing a dimension."""

    position: int

    def __str__(self) -> str:
        return f"d{self.position}"


@dataclass
class AffineSymExpr(AffineExpr):
    """An affine expression storage node representing a symbol."""

    position: int

    def __str__(self) -> str:
        return f"s{self.position}"


@dataclass
class AffineConstantExpr(AffineExpr):
    """An affine expression storage node representing a constant."""

    value: int

    def __str__(self) -> str:
        return f"{self.value}"
