from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass


class _AffineExprKind(Enum):
    """Enum for the kind of storage node used in AffineExpr."""

    Add = auto()
    Mul = auto()
    Mod = auto()
    FloorDiv = auto()
    CeilDiv = auto()
    Constant = auto()
    DimId = auto()
    SymbolId = auto()

    def get_token(self):
        """Get the token corresponding to the node kind."""
        match self:
            case _AffineExprKind.Add:
                return "+"
            case _AffineExprKind.Mul:
                return "*"
            case _AffineExprKind.Mod:
                return "mod"
            case _AffineExprKind.FloorDiv:
                return "floordiv"
            case _AffineExprKind.CeilDiv:
                return "ceildiv"
            case _AffineExprKind.Constant:
                return "const"
            case _AffineExprKind.DimId:
                return "d"
            case _AffineExprKind.SymbolId:
                return "s"


@dataclass
class _AffineExprStorage:
    """Base class for affine expression storage nodes."""

    kind: _AffineExprKind


@dataclass
class _AffineBinaryOpExprStorage(_AffineExprStorage):
    """An affine expression storage node representing a binary operation."""

    lhs: AffineExpr
    rhs: AffineExpr

    def __post_init__(self) -> None:
        if self.kind not in {
            _AffineExprKind.Add,
            _AffineExprKind.Mul,
            _AffineExprKind.Mod,
            _AffineExprKind.FloorDiv,
            _AffineExprKind.CeilDiv,
        }:
            raise ValueError(f"Invalid kind {self.kind} for _AffineBinaryOpExprStorage")

    def __str__(self) -> str:
        return f"({self.lhs} {self.kind.get_token()} {self.rhs})"


@dataclass
class _AffineDimExprStorage(_AffineExprStorage):
    """An affine expression storage node representing a dimension or symbol."""

    position: int
    """
    The position of the dimension or symbol. Position of dimension and symbol
    starts from 0 and is independent of each other. For example, if there are 2
    dimensions and 3 symbols, then the positions of the dimensions are 0 and 1,
    and the positions of the symbols are 0, 1, and 2.
    """

    def __post_init__(self) -> None:
        if self.kind != _AffineExprKind.DimId and self.kind != _AffineExprKind.SymbolId:
            raise ValueError(f"Invalid kind {self.kind} for _AffineDimExprStorage")
        self.kind = self.kind
        self.position = self.position

    def __str__(self) -> str:
        return f"{self.kind.get_token()}{self.position}"


@dataclass
class _AffineConstantExprStorage(_AffineExprStorage):
    """An affine expression storage node representing a constant."""

    value: int

    def __init__(self, value: int) -> None:
        self.kind = _AffineExprKind.Constant
        self.value = value

    def __str__(self) -> str:
        return f"{self.value}"


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

    _impl: _AffineExprStorage

    @staticmethod
    def constant(value: int) -> AffineExpr:
        return AffineExpr(_AffineConstantExprStorage(value))

    @staticmethod
    def dimension(position: int) -> AffineExpr:
        return AffineExpr(_AffineDimExprStorage(_AffineExprKind.DimId, position))

    @staticmethod
    def symbol(position: int) -> AffineExpr:
        return AffineExpr(_AffineDimExprStorage(_AffineExprKind.SymbolId, position))

    def eval(self, dims: list[int], symbols: list[int]) -> int:
        """Evaluate the affine expression with the given dimension and symbol values."""
        if isinstance(self._impl, _AffineConstantExprStorage):
            return self._impl.value

        if isinstance(self._impl, _AffineDimExprStorage):
            match self._impl.kind:
                case _AffineExprKind.DimId:
                    return dims[self._impl.position]
                case _AffineExprKind.SymbolId:
                    return symbols[self._impl.position]
                case _:
                    raise ValueError(f"Unreachable")

        if isinstance(self._impl, _AffineBinaryOpExprStorage):
            lhs = self._impl.lhs.eval(dims, symbols)
            rhs = self._impl.rhs.eval(dims, symbols)

            if self._impl.kind == _AffineExprKind.Add:
                return lhs + rhs
            elif self._impl.kind == _AffineExprKind.Mul:
                return lhs * rhs
            elif self._impl.kind == _AffineExprKind.Mod:
                return lhs % rhs
            elif self._impl.kind == _AffineExprKind.FloorDiv:
                return lhs // rhs
            elif self._impl.kind == _AffineExprKind.CeilDiv:
                return -(-lhs // rhs)

        raise ValueError("Unreachable")

    def __add__(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)
        # TODO (#1086): Simplify addition here before returning.
        return AffineExpr(_AffineBinaryOpExprStorage(_AffineExprKind.Add, self, other))

    def __radd__(self, other: AffineExpr | int) -> AffineExpr:
        return self.__add__(other)

    def __neg__(self) -> AffineExpr:
        return self * -1

    def __sub__(self, other: AffineExpr | int) -> AffineExpr:
        return self + (-1 * other)

    def __rsub__(self, other: AffineExpr | int) -> AffineExpr:
        return self.__sub__(other)

    def __mul__(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)
        if other._impl.kind != _AffineExprKind.Constant:
            # TODO (#1087): MLIR also supports multiplication by symbols also, making
            # maps semi-affine. Currently, we do not implement semi-affine maps.
            raise NotImplementedError(
                "Multiplication with non-constant (semi-affine) is not supported yet"
            )
        # TODO (#1086): Simplify multiplication here before returning.
        return AffineExpr(_AffineBinaryOpExprStorage(_AffineExprKind.Mul, self, other))

    def __rmul__(self, other: AffineExpr | int) -> AffineExpr:
        return self.__mul__(other)

    def floor_div(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)
        if other._impl.kind != _AffineExprKind.Constant:
            # TODO (#1087): MLIR also supports floor-division by symbols also, making
            # maps semi-affine. Currently, we do not implement semi-affine maps.
            raise NotImplementedError(
                "Floor division with non-constant (semi-affine) is not supported yet"
            )
        # TODO (#1086): Simplify floor division here before returning.
        return AffineExpr(
            _AffineBinaryOpExprStorage(_AffineExprKind.FloorDiv, self, other)
        )

    def ceil_div(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)
        if other._impl.kind != _AffineExprKind.Constant:
            # TODO (#1087): MLIR also supports ceil-division by symbols also, making
            # maps semi-affine. Currently, we do not implement semi-affine maps.
            raise NotImplementedError(
                "Ceil division with non-constant (semi-affine) is not supported yet"
            )
        # TODO (#1086): Simplify ceil division here before returning.
        return AffineExpr(
            _AffineBinaryOpExprStorage(_AffineExprKind.CeilDiv, self, other)
        )

    def __mod__(self, other: AffineExpr | int) -> AffineExpr:
        if isinstance(other, int):
            other = AffineExpr.constant(other)
        if other._impl.kind != _AffineExprKind.Constant:
            # TODO (#1087): MLIR also supports Mod by symbols also, making maps
            # semi-affine. Currently, we do not implement semi-affine maps.
            raise NotImplementedError(
                "Mod with non-constant (semi-affine) is not supported yet"
            )
        # TODO (#1086): Simplify modulo here before returning.
        return AffineExpr(_AffineBinaryOpExprStorage(_AffineExprKind.Mod, self, other))

    def __str__(self) -> str:
        return str(self._impl)


@dataclass
class AffineMap:
    """
    AffineMap represents a map from a set of dimensions and symbols to a
    multi-dimensional affine expression.
    """

    num_dims: int
    num_symbols: int
    results: list[AffineExpr]

    def eval(self, dims: list[int], symbols: list[int]) -> list[int]:
        """Evaluate the AffineMap given the values of dimensions and symbols."""
        assert len(dims) == self.num_dims
        assert len(symbols) == self.num_symbols
        return [expr.eval(dims, symbols) for expr in self.results]

    def __str__(self) -> str:
        # Create comma seperated list of dims.
        dims = [
            _AffineExprKind.DimId.get_token() + str(i) for i in range(self.num_dims)
        ]
        dims = ", ".join(dims)
        # Create comma seperated list of symbols.
        syms = [
            _AffineExprKind.SymbolId.get_token() + str(i)
            for i in range(self.num_symbols)
        ]
        syms = ", ".join(syms)
        # Create comma seperated list of results.
        results = ", ".join(str(expr) for expr in self.results)

        return f"({dims})[{syms}] -> ({results})"
