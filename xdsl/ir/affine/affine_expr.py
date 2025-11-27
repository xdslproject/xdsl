from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain
from typing import TYPE_CHECKING

from typing_extensions import assert_never

# Used for cyclic dependencies in type hints
if TYPE_CHECKING:
    from xdsl.ir.affine import AffineMap


@dataclass(frozen=True)
class AffineExpr(ABC):
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

    @staticmethod
    def from_flat_form(
        flat_exprs: Sequence[int],
        num_dims: int,
        num_symbols: int,
        local_exprs: Sequence[AffineExpr],
    ) -> AffineExpr:
        """
        Constructs an affine expression from a flat list of coefficients.
        If there are local identifiers (neither dimensional nor symbolic) that appear in
        the sum of products expression, `local_exprs` is expected to have the AffineExpr
        for it, and is substituted into.
        The list `flat_exprs` is expected to be in the format [*dims, *symbols, *locals,
        constant term].
        """
        assert len(flat_exprs) == num_dims + num_symbols + len(local_exprs) + 1, (
            f"unexpected number of local expressions {len(local_exprs)}, expected "
            f"{len(flat_exprs) - num_dims - num_symbols - 1}"
        )

        expr = sum(
            (
                e * f
                for e, f in zip(
                    chain(
                        (AffineExpr.dimension(d) for d in range(num_dims)),
                        (AffineExpr.symbol(s) for s in range(num_symbols)),
                        local_exprs,
                    ),
                    flat_exprs[:-1],
                    strict=True,
                )
                if f != 0
            ),
            start=AffineExpr.constant(0),
        )

        # Constant term
        const_term = flat_exprs[-1]
        if const_term != 0:
            expr = expr + const_term

        return expr

    def simplify(self, num_dims: int, num_symbols: int) -> AffineExpr:
        """
        Simplify the affine expression by flattening it and reconstructing it.
        """
        if not self.is_pure_affine():
            # Simplify semi-affine expressions separately
            raise NotImplementedError(
                "Simplification of semi-affine expressions is not implemented yet."
            )

        flattener = SimpleAffineExprFlattener(num_dims, num_symbols)
        return flattener.simplify(self)

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

    def dfs(self) -> Iterator[AffineExpr]:
        """
        Iterates nodes in depth-first order (parent-left-right).

        See external [documentation](https://en.wikipedia.org/wiki/Depth-first_search).
        """
        yield self

    def post_order(self) -> Iterator[AffineExpr]:
        """
        Iterates nodes in pre-order (left-right-parent).

        See external [documentation](https://en.wikipedia.org/wiki/Tree_traversal).
        """
        yield self

    def used_dims(self) -> set[int]:
        return {expr.position for expr in self.dfs() if isinstance(expr, AffineDimExpr)}

    @abstractmethod
    def is_pure_affine(self) -> bool:
        """
        Returns true if this is a pure affine expression, i.e., multiplication,
        floordiv, ceildiv, and mod is only allowed w.r.t constants.
        """
        ...


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

    def dfs(self) -> Iterator[AffineExpr]:
        yield self
        yield from self.lhs.dfs()
        yield from self.rhs.dfs()

    def post_order(self) -> Iterator[AffineExpr]:
        yield from self.lhs.post_order()
        yield from self.rhs.post_order()
        yield self

    def is_pure_affine(self) -> bool:
        match self.kind:
            case AffineBinaryOpKind.Add:
                return self.lhs.is_pure_affine() and self.rhs.is_pure_affine()
            case AffineBinaryOpKind.Mul:
                # Multiplication is only allowed with a constant on the right
                # or left for pure affine expressions.
                # Check if either lhs or rhs is a constant and the other is pure affine.
                lhs_is_const = isinstance(self.lhs, AffineConstantExpr)
                rhs_is_const = isinstance(self.rhs, AffineConstantExpr)

                return (
                    (lhs_is_const or rhs_is_const)
                    and self.lhs.is_pure_affine()
                    and self.rhs.is_pure_affine()
                )
            case (
                AffineBinaryOpKind.Mod
                | AffineBinaryOpKind.FloorDiv
                | AffineBinaryOpKind.CeilDiv
            ):
                # Mod, floordiv, ceildiv are only allowed with a constant on the right for pure affine
                return (
                    isinstance(self.rhs, AffineConstantExpr)
                    and self.lhs.is_pure_affine()
                )

    def __str__(self) -> str:
        return f"({self.lhs} {self.kind.get_token()} {self.rhs})"


@dataclass(frozen=True)
class AffineDimExpr(AffineExpr):
    """An affine expression storage node representing a dimension."""

    position: int

    def is_pure_affine(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"d{self.position}"


@dataclass(frozen=True)
class AffineSymExpr(AffineExpr):
    """An affine expression storage node representing a symbol."""

    position: int

    def is_pure_affine(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"s{self.position}"


@dataclass(frozen=True)
class AffineConstantExpr(AffineExpr):
    """An affine expression storage node representing a constant."""

    value: int

    def is_pure_affine(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.value}"


class SimpleAffineExprFlattener:
    """
    This class is used to flatten a pure affine expression (AffineExpr, which is in a
    tree form) into a sum of products (with respect to constants) when possible, thereby
    simplifying the expression. For modulo, floordiv, or ceildiv expressions, an
    additional identifier, called a local identifier, is introduced to rewrite the
    expression as a sum of product affine expression. Each local identifier is always,
    by construction, a floordiv of a pure add/mul affine function of dimensional,
    symbolic, and other local identifiers, in a non-mutually recursive way. Thus, every
    local identifier can ultimately always be recovered as an affine function of
    dimensional and symbolic identifiers (involving floordiv's); note, however, that by
    AffineExpr construction, some floordiv combinations are converted to mod's.
    The result of the flattening is a flattened expression and a set of
    constraints involving just the local variables.

    For example, `d2 + (d0 + d1) // 4` is flattened to `d2 + q` where `q` is
    the local variable introduced, with `localVarCst` containing
    `4*q <= d0 + d1 <= 4*q + 3`.

    The simplification performed includes the accumulation of contributions for
    each dimensional and symbolic identifier together, the simplification of
    floordiv/ceildiv/mod expressions, and other simplifications that in turn
    happen as a result. A simplification that this flattening naturally performs
    is simplifying the numerator and denominator of floordiv/ceildiv, and
    folding a modulo expression to zero, if possible. Three examples are below:

    ```
    (d0 + 3 * d1) + d0) - 2 * d1) - d0    simplified to     d0 + d1
    (d0 - d0 % 4 + 4) % 4                 simplified to     0
    (3*d0 + 2*d1 + d0) // 2 + d1          simplified to     2*d0 + 2*d1
    ```

    The way the flattening works for the second example is as follows: `d0 % 4` is
    replaced by `d0 - 4*q` with `q` being introduced; the expression then simplifies
    to: `(d0 - (d0 - 4q) + 4) = 4q + 4`, modulo of which with respect to 4
    simplifies to zero. Note that an affine expression may not always be
    expressible purely as a sum of products involving just the original
    dimensional and symbolic identifiers due to the presence of
    modulo/floordiv/ceildiv expressions that may not be eliminated after
    simplification; in such cases, the final expression can be reconstructed by
    replacing the local identifiers with their corresponding explicit form
    stored in `localExprs` (note that each of the explicit forms itself would
    have been simplified).

    The expression walk method here performs a linear time post-order walk that
    performs the above simplifications through visit methods, with partial
    results being stored in `operandExprStack`. When a parent expr is visited,
    the flattened expressions corresponding to its two operands would already be
    on the stack—the parent expression looks at the two flattened expressions
    and combines the two. It pops off the operand expressions and pushes the
    combined result (although this is done in-place on its LHS operand expr).
    When the walk is completed, the flattened form of the top-level expression
    would be left on the stack.

    A flattener can be repeatedly used for multiple affine expressions that bind
    to the same operands, for example, for all result expressions of an
    AffineMap or AffineValueMap. In such cases, using it for multiple
    expressions is more efficient than creating a new flattener for each
    expression since common identical div and mod expressions appearing across
    different expressions are mapped to the same local identifier (same column
    position in `localVarCst`).
    """

    # Flattend expression layout: [dims, symbols, locals, constant]
    # Stack that holds the LHS and RHS operands while visiting a binary op expr.
    operand_expr_stack: list[list[int]]
    """
    Flattend expression layout: [dims, symbols, locals, constant]
    Stack that holds the LHS and RHS operands while visiting a binary op expr.
    """
    num_dims: int
    num_symbols: int
    local_exprs: list[AffineExpr]

    def __init__(self, num_dims: int, num_symbols: int) -> None:
        self.operand_expr_stack = []
        self.num_dims = num_dims
        self.num_symbols = num_symbols
        self.local_exprs = []

    def visit_mul_expr(self, expr: AffineBinaryOpExpr) -> None:
        """
        In pure affine t = expr * c, we multiply each coefficient of lhs with c.
        In case of semi affine multiplication expressions, `t = expr * symbolic_expr`,
        introduce a local variable `p (= expr * symbolic_expr)`, and the affine expression
        `expr * symbolic_expr`` is added to `localExprs`.
        """
        assert len(self.operand_expr_stack) >= 2
        rhs = self.operand_expr_stack.pop()
        lhs = self.operand_expr_stack.pop()

        if not isinstance(expr.rhs, AffineConstantExpr):
            # Flatten semi-affine multiplication expressions by introducing a local
            # variable in place of the product; the affine expression
            # corresponding to the quantifier is added to `localExprs`.
            raise NotImplementedError("Semi-affine map flattening not implemented")

        rhs_const = rhs[self.get_constant_index()]
        self.operand_expr_stack.append([l * rhs_const for l in lhs])

    def visit_add_expr(self, expr: AffineBinaryOpExpr) -> None:
        assert len(self.operand_expr_stack) >= 2
        rhs = self.operand_expr_stack.pop()
        lhs = self.operand_expr_stack.pop()
        assert len(lhs) == len(rhs)
        self.operand_expr_stack.append([l + r for l, r in zip(lhs, rhs, strict=True)])

    def visit_dim_expr(self, expr: AffineDimExpr) -> None:
        row = [0] * self.get_num_cols()
        assert expr.position < self.num_dims, "Inconsistent number of dims"
        row[self.get_dim_start_index() + expr.position] = 1
        self.operand_expr_stack.append(row)

    def visit_symbol_expr(self, expr: AffineSymExpr) -> None:
        # Equivalent to SimpleAffineExprFlattener::visitSymbolExpr
        row = [0] * self.get_num_cols()
        assert expr.position < self.num_symbols, "Inconsistent number of symbols"
        row[self.get_symbol_start_index() + expr.position] = 1
        self.operand_expr_stack.append(row)

    def visit_constant_expr(self, expr: AffineConstantExpr) -> None:
        # Equivalent to SimpleAffineExprFlattener::visitConstantExpr
        row = [0] * self.get_num_cols()
        row[self.get_constant_index()] = expr.value
        self.operand_expr_stack.append(row)

    def visit_div_expr(self, expr: AffineBinaryOpExpr, *, is_ceil: bool) -> None:
        """
        Handles floor and ceil division for affine expressions.

        `t = expr floordiv c   <=> t = q, c * q <= expr <= c * q + c - 1`

        A floordiv is thus flattened by introducing a new local variable q, and
        replacing that expression with 'q' while adding the constraints
        `c * q <= expr <= c * q + c - 1` to `local_var_cst` (done by
        `add_local_floor_div_id`).

        A ceildiv is similarly flattened:
        `t = expr ceildiv c   <=> t = (expr + c - 1) floordiv c`

        Semi-affine expressions are not yet implemented.
        """
        assert len(self.operand_expr_stack) >= 2

        rhs = self.operand_expr_stack.pop()
        lhs = self.operand_expr_stack.pop()

        # Semi-affine division: rhs is not a constant
        if not isinstance(expr.rhs, AffineConstantExpr):
            # Flatten semi-affine division expressions by introducing a local variable
            # in place of the quotient, and the affine expression is added to
            # `localExprs`.
            raise NotImplementedError("Semi-affine map flattening not implemented")

        rhs_const = rhs[self.get_constant_index()]
        if rhs_const <= 0:
            raise ValueError(f"RHS of division must be positive, got {rhs_const}")

        # Compute GCD for all of lhs and rhs_const
        gcd = math.gcd(*(abs(l) for l in lhs), rhs_const)

        # Simplify numerator and divisor by GCD
        if gcd != 1:
            lhs = [l // gcd for l in lhs]
        divisor = rhs_const // gcd

        # If divisor is 1, the division can be omitted
        if divisor == 1:
            self.operand_expr_stack.append(lhs)
            return

        # At this point, need to introduce a local variable for the division result
        # Find or create a local id for this div expression

        # Build the AffineExpr for lhs and rhs (divisor)
        a = AffineExpr.from_flat_form(
            lhs, self.num_dims, self.num_symbols, self.local_exprs
        )
        b = AffineExpr.constant(divisor)

        div_expr = a.ceil_div(b) if is_ceil else a // b

        loc = self.find_local_id(div_expr)
        if loc == -1:
            if is_ceil:
                # lhs ceildiv c <=>  (lhs + c - 1) floordiv c
                dividend = lhs.copy()
                dividend[-1] += divisor - 1  # Adjust constant term in flat form
                self.add_local_floordiv_id(dividend, divisor, div_expr)
            else:
                dividend = lhs.copy()
                self.add_local_floordiv_id(dividend, divisor, div_expr)
            loc = len(self.local_exprs) - 1  # The new local just added

        # Set the expression on stack to the local var introduced to capture the
        # result of the division (floor or ceil).
        new_row = [0] * self.get_num_cols()
        new_row[self.get_local_var_start_index() + loc] = 1
        self.operand_expr_stack.append(new_row)

    def visit_mod_expr(self, expr: AffineBinaryOpExpr) -> None:
        """
        t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1

        A mod expression "expr mod c" is thus flattened by introducing a new local
        variable q (= expr floordiv c), such that expr mod c is replaced with
        'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.

        In case of semi-affine modulo expressions, t = expr mod symbolic_expr,
        introduce a local variable m (= expr mod symbolic_expr), and the affine
        expression expr mod symbolic_expr is added to `localExprs`.
        """
        assert len(self.operand_expr_stack) >= 2
        rhs = self.operand_expr_stack.pop()
        lhs = self.operand_expr_stack.pop()

        if not isinstance(expr.rhs, AffineConstantExpr):
            # Flatten semi affine modulo expressions by introducing a local
            # variable in place of the modulo value, and the affine expression
            # corresponding to the quantifier is added to `localExprs`.
            raise NotImplementedError("Semi-affine map flattening not implemented")

        rhs_const = rhs[self.get_constant_index()]
        assert rhs_const > 0, (
            "Cannot simplify expression with negative modulo expression with factor "
            f"{rhs_const}"
        )

        # Check if the LHS expression is a multiple of modulo factor.
        if not any(l % rhs_const for l in lhs):
            # If yes, module expression here simplifies to zero
            self.operand_expr_stack.append([0 for _ in lhs])
            return

        # Add a local variable for the quotient, i.e., expr % c is replaced by
        # (expr - q * c) where q = expr floordiv c. Do this while canceling out
        # the GCD of expr and c.

        gcd = math.gcd(*(abs(l) for l in lhs), rhs_const)

        # Simplify the numerator and the denominator.
        if gcd != 1:
            floor_dividend = [fd // gcd for fd in lhs]
        else:
            floor_dividend = lhs.copy()

        floor_divisor = rhs_const // gcd

        # Construct the AffineExpr form of the floordiv to store in localExprs.

        dividend_expr = AffineExpr.from_flat_form(
            floor_dividend, self.num_dims, self.num_symbols, self.local_exprs
        )
        divisor_expr = AffineExpr.constant(floor_divisor)
        floor_div_expr = dividend_expr // divisor_expr

        if (loc := self.find_local_id(floor_div_expr)) == -1:
            self.add_local_floordiv_id(floor_dividend, floor_divisor, floor_div_expr)
            # Set result at top of stack to `lhs - rhs_const * q``
            lhs.insert(-1, -rhs_const)
        else:
            # Reuse the existing local id
            lhs[self.get_local_var_start_index() + loc] -= rhs_const
        self.operand_expr_stack.append(lhs)

    def simplify(self, expr: AffineExpr):
        for inner in expr.post_order():
            match inner:
                case AffineBinaryOpExpr():
                    match inner.kind:
                        case AffineBinaryOpKind.Mul:
                            self.visit_mul_expr(inner)
                        case AffineBinaryOpKind.Add:
                            self.visit_add_expr(inner)
                        case AffineBinaryOpKind.Mod:
                            self.visit_mod_expr(inner)
                        case AffineBinaryOpKind.FloorDiv:
                            self.visit_div_expr(inner, is_ceil=False)
                        case AffineBinaryOpKind.CeilDiv:
                            self.visit_div_expr(inner, is_ceil=True)
                case AffineDimExpr():
                    self.visit_dim_expr(inner)
                case AffineConstantExpr():
                    self.visit_constant_expr(inner)
                case AffineSymExpr():
                    self.visit_symbol_expr(inner)
                case _:
                    raise ValueError("Unreachable")

        return AffineExpr.from_flat_form(
            self.operand_expr_stack.pop(),
            self.num_dims,
            self.num_symbols,
            self.local_exprs,
        )

    def add_local_floordiv_id(
        self, dividend: list[int], divisor: int, local_expr: AffineExpr
    ) -> None:
        """
        Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
        The local identifier added is always a floordiv of a pure add/mul affine
        function of other identifiers, coefficients of which are specified in
        dividend and with respect to a positive constant divisor. local_expr is the
        simplified tree expression (AffineExpr) corresponding to the quantifier.
        """
        assert divisor > 0, "positive constant divisor expected"
        for sub_expr in self.operand_expr_stack:
            sub_expr.insert(
                self.get_local_var_start_index() + len(self.local_exprs),
                0,
            )
        self.local_exprs.append(local_expr)

    def find_local_id(self, local_expr: AffineExpr) -> int:
        """
        Returns the index of the `local_expr` in `local_exprs`, or `-1` if not found.
        """
        try:
            return self.local_exprs.index(local_expr)
        except ValueError:
            return -1

    def get_num_cols(self) -> int:
        return self.num_dims + self.num_symbols + len(self.local_exprs) + 1

    def get_constant_index(self) -> int:
        return self.get_num_cols() - 1

    def get_local_var_start_index(self) -> int:
        return self.num_dims + self.num_symbols

    def get_symbol_start_index(self) -> int:
        return self.num_dims

    def get_dim_start_index(self) -> int:
        return 0
