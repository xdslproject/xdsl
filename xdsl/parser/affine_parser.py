from __future__ import annotations

from collections.abc import Callable, Sequence

from xdsl.ir.affine import (
    AffineConstraintExpr,
    AffineConstraintKind,
    AffineExpr,
    AffineMap,
    AffineSet,
)
from xdsl.utils.exceptions import ParseError
from xdsl.utils.mlir_lexer import MLIRToken, MLIRTokenKind

from .base_parser import BaseParser  # noqa: TID251
from .generic_parser import ParserState  # noqa: TID251


class AffineParser(BaseParser):
    _BINOP_PRECEDENCE = {
        "+": 10,
        "-": 10,
        "*": 20,
        "ceildiv": 20,
        "floordiv": 20,
        "mod": 20,
    }

    def __init__(self, state: ParserState[MLIRTokenKind]) -> None:
        self._resume_from(state)

    def _get_parse_optional_bare_id(
        self, dims: list[str], syms: list[str]
    ) -> Callable[[], AffineExpr | None]:
        def parse_optional_bare_id():
            if (token := self._parse_optional_token(MLIRTokenKind.BARE_IDENT)) is None:
                return
            # Handle bare id
            span = token.span
            bare_id = span.text
            if bare_id in dims:
                return AffineExpr.dimension(dims.index(bare_id))
            elif bare_id in syms:
                return AffineExpr.symbol(syms.index(bare_id))
            else:
                raise ParseError(span, f"Identifier not in space {bare_id}")

        return parse_optional_bare_id

    def _parse_primary(
        self, parse_optional_bare_id: Callable[[], AffineExpr | None]
    ) -> AffineExpr:
        """
        primary ::= `(` affine-expr `)`
                  | bare-id
                  | integer-literal
                  | `-` primary
        """
        if (bare_id := parse_optional_bare_id()) is not None:
            return bare_id
        current_token = self._consume_token()
        match current_token.kind:
            case MLIRTokenKind.L_PAREN:
                # Handle parentheses
                expr = self._parse_affine_expr(parse_optional_bare_id)
                self._parse_token(MLIRTokenKind.R_PAREN, "Expected closing parenthesis")
                return expr
            case MLIRTokenKind.INTEGER_LIT:
                # Handle integer literal
                return AffineExpr.constant(
                    current_token.kind.get_int_value(current_token.span)
                )
            case MLIRTokenKind.MINUS:
                # Handle negative primary
                return -self._parse_primary(parse_optional_bare_id)
            case _:
                raise ParseError(current_token.span, "Expected primary expression")

    def _get_token_precedence(self) -> int:
        return self._BINOP_PRECEDENCE.get(self._current_token.text, -1)

    def _create_binop_expr(
        self, lhs: AffineExpr, rhs: AffineExpr, binop: MLIRToken
    ) -> AffineExpr:
        match binop.text:
            case "+":
                return lhs + rhs
            case "-":
                return lhs - rhs
            case "*":
                return lhs * rhs
            case "ceildiv":
                return lhs.ceil_div(rhs)
            case "floordiv":
                return lhs // rhs
            case "mod":
                return lhs % rhs
            case _:
                raise ParseError(binop.span, f"Unknown binary operator {binop.text}")

    def _parse_binop_rhs(
        self,
        lhs: AffineExpr,
        prec: int,
        parse_optional_bare_id: Callable[[], AffineExpr | None],
    ) -> AffineExpr:
        while True:
            tok_prec = self._get_token_precedence()
            # This works even if the token does not exist, since -1 is returned.
            if tok_prec < prec:
                return lhs
            # Get the binop
            binop = self._consume_token()
            # Parse the primary expression after the binary operator.
            rhs = self._parse_primary(parse_optional_bare_id)
            next_prec = self._get_token_precedence()
            if tok_prec < next_prec:
                # Increase the precision of the current operator to parse
                # it before the next one in case they have same precedence.
                rhs = self._parse_binop_rhs(rhs, tok_prec + 1, parse_optional_bare_id)
            lhs = self._create_binop_expr(lhs, rhs, binop)

    # TODO: Extend to semi-affine maps
    def _parse_affine_expr(
        self, parse_optional_bare_id: Callable[[], AffineExpr | None]
    ) -> AffineExpr:
        """
        affine-expr ::= `(` affine-expr `)`
                      | `-`? integer-literal
                      | bare-id
                      | `-`affine-expr
                      | `-`? integer-literal `*` affine-expr
                      | affine-expr `ceildiv` integer-literal
                      | affine-expr `floordiv` integer-literal
                      | affine-expr `mod` integer-literal
                      | affine-expr `+` affine-expr
                      | affine-expr `-` affine-expr
        """
        lhs = self._parse_primary(parse_optional_bare_id)
        return self._parse_binop_rhs(lhs, 0, parse_optional_bare_id)

    def _parse_multi_affine_expr(
        self, parse_optional_bare_id: Callable[[], AffineExpr | None]
    ) -> list[AffineExpr]:
        """
        multi-affine-expr ::= `(` `)`
                                | `(` affine-expr (`,` affine-expr)* `)`
        """

        def parse_expr() -> AffineExpr:
            return self._parse_affine_expr(parse_optional_bare_id)

        return self.parse_comma_separated_list(self.Delimiter.PAREN, parse_expr)

    # TODO: Extend to semi-affine maps; see https://github.com/xdslproject/xdsl/issues/1087
    def _parse_affine_constraint(
        self, parse_optional_bare_id: Callable[[], AffineExpr | None]
    ) -> AffineConstraintExpr:
        """
        affine-expr ::= `(` affine-expr `)`
                      | `-`? integer-literal
                      | bare-id
                      | `-`affine-expr
                      | `-`? integer-literal `*` affine-expr
                      | affine-expr `ceildiv` integer-literal
                      | affine-expr `floordiv` integer-literal
                      | affine-expr `mod` integer-literal
                      | affine-expr `+` affine-expr
                      | affine-expr `-` affine-expr
        """
        lhs = self._parse_affine_expr(parse_optional_bare_id)
        op = self._consume_token().text + self._consume_token().text
        if op not in set(k.value for k in AffineConstraintKind):
            self.raise_error(
                f"Expected one of {', '.join(f'`{k.value}`' for k in AffineConstraintKind)}, got {op}."
            )
        op = AffineConstraintKind(op)
        rhs = self._parse_affine_expr(parse_optional_bare_id)

        return AffineConstraintExpr(op, lhs, rhs)

    def _parse_multi_affine_constaint(
        self, parse_optional_bare_id: Callable[[], AffineExpr | None]
    ) -> list[AffineConstraintExpr]:
        """
        multi-affine-expr ::= `(` `)`
                                | `(` affine-expr (`,` affine-expr)* `)`
        """

        def parse_constraint() -> AffineConstraintExpr:
            return self._parse_affine_constraint(parse_optional_bare_id)

        return self.parse_comma_separated_list(self.Delimiter.PAREN, parse_constraint)

    def _parse_affine_space(self) -> tuple[list[str], list[str]]:
        """
        dims ::= `(` ssa-use-list? `)`
        syms ::= `[` ssa-use-list? `]`
        affine-space ::= dims syms?
        """

        def parse_id() -> str:
            return self._parse_token(
                MLIRTokenKind.BARE_IDENT, "Expected identifier"
            ).text

        # Parse dimensions
        dims = self.parse_comma_separated_list(self.Delimiter.PAREN, parse_id)
        # Parse optional symbols
        if self._current_token.kind != MLIRTokenKind.L_SQUARE:
            syms = []
        else:
            syms = self.parse_comma_separated_list(self.Delimiter.SQUARE, parse_id)
        # TODO: Do not allow duplicate ids.
        return dims, syms

    def parse_affine_map(self) -> AffineMap:
        """
        affine-map
           ::= affine-space `->` multi-affine-expr
        """
        # Parse affine space
        dims, syms = self._parse_affine_space()
        # Parse : delimiter
        self._parse_token(MLIRTokenKind.ARROW, "Expected `->`")
        # Parse list of affine expressions
        exprs = self._parse_multi_affine_expr(
            self._get_parse_optional_bare_id(dims, syms)
        )
        # Create map and return.
        return AffineMap(len(dims), len(syms), tuple(exprs))

    def parse_affine_set(self) -> AffineSet:
        """
        affine-map
           ::= affine-space `:` `(` (affine-constraint)* `)`
        """
        # Parse affine space
        dims, syms = self._parse_affine_space()
        # Parse : delimiter
        self._parse_token(MLIRTokenKind.COLON, "Expected `:`")
        # Parse list of affine expressions
        constraints = self._parse_multi_affine_constaint(
            self._get_parse_optional_bare_id(dims, syms)
        )
        # Create map and return.
        return AffineSet(len(dims), len(syms), tuple(constraints))

    def _get_parse_optional_ssa_value(
        self,
    ) -> tuple[Callable[[], AffineExpr | None], dict[str, int]]:
        """
        Returns function to parse an affine symbol expr represented by an SSA value
        identifier, and the dictionary mapping SSA value name to the corresponding
        symbol index, populated by the function as it encounters new values.
        """
        symbol_by_ssa_name: dict[str, int] = {}

        def parse_optional_ssa_value() -> AffineExpr | None:
            if (
                ident_token := self._parse_optional_token(MLIRTokenKind.PERCENT_IDENT)
            ) is not None:
                ident = ident_token.span.text
                try:
                    symbol = symbol_by_ssa_name[ident]
                except KeyError:
                    symbol = len(symbol_by_ssa_name)
                    symbol_by_ssa_name[ident] = symbol
                return AffineExpr.symbol(symbol)

        return parse_optional_ssa_value, symbol_by_ssa_name

    def parse_affine_map_of_ssa_ids(self) -> tuple[AffineMap, Sequence[str]]:
        """
        Parse an affine map where ssa values can be used inside the expressions.
        ```
        `[` affine-expr (`,` affine-expr)* `]`
        ```
        """
        parse_optional_bare_id, symbol_by_ssa_name = (
            self._get_parse_optional_ssa_value()
        )
        exprs = self.parse_comma_separated_list(
            self.Delimiter.SQUARE,
            lambda: self._parse_affine_expr(parse_optional_bare_id),
        )
        syms = tuple(symbol_by_ssa_name)
        return AffineMap(0, len(syms), tuple(exprs)), syms
