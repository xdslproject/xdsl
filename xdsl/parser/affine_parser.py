from __future__ import annotations

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

    def _parse_primary(self, dims: list[str], syms: list[str]) -> AffineExpr:
        """
        primary ::= `(` affine-expr `)`
                  | bare-id
                  | integer-literal
                  | `-` primary
        """
        # Handle parentheses
        if self._parse_optional_token(MLIRTokenKind.L_PAREN):
            expr = self._parse_affine_expr(dims, syms)
            self._parse_token(MLIRTokenKind.R_PAREN, "Expected closing parenthesis")
            return expr
        # Handle bare id
        if bare_id := self._parse_optional_token(MLIRTokenKind.BARE_IDENT):
            if bare_id.text in dims:
                return AffineExpr.dimension(dims.index(bare_id.text))
            elif bare_id.text in syms:
                return AffineExpr.symbol(syms.index(bare_id.text))
            else:
                raise ParseError(
                    bare_id.span, f"Identifier not in space {bare_id.text}"
                )
        # Handle integer literal
        if int_lit := self._parse_optional_token(MLIRTokenKind.INTEGER_LIT):
            return AffineExpr.constant(int_lit.kind.get_int_value(int_lit.span))
        # Handle negative primary
        if self._parse_optional_token(MLIRTokenKind.MINUS):
            return -self._parse_primary(dims, syms)

        raise ParseError(self._current_token.span, "Expected primary expression")

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
        self, lhs: AffineExpr, prec: int, dims: list[str], syms: list[str]
    ) -> AffineExpr:
        while True:
            tok_prec = self._get_token_precedence()
            # This works even if the token does not exist, since -1 is returned.
            if tok_prec < prec:
                return lhs
            # Get the binop
            binop = self._consume_token()
            # Parse the primary expression after the binary operator.
            rhs = self._parse_primary(dims, syms)
            next_prec = self._get_token_precedence()
            if tok_prec < next_prec:
                # Increase the precision of the current operator to parse
                # it before the next one in case they have same precedence.
                rhs = self._parse_binop_rhs(rhs, tok_prec + 1, dims, syms)
            lhs = self._create_binop_expr(lhs, rhs, binop)

    # TODO: Extend to semi-affine maps
    def _parse_affine_expr(self, dims: list[str], syms: list[str]) -> AffineExpr:
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
        lhs = self._parse_primary(dims, syms)
        return self._parse_binop_rhs(lhs, 0, dims, syms)

    def _parse_multi_affine_expr(
        self, dims: list[str], syms: list[str]
    ) -> list[AffineExpr]:
        """
        multi-affine-expr ::= `(` `)`
                                | `(` affine-expr (`,` affine-expr)* `)`
        """

        def parse_expr() -> AffineExpr:
            return self._parse_affine_expr(dims, syms)

        return self.parse_comma_separated_list(self.Delimiter.PAREN, parse_expr)

    # TODO: Extend to semi-affine maps; see https://github.com/xdslproject/xdsl/issues/1087
    def _parse_affine_constraint(
        self, dims: list[str], syms: list[str]
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
        lhs = self._parse_affine_expr(dims, syms)
        op = self._consume_token().text + self._consume_token().text
        if op not in set(k.value for k in AffineConstraintKind):
            self.raise_error(
                f"Expected one of {', '.join(f'`{k.value}`' for k in AffineConstraintKind)}, got {op}."
            )
        op = AffineConstraintKind(op)
        rhs = self._parse_affine_expr(dims, syms)

        return AffineConstraintExpr(op, lhs, rhs)

    def _parse_multi_affine_constaint(
        self, dims: list[str], syms: list[str]
    ) -> list[AffineConstraintExpr]:
        """
        multi-affine-expr ::= `(` `)`
                                | `(` affine-expr (`,` affine-expr)* `)`
        """

        def parse_constraint() -> AffineConstraintExpr:
            return self._parse_affine_constraint(dims, syms)

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
        exprs = self._parse_multi_affine_expr(dims, syms)
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
        constraints = self._parse_multi_affine_constaint(dims, syms)
        # Create map and return.
        return AffineSet(len(dims), len(syms), tuple(constraints))
