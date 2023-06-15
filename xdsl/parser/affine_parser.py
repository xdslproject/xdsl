from __future__ import annotations

from .core import Parser, ParserState, Token
from xdsl.ir.affine import AffineExpr, AffineMap


class AffineParser(Parser):
    def __init__(self, state: ParserState) -> None:
        self.resume_from_state(state)

    # TODO: Extend to semi-affine maps
    def _parse_affine_expr(self, dims: list[str], syms: list[str]) -> AffineExpr:
        """
        affine-expr ::= `(` affine-expr `)`
                      | affine-expr `+` affine-expr
                      | affine-expr `-` affine-expr
                      | `-`? integer-literal `*` affine-expr
                      | affine-expr `ceildiv` integer-literal
                      | affine-expr `floordiv` integer-literal
                      | affine-expr `mod` integer-literal
                      | `-`affine-expr
                      | bare-id
                      | `-`? integer-literal
        """
        # FIXME: Ignore all tokens for testing
        while (
            self._current_token.kind != Token.Kind.COMMA
            and self._current_token.kind != Token.Kind.R_PAREN
        ):
            self._consume_token()
        return AffineExpr.constant(0)

    def _parse_multi_affine_expr(
        self, dims: list[str], syms: list[str]
    ) -> list[AffineExpr]:
        """
        multi-dim-affine-expr ::= `(` `)`
                                | `(` affine-expr (`,` affine-expr)* `)`
        """

        def parse_expr() -> AffineExpr:
            return self._parse_affine_expr(dims, syms)

        return self.parse_comma_separated_list(self.Delimiter.PAREN, parse_expr)

    def _parse_affine_space(self) -> tuple[list[str], list[str]]:
        """
        dims ::= `(` ssa-use-list? `)`
        syms ::= `[` ssa-use-list? `]`
        affine-space ::= dims syms?
        """

        def parse_id() -> str:
            return self._consume_token(Token.Kind.BARE_IDENT).text

        # Parse dimensions
        dims = self.parse_comma_separated_list(self.Delimiter.PAREN, parse_id)
        # Parse optional symbols
        if self._current_token.kind != Token.Kind.L_SQUARE:
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
        self._consume_token(Token.Kind.ARROW)
        # Parse list of affine expressions
        exprs = self._parse_multi_affine_expr(dims, syms)
        # Create map and return.
        return AffineMap(len(dims), len(syms), exprs)
