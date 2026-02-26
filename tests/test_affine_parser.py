import pytest

from xdsl.ir.affine import AffineMap
from xdsl.parser import AffineParser, ParserState
from xdsl.utils.lexer import Input
from xdsl.utils.mlir_lexer import MLIRLexer


@pytest.mark.parametrize(
    "text, expected_map, expected_syms",
    [
        # Only constants
        ("[3, 7]", AffineMap.from_callable(lambda: (3, 7)), ()),
        # Just SSA dims
        (
            "[%i0, %i1]",
            AffineMap.from_callable(lambda s0, s1: (s0, s1), dim_symbol_split=(0, 2)),
            ("%i0", "%i1"),
        ),
        # SSA dim & constant
        (
            "[%i0, 7]",
            AffineMap.from_callable(lambda s0: (s0, 7), dim_symbol_split=(0, 1)),
            ("%i0",),
        ),
        # SSA dims + const sums
        (
            "[%i0 + 3, %i1 + 7]",
            AffineMap.from_callable(
                lambda s0, s1: (s0 + 3, s1 + 7), dim_symbol_split=(0, 2)
            ),
            ("%i0", "%i1"),
        ),
        # SSA dim and dim + const
        (
            "[%i0, %i1 + 7]",
            AffineMap.from_callable(
                lambda s0, s1: (s0, s1 + 7), dim_symbol_split=(0, 2)
            ),
            ("%i0", "%i1"),
        ),
        # constant + d1, d1 + const
        (
            "[3 + %i1, %i1 + 7]",
            AffineMap.from_callable(
                lambda s0: (3 + s0, s0 + 7), dim_symbol_split=(0, 1)
            ),
            ("%i1",),
        ),
        # Mix of constant + dim0 * const + dim1
        (
            "[3 + %i0 * 7 + %i1, 7]",
            AffineMap.from_callable(
                lambda s0, s1: (3 + s0 * 7 + s1, 7), dim_symbol_split=(0, 2)
            ),
            ("%i0", "%i1"),
        ),
    ],
)
def test_parse_affine_map_of_ssa_ids(
    text: str, expected_map: AffineMap, expected_syms: list[str]
):
    parser = AffineParser(ParserState(MLIRLexer(Input(text, ""))))
    amap, syms = parser.parse_affine_map_of_ssa_ids()

    assert amap == expected_map
    assert syms == expected_syms
