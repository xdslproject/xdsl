from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import linalg
from xdsl.dialects.builtin import DenseArrayBase, IntegerType, ModuleOp
from xdsl.dialects.linalg.transforms.tiling import tile_linalg_generic
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class TileLinalgGenericFromAttributePattern(RewritePattern):
    """
    Rewrite supported `linalg.generic` ops annotated with `test_tile_sizes` into tiled form.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.ops.GenericOp, rewriter: PatternRewriter, /
    ) -> None:

        tile_sizes_attr = op.attributes.get("test_tile_sizes")
        if tile_sizes_attr is None:
            return

        assert isa(tile_sizes_attr, DenseArrayBase[IntegerType])
        tile_sizes = tuple(tile_sizes_attr.get_values())
        tile_linalg_generic(rewriter, op, tile_sizes)


@dataclass(frozen=True)
class TestLinalgTilingPass(ModulePass):
    """
    Tile supported memref-based `linalg.generic` ops annotated with `test_tile_sizes`.
    """

    name = "test-linalg-tiling"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            TileLinalgGenericFromAttributePattern(),
            apply_recursively=False,
        ).rewrite_module(op)
