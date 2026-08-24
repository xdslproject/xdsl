from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import linalg, test
from xdsl.dialects.builtin import (
    DenseArrayBase,
    IndexType,
    IntegerType,
    ModuleOp,
)
from xdsl.dialects.linalg.transforms.tiling import tile_structured_op
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa


class TileLinalgFromAttributePattern(RewritePattern):
    """
    Rewrite supported structured linalg ops annotated with `test_tile_sizes` into
    tiled form.

    A tile size is normally taken straight from that attribute. Dimensions named
    by `test_dynamic_tile_sizes` instead take a tile size that is not known until
    the op runs, which the pass has no way of writing in an attribute, so one is
    produced by a `test.op` for the tiling to consume.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self,
        op: linalg.abstract_ops.LinalgStructuredOperation,
        rewriter: PatternRewriter,
        /,
    ) -> None:

        tile_sizes_attr = op.attributes.get("test_tile_sizes")
        if tile_sizes_attr is None:
            return

        assert isa(tile_sizes_attr, DenseArrayBase[IntegerType])
        tile_sizes: list[SSAValue | int] = list(tile_sizes_attr.get_values())

        dynamic_dims_attr = op.attributes.get("test_dynamic_tile_sizes")
        if dynamic_dims_attr is not None:
            assert isa(dynamic_dims_attr, DenseArrayBase[IntegerType])
            for dim in dynamic_dims_attr.get_values():
                tile_size_op = test.TestOp(result_types=[IndexType()])
                rewriter.insert(tile_size_op, InsertPoint.before(op))
                tile_sizes[dim] = tile_size_op.res[0]

        tile_structured_op(rewriter, op, tile_sizes)


@dataclass(frozen=True)
class TestLinalgTilingPass(ModulePass):
    """
    Tile supported structured linalg ops annotated with `test_tile_sizes`.
    """

    name = "test-linalg-tiling"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            TileLinalgFromAttributePattern(),
            apply_recursively=False,
        ).rewrite_module(op)
