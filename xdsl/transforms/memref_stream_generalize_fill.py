from dataclasses import dataclass
from typing import cast

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import memref, memref_stream
from xdsl.dialects.builtin import (
    AffineMapAttr,
    ArrayAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
)
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class GeneralizeFillPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.FillOp, rewriter: PatternRewriter
    ) -> None:
        block = Block(arg_types=(op.value.type, op.value.type))

        with ImplicitBuilder(block) as (arg0, _):
            memref_stream.YieldOp(arg0)

        assert isinstance(memref_type := op.memref.type, memref.MemRefType)

        memref_type = cast(MemRefType, memref_type)

        shape = memref_type.get_shape()
        index = IndexType()
        ubs = ArrayAttr(IntegerAttr(ub, index) for ub in shape)

        rewriter.replace_op(
            op,
            memref_stream.GenericOp(
                (op.value,),
                (op.memref,),
                (),
                Region((block,)),
                ArrayAttr(
                    (
                        AffineMapAttr(AffineMap(len(shape), 0, ())),
                        AffineMapAttr(AffineMap.identity(len(shape))),
                    )
                ),
                ArrayAttr((memref_stream.IteratorTypeAttr.parallel(),) * len(shape)),
                ubs,
                ArrayAttr(()),
            ),
        )


@dataclass(frozen=True)
class MemRefStreamGeneralizeFillPass(ModulePass):
    """
    Generalizes memref_stream.fill ops.
    """

    name = "memref-stream-generalize-fill"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GeneralizeFillPattern(),
            apply_recursively=False,
        ).rewrite_module(op)
