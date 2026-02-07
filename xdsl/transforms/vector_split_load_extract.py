from dataclasses import dataclass
from typing import cast

from xdsl.context import Context
from xdsl.dialects import arith, ptr, vector
from xdsl.dialects.builtin import (
    CompileTimeFixedBitwidthType,
    IndexType,
    IntegerAttr,
    ModuleOp,
    VectorType,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


class VectorSplitLoadExtract(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ptr.LoadOp, rewriter: PatternRewriter) -> None:
        if not all(
            isinstance(use.operation, vector.ExtractOp)
            and not use.operation.dynamic_position
            for use in op.res.uses
        ):
            return

        vector_type = cast(VectorType, op.res.type)
        element_type = vector_type.element_type
        if not isinstance(element_type, CompileTimeFixedBitwidthType):
            raise NotImplementedError

        if len(vector_type.shape) != 1:
            raise NotImplementedError

        element_size = element_type.compile_time_size
        name_hint = op.res.name_hint

        for use in op.res.uses:
            user = cast(vector.ExtractOp, use.operation)
            indices = user.static_position.get_values()
            assert len(indices) == 1
            (element_index,) = indices
            rewriter.insert_op(
                (
                    constant_op := arith.ConstantOp(
                        IntegerAttr(element_size * element_index, IndexType())
                    ),
                    add_op := ptr.PtrAddOp(op.addr, constant_op.result),
                    load_op := ptr.LoadOp(add_op.result, element_type),
                )
            )
            rewriter.replace_all_uses_with(user.result, load_op.res)
            rewriter.erase_op(user)
            constant_op.result.name_hint = name_hint
            add_op.result.name_hint = name_hint
            load_op.res.name_hint = name_hint

        rewriter.erase_op(op)


@dataclass(frozen=True)
class VectorSplitLoadExtractPass(ModulePass):
    """
    Rewrites a vector load followed only by extracts with scalar loads.
    """

    name = "vector-split-load-extract"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            VectorSplitLoadExtract(),
            apply_recursively=False,
        ).rewrite_module(op)
