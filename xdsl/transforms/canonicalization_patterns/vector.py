from typing import cast

from xdsl.dialects import arith, ptr, vector
from xdsl.dialects.builtin import (
    CompileTimeFixedBitwidthType,
    IndexType,
    IntegerAttr,
    VectorType,
)
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)


class VectorExtractToLoad(RewritePattern):
    # WIP WIP WIP proper pass
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: vector.ExtractOp, rewriter: PatternRewriter):
        if not isinstance(load_op := op.vector.owner, ptr.LoadOp):
            return

        if op.dynamic_position:
            raise NotImplementedError

        if len(op.static_position) != 1:
            raise NotImplementedError

        vector_type = cast(VectorType, op.vector.type)
        if not isinstance(vector_type.element_type, CompileTimeFixedBitwidthType):
            raise NotImplementedError

        (index,) = op.static_position.get_values()
        bytes_per_element = vector_type.element_type.compile_time_size
        offset_op = arith.ConstantOp(
            IntegerAttr(index * bytes_per_element, IndexType())
        )

        ptr_add_op = ptr.PtrAddOp(load_op.addr, offset_op.result)
        load_op = ptr.LoadOp(ptr_add_op.result, op.result.type)

        rewriter.replace_matched_op((offset_op, ptr_add_op, load_op))
