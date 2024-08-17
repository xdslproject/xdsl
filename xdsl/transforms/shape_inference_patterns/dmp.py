from xdsl.dialects import builtin, stencil
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute
from xdsl.pattern_rewriter import (
    PatternRewriter,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.utils.hints import isa


class DmpSwapShapeInference(RewritePattern):
    """
    Infer the exact exchanges this `dmp.swap` needs to perform.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewrite: PatternRewriter):
        core_lb: stencil.IndexAttr | None = None
        core_ub: stencil.IndexAttr | None = None

        for use in op.input_stencil.uses:
            if not isinstance(use.operation, stencil.ApplyOp):
                continue
            assert use.operation.res
            bounds = use.operation.get_bounds()
            if isinstance(bounds, builtin.IntAttr):
                return
            core_lb = bounds.lb
            core_ub = bounds.ub
            break

        # this shouldn't have changed since the op was created!
        temp = op.input_stencil.type
        assert isa(temp, stencil.TempType[Attribute])
        if not isinstance(temp.bounds, stencil.StencilBoundsAttr):
            return
        buff_lb = temp.bounds.lb
        buff_ub = temp.bounds.ub

        if core_lb is None or core_ub is None:
            return

        # drop 0 element exchanges
        swaps = builtin.ArrayAttr(
            exchange
            for exchange in op.strategy.halo_exchange_defs(
                dmp.ShapeAttr.from_index_attrs(
                    buff_lb=buff_lb,
                    core_lb=core_lb,
                    buff_ub=buff_ub,
                    core_ub=core_ub,
                )
            )
            if exchange.elem_count > 0
        )
        if swaps != op.swaps:
            op.swaps = swaps
            rewrite.handle_operation_modification(op)
