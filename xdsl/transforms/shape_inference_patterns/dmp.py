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
    Infer the shape of the `dmp.swap` operation.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewrite: PatternRewriter):
        if not op.swapped_values:
            return
        swap_t = op.swapped_values.type
        if op.input_stencil.type != swap_t:
            rewrite.replace_value_with_new_type(op.input_stencil, swap_t)
            rewrite.handle_operation_modification(op)


class DmpSwapSwapsInference(RewritePattern):
    """
    Infer the exact exchanges this `dmp.swap` needs to perform.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewrite: PatternRewriter):
        core_lb: stencil.IndexAttr | None = None
        core_ub: stencil.IndexAttr | None = None

        if not op.swapped_values:
            return

        for use in op.swapped_values.uses:
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

        # if buff_* has fewer dimensions than core_*, we need to find out which dimensions of core_*
        # those in buff_* map to. This information only exists in the offset_mapping of a stencil.access
        if len(buff_lb) < len(core_lb):
            for use in op.swapped_values.uses:
                if not isinstance(apply_op := use.operation, stencil.ApplyOp):
                    continue
                arg_idx = apply_op.operands.index(op.swapped_values)
                arg = apply_op.region.block.args[arg_idx]
                for arg_use in arg.uses:
                    if (
                        not isinstance(access_op := arg_use.operation, stencil.AccessOp)
                        or not access_op.offset_mapping
                    ):
                        continue
                    accessed_dims = tuple(access_op.offset_mapping)

                    # reconstruct what dims buff_* maps to
                    new_lb = [0] * len(core_lb)
                    new_ub = [0] * len(core_ub)
                    for idx, dim in enumerate(accessed_dims):
                        new_lb[dim] = buff_lb.array.data[idx].data
                        new_ub[dim] = buff_ub.array.data[idx].data
                    buff_lb = stencil.IndexAttr.get(*new_lb)
                    buff_ub = stencil.IndexAttr.get(*new_ub)

                    # todo breaking here means we do not verify that all accesses map to the same dimension
                    break

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
            # drop empty exchanges
            if exchange.elem_count > 0
        )
        if swaps != op.swaps:
            op.swaps = swaps
            rewrite.handle_operation_modification(op)
