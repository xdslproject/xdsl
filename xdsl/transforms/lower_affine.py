from collections.abc import Sequence

from xdsl.builder import InsertPoint
from xdsl.context import MLContext
from xdsl.dialects import affine, arith, builtin, memref, scf
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
    AffineConstraintKind,
    AffineDimExpr,
    AffineSymExpr,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


def affine_expr_ops(
    expr: affine.AffineExpr,
    dims: Sequence[SSAValue],
    symbols: Sequence[SSAValue],
) -> tuple[list[Operation], SSAValue]:
    """
    Returns the operations that evaluate the affine expression when given input SSA
    values, along with the SSAValue representing the result.
    """
    if isinstance(expr, AffineConstantExpr):
        constant = arith.Constant(builtin.IntegerAttr.from_index_int_value(expr.value))
        return [constant], constant.result

    if isinstance(expr, AffineDimExpr):
        return [], dims[expr.position]
    if isinstance(expr, AffineSymExpr):
        return [], symbols[expr.position]

    if isinstance(expr, AffineBinaryOpExpr):
        lhs_ops, lhs_val = affine_expr_ops(expr.lhs, dims, symbols)
        rhs_ops, rhs_val = affine_expr_ops(expr.rhs, dims, symbols)

        match expr.kind:
            case AffineBinaryOpKind.Add:
                op = arith.Addi(lhs_val, rhs_val)
            case AffineBinaryOpKind.Mul:
                op = arith.Muli(lhs_val, rhs_val)
            case AffineBinaryOpKind.Mod:
                op = arith.RemSI(lhs_val, rhs_val)
            case AffineBinaryOpKind.FloorDiv:
                op = arith.FloorDivSI(lhs_val, rhs_val)
            case AffineBinaryOpKind.CeilDiv:
                op = arith.CeilDivSI(lhs_val, rhs_val)

        return [*lhs_ops, *rhs_ops, op], op.result

    assert False, "Unreachable"


def insert_affine_map_ops(
    map: affine.AffineMapAttr | None,
    dims: Sequence[SSAValue],
    symbols: list[SSAValue],
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Returns operations that evaluate the affine map when given input SSA values and the
    resulting indices.
    """
    ops: list[Operation] = []
    if map is None:
        indices = list(dims)
    else:
        indices: list[SSAValue] = []
        for expr in map.data.results:
            new_ops, val = affine_expr_ops(expr, dims, [])
            ops.extend(new_ops)
            indices.append(val)

    return ops, indices


class LowerAffineStore(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.Store, rewriter: PatternRewriter):
        ops, indices = insert_affine_map_ops(op.map, op.indices, [])
        rewriter.insert_op_before_matched_op(ops)

        # TODO: add nontemporal=false once that's added to memref
        # https://github.com/xdslproject/xdsl/issues/1482
        rewriter.replace_matched_op(memref.Store.get(op.value, op.memref, indices))


class LowerAffineLoad(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.Load, rewriter: PatternRewriter):
        ops, indices = insert_affine_map_ops(op.map, op.indices, [])
        rewriter.insert_op_before_matched_op(ops)

        # TODO: add nontemporal=false once that's added to memref
        # https://github.com/xdslproject/xdsl/issues/1482
        rewriter.replace_matched_op(memref.Load.get(op.memref, indices))


class LowerAffineFor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.For, rewriter: PatternRewriter):
        lb_map = op.lowerBoundMap.data
        ub_map = op.upperBoundMap.data
        assert len(lb_map.results) == 1, "Affine for lower_bound must have one result"
        assert len(ub_map.results) == 1, "Affine for upper_bound must have one result"
        lb_ops, lb_val = affine_expr_ops(lb_map.results[0], [], [])
        rewriter.insert_op_before_matched_op(lb_ops)
        ub_ops, ub_val = affine_expr_ops(ub_map.results[0], [], [])
        rewriter.insert_op_before_matched_op(ub_ops)
        step_op = arith.Constant(op.step)
        rewriter.insert_op_before_matched_op(step_op)
        rewriter.replace_matched_op(
            scf.For(
                lb_val,
                ub_val,
                step_op.result,
                op.inits,
                rewriter.move_region_contents_to_new_regions(op.body),
            )
        )


class LowerAffineYield(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.Yield, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(scf.Yield(*op.arguments))


class LowerAffineApply(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.ApplyOp, rewriter: PatternRewriter, /):
        affine_map = op.map.data
        assert len(affine_map.results) == 1

        operands = op.mapOperands
        assert affine_map.num_dims + affine_map.num_symbols == len(operands)

        dims = operands[: affine_map.num_dims]
        symbols = operands[affine_map.num_dims :]

        new_ops: list[Operation] = []
        new_results: list[SSAValue] = []

        ops, val = affine_expr_ops(affine_map.results[0], dims, symbols)
        new_ops.extend(ops)
        new_results.append(val)
        rewriter.replace_matched_op(new_ops, new_results)


class LowerAffineIf(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.If, rewriter: PatternRewriter):
        # Now we just have to handle the condition logic.

        # Calculate cond as conjunction without short-circuiting.
        cond = None

        zero_constant = arith.Constant(arith.IntegerAttr.from_index_int_value(0))
        rewriter.insert_op(zero_constant, InsertPoint.before(op))

        for constraint in op.condition.data.constraints:
            is_equality = constraint.kind == AffineConstraintKind.eq

            # Build and apply an affine expression
            expr = constraint.lhs
            ops, val = affine_expr_ops(expr, list(op.operands), [])

            if not val:
                return
            rewriter.insert_op(ops, InsertPoint.before(op))

            if is_equality:
                cmp_val = arith.Cmpi(val, zero_constant, "eq")
            else:
                cmp_val = arith.Cmpi(val, zero_constant, "sge")

            if cond:
                and_cond = arith.AndI(cond, cmp_val)
                rewriter.insert_op(and_cond, InsertPoint.before(op))
                cond = and_cond.result
            else:
                cond = cmp_val
                rewriter.insert_op(cond, InsertPoint.before(op))

        if not cond:
            one_constant = arith.Constant(arith.IntegerAttr.from_int_and_width(1, 1))
            rewriter.insert_op(one_constant, InsertPoint.before(op))

        has_else_region = len(op.else_region.blocks) > 0
        result_types = [r.type for r in op.res]

        assert isinstance(cond, arith.Cmpi | arith.AndI)
        if_op = scf.If(cond.result, result_types, Region(Block()), Region(Block()))

        rewriter.inline_region_before(op.then_region, if_op.true_region.blocks[-1])
        empty_block = if_op.true_region.detach_block(if_op.true_region.blocks[-1])
        empty_block.erase()

        true_terminator = list(
            filter(
                lambda x: isinstance(x, affine.Yield), list(if_op.true_region.walk())
            )
        )[0]
        rewriter.replace_op(true_terminator, scf.Yield())

        if has_else_region:
            rewriter.inline_region_before(op.else_region, if_op.false_region.blocks[-1])
            false_terminator = list(
                filter(
                    lambda x: isinstance(x, affine.Yield),
                    list(if_op.false_region.walk()),
                )
            )[0]
            rewriter.replace_op(false_terminator, scf.Yield())

        empty_block = if_op.false_region.detach_block(if_op.false_region.blocks[-1])
        empty_block.erase()

        rewriter.replace_matched_op(if_op)


class LowerAffinePass(ModulePass):
    name = "lower-affine"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAffineStore(),
                    LowerAffineLoad(),
                    LowerAffineFor(),
                    LowerAffineYield(),
                    LowerAffineApply(),
                    LowerAffineIf(),
                ]
            )
        ).rewrite_module(op)
