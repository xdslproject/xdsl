from collections.abc import Sequence

from xdsl.context import Context
from xdsl.dialects import affine, arith, builtin, memref, scf
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
    AffineConstantExpr,
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
    match expr:
        case AffineConstantExpr():
            constant = arith.ConstantOp(
                builtin.IntegerAttr.from_index_int_value(expr.value)
            )
            return [constant], constant.result
        case AffineDimExpr():
            return [], dims[expr.position]
        case AffineSymExpr():
            return [], symbols[expr.position]
        case AffineBinaryOpExpr():
            lhs_ops, lhs_val = affine_expr_ops(expr.lhs, dims, symbols)
            rhs_ops, rhs_val = affine_expr_ops(expr.rhs, dims, symbols)

            match expr.kind:
                case AffineBinaryOpKind.Add:
                    op = arith.AddiOp(lhs_val, rhs_val)
                case AffineBinaryOpKind.Mul:
                    op = arith.MuliOp(lhs_val, rhs_val)
                case AffineBinaryOpKind.Mod:
                    op = arith.RemSIOp(lhs_val, rhs_val)
                case AffineBinaryOpKind.FloorDiv:
                    op = arith.FloorDivSIOp(lhs_val, rhs_val)
                case AffineBinaryOpKind.CeilDiv:
                    op = arith.CeilDivSIOp(lhs_val, rhs_val)

            return [*lhs_ops, *rhs_ops, op], op.result
        case _:
            raise ValueError(f"Unexpected affine expr: {expr}")


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
    def match_and_rewrite(self, op: affine.StoreOp, rewriter: PatternRewriter):
        ops, indices = insert_affine_map_ops(op.map, op.indices, [])
        rewriter.insert_op(ops)

        # TODO: add nontemporal=false once that's added to memref
        # https://github.com/xdslproject/xdsl/issues/1482
        rewriter.replace_op(op, memref.StoreOp.get(op.value, op.memref, indices))


class LowerAffineLoad(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.LoadOp, rewriter: PatternRewriter):
        ops, indices = insert_affine_map_ops(op.map, op.indices, [])
        rewriter.insert_op(ops)

        # TODO: add nontemporal=false once that's added to memref
        # https://github.com/xdslproject/xdsl/issues/1482
        rewriter.replace_op(op, memref.LoadOp.get(op.memref, indices))


class LowerAffineFor(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.ForOp, rewriter: PatternRewriter):
        lb_map = op.lowerBoundMap.data
        ub_map = op.upperBoundMap.data
        assert len(lb_map.results) == 1, "Affine for lower_bound must have one result"
        assert len(ub_map.results) == 1, "Affine for upper_bound must have one result"
        lb_ops, lb_val = affine_expr_ops(lb_map.results[0], [], [])
        rewriter.insert_op(lb_ops)
        ub_ops, ub_val = affine_expr_ops(ub_map.results[0], [], [])
        rewriter.insert_op(ub_ops)
        step_op = arith.ConstantOp(op.step)
        rewriter.insert_op(step_op)
        rewriter.replace_op(
            op,
            scf.ForOp(
                lb_val,
                ub_val,
                step_op.result,
                op.inits,
                rewriter.move_region_contents_to_new_regions(op.body),
            ),
        )


class LowerAffineYield(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.YieldOp, rewriter: PatternRewriter, /):
        rewriter.replace_op(op, scf.YieldOp(*op.arguments))


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
        rewriter.replace_op(op, new_ops, new_results)


class LowerAffinePass(ModulePass):
    name = "lower-affine"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerAffineStore(),
                    LowerAffineLoad(),
                    LowerAffineFor(),
                    LowerAffineYield(),
                    LowerAffineApply(),
                ]
            )
        ).rewrite_module(op)
