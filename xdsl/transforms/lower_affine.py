from xdsl.dialects import affine, arith, builtin, memref, scf
from xdsl.ir import MLContext, Operation, SSAValue
from xdsl.ir.affine import AffineConstantExpr
from xdsl.ir.affine.affine_expr import (
    AffineBinaryOpExpr,
    AffineBinaryOpKind,
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
    dims: list[SSAValue],
    symbols: list[SSAValue],
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
    dims: list[SSAValue],
    symbols: list[SSAValue],
) -> tuple[list[Operation], list[SSAValue]]:
    """
    Returns operations that evaluate the affine map when given input SSA values and the
    resulting indices.
    """
    ops: list[Operation] = []
    if map is None:
        indices = dims
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
        lb_map = op.lower_bound.data
        ub_map = op.upper_bound.data
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
                op.arguments,
                rewriter.move_region_contents_to_new_regions(op.body),
            )
        )


class LowerAffineYield(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: affine.Yield, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(scf.Yield(*op.arguments))


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
                ]
            )
        ).rewrite_module(op)
