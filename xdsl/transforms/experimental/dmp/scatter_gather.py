from dataclasses import dataclass

from xdsl.builder import Builder
from xdsl.dialects import arith, builtin, memref, mpi, scf
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute, BlockArgument, MLContext
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.experimental.dmp.decompositions import DomainDecompositionStrategy
from xdsl.transforms.experimental.dmp.stencil_global_to_local import (
    _grid_coords_from_rank,  # type: ignore[reportPrivateUsage]
)
from xdsl.transforms.experimental.dmp.stencil_global_to_local import (
    DmpDecompositionPass,
)
from xdsl.utils.hints import isa

idx = builtin.IndexType()


@dataclass
class LowerDmpScatter(RewritePattern):
    """
    Lower a dmp.scatter in the most trivial way possible:

    We are given the global domain.
    We copy our cells data into the origin of the domain.
    We are done.
    """

    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.ScatterOp, rewriter: PatternRewriter, /):
        shape_info = op.global_shape

        core_size = tuple(
            shape_info.core_size(i) for i in range(len(shape_info.core_lb))
        )

        local_domain = self.strategy.calc_resize(core_size)

        grid = self.strategy.comm_layout()

        # assert that we are only in 2d for now
        # TODO: write the rank->node coords mapping for n-dims
        assert len(core_size) == 2
        grid_tuple = grid.as_tuple()
        if len(grid_tuple) == 1:
            grid_tuple = (grid_tuple[0], 1)
        assert len(grid_tuple) == 2

        grid_coord_ops, grid_coords = _grid_coords_from_rank(op.my_rank, grid)

        # pad grid to 2d
        if len(grid_coords) == 1:
            cst0_ = arith.Constant.from_int_and_width(0, idx)
            grid_coord_ops.append(cst0_)
            grid_coords.append(cst0_.result)

        node_id_x, node_id_y = grid_coords

        ops = [
            *grid_coord_ops,
            # get local domain sizes
            local_domain_height := arith.Constant.from_int_and_width(
                local_domain[0], idx
            ),
            local_domain_width := arith.Constant.from_int_and_width(
                local_domain[1], idx
            ),
            # calculate the offset of our local domain to the core
            # (lower and upper bounds)
            offset_x := arith.Muli(node_id_x, local_domain_height),
            offset_y := arith.Muli(node_id_y, local_domain_width),
            offset_x_ub := arith.Addi(offset_x, local_domain_height),
            offset_y_ub := arith.Addi(offset_y, local_domain_width),
            # get the halo size
            halo_x := arith.Constant.from_int_and_width(
                shape_info.halo_size(0) * 2, idx
            ),
            halo_y := arith.Constant.from_int_and_width(
                shape_info.halo_size(1) * 2, idx
            ),
            # translate from core dims to buff dims
            # loop_lb_x := arith.Addi(offset_x, halo_x),
            # loop_lb_y := arith.Addi(offset_y, halo_y),
            loop_ub_x := arith.Addi(offset_x_ub, halo_x),
            loop_ub_y := arith.Addi(offset_y_ub, halo_y),
            # calc global halo local halo difference
            # get a constant one for the step
            cst1 := arith.Constant.from_int_and_width(1, idx),
        ]

        rewriter.insert_op_before_matched_op(ops)

        @Builder.implicit_region([idx, idx])
        def lööp_body(args: tuple[BlockArgument, ...]):
            x, y = args
            x.name_hint = "x"
            y.name_hint = "y"
            val = memref.Load.get(op.global_field, [x, y])
            x_dest = arith.Subi(x, offset_x)
            y_dest = arith.Subi(y, offset_y)
            memref.Store.get(val, op.global_field, [x_dest, y_dest])
            scf.Yield.get()

        lööp = scf.ParallelOp.get(
            [offset_x, offset_y], [loop_ub_x, loop_ub_y], [cst1, cst1], lööp_body
        )

        rewriter.replace_matched_op(lööp)


@dataclass
class LowerDmpGather(RewritePattern):
    """
    Lower a dmp.gather
    """

    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.GatherOp, rewriter: PatternRewriter, /):
        shape_info = op.global_shape

        core_size = tuple(
            shape_info.core_size(i) for i in range(len(shape_info.core_lb))
        )

        local_domain = self.strategy.calc_resize(core_size)

        rank = len(local_domain)

        # Step 1: Copy local data into separate buffer

        # calc offset

        assert isa(op.local_field.typ, memref.MemRefType[Attribute])
        el_typ = op.local_field.typ.element_type

        lbs = [
            arith.Constant.from_int_and_width(shape_info.halo_size(x), idx)
            for x in range(rank)
        ]
        ubs = [
            arith.Constant.from_int_and_width(
                shape_info.halo_size(x) + local_domain[x], idx
            )
            for x in range(rank)
        ]

        ops = [
            tmp_buff := memref.Alloc.get(el_typ, 32, local_domain),
            # lower bounds:
            *lbs,
            # upper bounds:
            *ubs,
            cst1 := arith.Constant.from_int_and_width(1, idx),
        ]

        rewriter.insert_op_before_matched_op(ops)

        @Builder.implicit_region([idx] * rank)
        def lööp_body(indices: tuple[BlockArgument, ...]):
            val = memref.Load.get(op.local_field, indices)
            offset_indices = [arith.Subi(i, lb) for i, lb in zip(indices, lbs)]
            memref.Store.get(val, tmp_buff, offset_indices)
            scf.Yield.get()

        lööp = scf.ParallelOp.get(lbs, ubs, [cst1] * rank, lööp_body)

        # Step 2: Issue mpi.gather()
        rewriter.insert_op_before_matched_op(
            [
                lööp,
                unwrapped_local := mpi.UnwrapMemrefOp.get(tmp_buff),
                unwrapped_global := mpi.UnwrapMemrefOp.get(op.local_field),
                my_rank := arith.IndexCastOp.get(op.my_rank, builtin.i32),
                root_rank := arith.Constant.from_int_and_width(
                    op.root_rank.value, builtin.i32
                ),
            ]
        )

        # Step 3: Insert the "if I'm root" block into a proper scf.if
        block = op.when_root_block.detach_block(0)
        global_ref = block.args[0]
        global_ref.replace_by(op.local_field)
        block.erase_arg(global_ref)

        rewriter.insert_op_at_end(scf.Yield.get(), block)

        rewriter.insert_op_after_matched_op(
            [
                am_root := arith.Cmpi.get(my_rank, root_rank, "eq"),
                scf.If.get(am_root, [], [block]),
            ]
        )

        rewriter.replace_matched_op(
            mpi.GatherOp(
                unwrapped_local.ptr,
                unwrapped_local.len,
                unwrapped_local.typ,
                unwrapped_global.ptr,
                unwrapped_local.len,
                unwrapped_local.typ,
                root_rank,
            )
        )


@dataclass
class DmpScatterGatherTrivialLowering(DmpDecompositionPass):
    name = "dmp-setup-and-teardown"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        strat = self.get_strategy()
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerDmpGather(strat),
                    LowerDmpScatter(strat),
                ]
            )
        ).rewrite_module(op)
