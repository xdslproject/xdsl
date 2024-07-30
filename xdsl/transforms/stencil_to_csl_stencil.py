from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from xdsl.context import MLContext
from xdsl.dialects import arith, stencil, tensor
from xdsl.dialects.builtin import (
    AnyMemRefType,
    AnyTensorType,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.csl import csl_stencil
from xdsl.dialects.experimental import dmp
from xdsl.ir import Attribute, Block, BlockArgument, Operation, OpResult, Region
from xdsl.irdl import Operand, base
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.experimental.stencil_tensorize_z_dimension import (
    BackpropagateStencilShapes,
)
from xdsl.utils.hints import isa
from xdsl.utils.isattr import isattr


def get_stencil_access_operands(op: Operand) -> set[Operand]:
    """
    Returns the symbols of all stencil accessess by op and all its dependencies.
    """
    res: set[Operand] = set()
    frontier: set[Operand] = {op}
    next: set[Operand] = set()
    while len(frontier) > 0:
        for o in frontier:
            if isinstance(o, OpResult):
                if isinstance(o.op, csl_stencil.AccessOp):
                    res.add(o.op.op)
                else:
                    next.update(o.op.operands)
        frontier = next
        next = set()

    return res


@dataclass(frozen=True)
class RestructureSymmetricReductionPattern(RewritePattern):
    """
    Consume data first where that data comes from stencil accesses to `buf`.

    Identifies a pattern of 2 connected binary ops with 3 args, e.g. of the form `(a+b)+c` with different ops and
    bracketings supported, and attempts to re-structure the order of computation.

    Being in principle similarly to constant folding, the difference is that args are *not* required to be stencil
    accesses, but could have further compute applied before being passed to the reduction function.
    Uses helper function `get_stencil_accessed_symbols` to check which bufs are stencil-accessed in each of these args,
    and to distinguish the following three cases:

     (1) all accesses in an arg tree are to `buf`  - arg should be moved forward in the computation
     (2) no accesses are to `buf`                  - arg should be moved backward in the computation
     (3) there's a mix                             - unknown, take any or no action

    If two args are identified that should be moved forward, or two args are identified that should be moved backwards,
    the computation is restructured accordingly.
    """

    buf: BlockArgument

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: arith.Addf | arith.Mulf, rewriter: PatternRewriter, /
    ):
        # this rewrite requires exactly 1 use which is the same type of operation
        if len(op.result.uses) != 1 or not isinstance(
            use := list(op.result.uses)[0].operation, type(op)
        ):
            return
        c_op = use.operands[0] if use.operands[1] == op.result else use.operands[1]

        def rewrite(one: Operand, two: Operand, three: Operand):
            """Builds `(one+two)+three` where `'+' == type(op)`"""

            first_compute = type(op)(one, two)
            second_compute = type(op)(first_compute, three)

            # Both ops are inserted at the later point to ensure all dependencies are present when moving compute around.
            # Moving the replacement of `op` backwards is safe because we previously asserted at `op` only has one use (ie. in `use`)
            rewriter.replace_op(op, [], [first_compute.results[0]])
            rewriter.replace_op(
                use, [first_compute, second_compute], [second_compute.results[0]]
            )

        a = get_stencil_access_operands(a_op := op.lhs)
        b = get_stencil_access_operands(b_op := op.rhs)
        c = get_stencil_access_operands(c_op)

        if self.move_fwd(a) and self.move_fwd(b):
            return
        elif self.move_back(a) and self.move_back(b):
            return
        elif self.move_fwd(c) and self.move_back(b):
            rewrite(a_op, c_op, b_op)

    def move_fwd(self, accs: set[Operand]) -> bool:
        return self.buf in accs and len(accs) == 1

    def move_back(self, accs: set[Operand]) -> bool:
        return self.buf not in accs


@dataclass(frozen=True)
class ConvertAccessOpFromPrefetchPattern(RewritePattern):
    """
    Rebuilds stencil.access by csl_stencil.access which operates on prefetched accesses.

    stencil.access operates on stencil.temp types found at arg_index
    csl_stencil.access operates on memref< num_neighbors x tensor< buf_size x data_type >> found at last arg index

    Note: This is intended to be called in a nested pattern rewriter, such that the above precondition is met.
    """

    arg_index: int

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.AccessOp, rewriter: PatternRewriter, /):
        assert len(op.offset) == 2
        if op.temp != op.get_apply().region.block.args[self.arg_index]:
            return

        # translate access to own data, which operates on stencil.TempType
        if tuple(op.offset) == (0, 0):
            assert isattr(op.res.type, base(AnyTensorType))
            rewriter.replace_matched_op(
                csl_stencil.AccessOp(
                    op=op.temp,
                    offset=op.offset,
                    offset_mapping=op.offset_mapping,
                    result_type=op.res.type,
                )
            )
            return

        prefetched_arg = op.get_apply().region.block.args[-1]
        assert isa(t_type := prefetched_arg.type, TensorType[Attribute])

        csl_access_op = csl_stencil.AccessOp(
            op=prefetched_arg,
            offset=op.offset,
            offset_mapping=op.offset_mapping,
            result_type=TensorType(t_type.get_element_type(), t_type.get_shape()[1:]),
        )

        # The stencil-tensorize-z-dimension pass inserts tensor.ExtractSliceOps after stencil.access to remove ghost cells.
        # Since ghost cells are not prefetched, these ops can be removed again. Check if the ExtractSliceOp
        # has no other effect and if so, remove both.
        if (
            len(op.res.uses) == 1
            and isinstance(use := list(op.res.uses)[0].operation, tensor.ExtractSliceOp)
            and tuple(d.data for d in use.static_sizes.data) == t_type.get_shape()[1:]
            and tuple(d.data for d in use.static_offsets.data) == (0,)
            and tuple(d.data for d in use.static_strides.data) == (1,)
            and len(use.offsets) == 0
            and len(use.sizes) == 0
            and len(use.strides) == 0
        ):
            rewriter.replace_op(use, csl_access_op)
            rewriter.erase_op(op)
        else:
            rewriter.replace_matched_op(csl_access_op)


@dataclass(frozen=True)
class ConvertSwapToPrefetchPattern(RewritePattern):
    """
    Translates dmp.swap to csl_stencil.prefetch
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        # remove op if it contains no swaps
        if op.swaps is None or len(op.swaps) == 0:
            rewriter.erase_matched_op(False)
            return

        assert all(
            len(swap.size) == 3 for swap in op.swaps
        ), "currently only 3-dimensional stencils are supported"

        assert all(
            swap.size[:2] == (1, 1) for swap in op.swaps
        ), "invoke dmp to decompose from (x,y,z) to (1,1,z)"

        # check that size is uniform
        uniform_size = op.swaps.data[0].size[2]
        assert all(
            swap.size[2] == uniform_size for swap in op.swaps
        ), "all swaps need to be of uniform size"

        assert isinstance(op.input_stencil, OpResult)
        assert isattr(
            op.input_stencil.type,
            base(AnyMemRefType) | base(stencil.AnyTempType),
        )
        assert isa(
            t_type := op.input_stencil.type.get_element_type(), TensorType[Attribute]
        )
        assert op.topo is not None, f"topology on {type(op)} is not given"

        # when translating swaps, remove third dimension
        prefetch_op = csl_stencil.PrefetchOp(
            input_stencil=op.input_stencil.op,
            topo=op.topo,
            swaps=[
                csl_stencil.ExchangeDeclarationAttr(swap.neighbor[:2])
                for swap in op.swaps
            ],
            result_type=TensorType(
                t_type.get_element_type(),
                (len(op.swaps), uniform_size),
            ),
        )

        # a little hack to get around a check that prevents replacing a no-results op with an n-results op
        rewriter.replace_matched_op(prefetch_op, new_results=[])

        # uses have to be retrieved *before* the loop because of the rewriting happening inside the loop
        uses = list(op.input_stencil.uses)

        # csl_stencil.prefetch, unlike dmp.swap, has a return value. This is added as the last arg
        # to stencil.apply, before rebuilding the op and replacing stencil.access ops by csl_stencil.access ops
        # that reference the prefetched buffers (note, this is only done for neighbor accesses)
        for use in uses:
            if not isinstance(use.operation, stencil.ApplyOp):
                continue
            apply_op = use.operation

            # arg_idx points to the stencil.temp type whose data is prefetched in a separate buffer
            arg_idx = apply_op.args.index(op.input_stencil)

            # add the prefetched buffer as the last arg to stencil.access
            apply_op.region.block.insert_arg(
                prefetch_op.result.type, len(apply_op.args)
            )

            # rebuild stencil.apply op
            r_types = [r.type for r in apply_op.results]
            assert isa(r_types, Sequence[stencil.TempType[Attribute]])
            new_apply_op = stencil.ApplyOp.get(
                [*apply_op.args, prefetch_op.result],
                apply_op.detach_region(apply_op.region),
                r_types,
            )
            rewriter.replace_op(apply_op, new_apply_op)

            # replace stencil.access (operating on stencil.temp at arg_index)
            # with csl_stencil.access (operating on memref at last arg index)
            nested_rewriter = PatternRewriteWalker(
                ConvertAccessOpFromPrefetchPattern(arg_idx)
            )

            nested_rewriter.rewrite_op(new_apply_op)


def get_op_split(
    ops: Sequence[Operation], buf: BlockArgument
) -> tuple[Sequence[Operation], Sequence[Operation]]:
    """
    Returns a split of `ops` into an `(a,b)` tuple, such that:

    - `a` contains neighbour accesses to `buf` plus the minumum set of instructions to reduce the accessed data to 1 thing
    - `b` contains everything else

    If no valid split can be found, return `(ops, [])`.

    This function does not attempt to arithmetically re-structure the computation to obtain a good split. To do this,
    `RestructureSymmetricReductionPattern()` may be executed first.
    """
    a: list[Operation] = []
    b: list[Operation] = []
    rem: list[Operation] = []
    for op in ops:
        if isinstance(op, csl_stencil.AccessOp):
            (b, a)[op.op == buf].append(op)
        else:
            rem.append(op)

    # loop until we can make no more changes, or until only 1 thing computed in `a` is used outside of it
    has_changes = True
    while (
        len(
            # ops in `a` whose results are used outside of `a`
            a_exports := set(
                op
                for op in a
                for result in op.results
                for use in result.uses
                if use.operation not in a
            )
        )
        > 1
        and has_changes
    ):
        has_changes = False

        # find ops that directly depend on `a` but are not themselves in `a`
        for exp in a_exports:
            for result in exp.results:
                for use in result.uses:
                    # op is only movable if *all* operands are already in `a` (and it hasn't been moved yet)
                    if (op := use.operation) in rem and all(
                        x.op in a for x in op.operands if isinstance(x, OpResult)
                    ):
                        has_changes = True
                        a.append(use.operation)
                        rem.remove(use.operation)

    if len(a_exports) == 1:
        return a, b + rem

    # fallback
    # always place `stencil.return` in second block
    return ops[:-1], [ops[-1]]


@dataclass(frozen=True)
class ConvertApplyOpPattern(RewritePattern):
    """
    Fuses a `csl_stencil.prefetch` and a `stencil.apply` to build a `csl_stencil.apply`.

    If there are several candidate prefetch ops, the one with the largest result buffer size is selected.
    The selection is greedy, and could in the future be expanded into a more global selection optimising for minimal
    prefetch overhead across multiple apply ops.

    args:
        num_chunks - number of chunks into which communication and computation should be split.
                     Effectively, the number of times `csl_stencil.apply.chunk_reduce` will be executed and the
                     tensor sizes it handles. Higher values may increase compute overhead but reduce size of
                     communication buffers when lowered.
    """

    num_chunks: int = 1

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.ApplyOp, rewriter: PatternRewriter, /):
        # calculate memory cost of all prefetch operands
        def get_prefetch_overhead(o: OpResult):
            assert isa(o.type, TensorType[Attribute])
            buf_count = o.type.get_shape()[0]
            buf_size = prod(o.type.get_shape()[1:])
            return buf_count * buf_size

        candidate_prefetches = [
            (get_prefetch_overhead(o), o)
            for o in op.operands
            if isinstance(o, OpResult) and isinstance(o.op, csl_stencil.PrefetchOp)
        ]
        if len(candidate_prefetches) == 0:
            return

        # select the prefetch with the biggest communication overhead to be fused with matched stencil.apply
        prefetch = max(candidate_prefetches)[1]
        prefetch_idx = op.operands.index(prefetch)
        assert isinstance(prefetch.op, csl_stencil.PrefetchOp)
        communicated_stencil_idx = op.operands.index(prefetch.op.input_stencil)
        assert isinstance(prefetch.op, csl_stencil.PrefetchOp)
        assert isa(prefetch.type, TensorType[Attribute])
        communicated_stencil_op_arg = prefetch.op.input_stencil

        # add empty tensor before op to be used as `iter_arg`
        # this could potentially be re-used if we have one of the same size lying around
        iter_arg = tensor.EmptyOp(
            (),
            TensorType(prefetch.type.get_element_type(), prefetch.type.get_shape()[1:]),
        )
        rewriter.insert_op(iter_arg, InsertPoint.before(op))

        # run pass (on this apply's region only) to consume data from `prefetch` accesses first
        nested_rewriter = PatternRewriteWalker(
            RestructureSymmetricReductionPattern(op.region.block.args[prefetch_idx]),
            walk_reverse=True,
        )
        nested_rewriter.rewrite_op(op)

        # determine how ops should be split across the two regions
        chunk_reduce_ops, post_process_ops = get_op_split(
            list(op.region.block.ops), op.region.block.args[prefetch_idx]
        )

        # fetch what chunk_reduce is computing for
        if isinstance(chunk_reduce_ops[-1], stencil.ReturnOp):
            chunk_res = chunk_reduce_ops[-1].operands[0]
        else:
            chunk_res = chunk_reduce_ops[-1].results[0]

        # after region split, check which block args (from the old ops block) are being accessed in each of the new regions
        # ignore accesses block args which already are part of the region's required signature
        chunk_reduce_used_block_args = sorted(
            set(
                x
                for o in chunk_reduce_ops
                for x in o.operands
                if isinstance(x, BlockArgument) and x.index != prefetch_idx
            ),
            key=lambda b: b.index,
        )
        post_process_used_block_args = sorted(
            set(
                x
                for o in post_process_ops
                for x in o.operands
                if isinstance(x, BlockArgument) and x.index != communicated_stencil_idx
            ),
            key=lambda b: b.index,
        )

        # set up region signatures, comprising fixed and optional args - see docs on `csl_stencil.apply` for details
        chunk_reduce_args = [
            # required arg 0: slice of type(%prefetch)
            TensorType(
                prefetch.type.get_element_type(),
                (
                    len(prefetch.op.swaps),
                    prefetch.type.get_shape()[1] // self.num_chunks,
                ),
            ),
            # required arg 1: %offset
            IndexType(),
            # required arg 2: %iter_arg
            iter_arg.results[0].type,
            # optional args: as needed by the ops
            *[a.type for a in chunk_reduce_used_block_args],
        ]
        post_process_args = [
            # required arg 0: stencil.temp to access own data
            communicated_stencil_op_arg.type,
            # required arg 1: %iter_arg
            iter_arg.results[0].type,
            # optional args: as needed by the ops
            *[a.type for a in post_process_used_block_args],
        ]

        # set up two regions
        chunk_reduce = Region(Block(arg_types=chunk_reduce_args))
        post_process = Region(Block(arg_types=post_process_args))

        # translate old to new block arg index for optional args
        chunk_reduce_oprnd_table = dict[Operand, Operand](
            (old, chunk_reduce.block.args[idx])
            for idx, old in enumerate(chunk_reduce_used_block_args, start=3)
        )
        post_process_oprnd_table = dict[Operand, Operand](
            (old, post_process.block.args[idx])
            for idx, old in enumerate(post_process_used_block_args, start=2)
        )

        # add translation from old to new arg index for non-optional args - note, access to iter_arg must be handled separately below
        chunk_reduce_oprnd_table[op.region.block.args[prefetch_idx]] = (
            chunk_reduce.block.args[0]
        )
        post_process_oprnd_table[op.region.block.args[communicated_stencil_idx]] = (
            post_process.block.args[0]
        )
        post_process_oprnd_table[chunk_res] = post_process.block.args[1]

        # detach ops from old region
        for o in op.region.block.ops:
            op.region.block.detach_op(o)

        # add operations from list to chunk_reduce, use translation table to rebuild operands
        for o in chunk_reduce_ops:
            if isinstance(o, stencil.ReturnOp | csl_stencil.YieldOp):
                break
            o.operands = [chunk_reduce_oprnd_table.get(x, x) for x in o.operands]
            chunk_reduce.block.add_op(o)

        # put `chunk_res` into `iter_arg` (using tensor.insert_slice) and yield the result
        chunk_reduce.block.add_ops(
            [
                insert_slice_op := tensor.InsertSliceOp.get(
                    source=chunk_res,
                    dest=chunk_reduce.block.args[2],
                    offsets=(chunk_reduce.block.args[1],),
                    static_sizes=(prefetch.type.get_shape()[1] // self.num_chunks,),
                ),
                csl_stencil.YieldOp(insert_slice_op.result),
            ]
        )

        # add operations from list to post_process, use translation table to rebuild operands
        for o in post_process_ops:
            o.operands = [post_process_oprnd_table.get(x, x) for x in o.operands]
            post_process.block.add_op(o)
            if isinstance(o, stencil.ReturnOp):
                rewriter.replace_op(o, csl_stencil.YieldOp(*o.operands))

        rewriter.replace_matched_op(
            csl_stencil.ApplyOp(
                operands=[
                    communicated_stencil_op_arg,
                    iter_arg,
                    [op.operands[a.index] for a in chunk_reduce_used_block_args]
                    + [op.operands[a.index] for a in post_process_used_block_args],
                ],
                properties={
                    "swaps": prefetch.op.swaps,
                    "topo": prefetch.op.topo,
                    "num_chunks": IntegerAttr(self.num_chunks, IntegerType(64)),
                },
                regions=[
                    chunk_reduce,
                    post_process,
                ],
                result_types=[r.type for r in op.results],
            )
        )

        if len(prefetch.uses) == 0:
            rewriter.erase_op(prefetch.op)


@dataclass(frozen=True)
class StencilToCslStencilPass(ModulePass):
    name = "stencil-to-csl-stencil"

    # chunks into which to slice communication
    num_chunks: int = 1

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertSwapToPrefetchPattern(),
                    ConvertApplyOpPattern(num_chunks=self.num_chunks),
                ]
            ),
            walk_reverse=False,
            apply_recursively=True,
        )
        module_pass.rewrite_module(op)

        if self.num_chunks > 1:
            BackpropagateStencilShapes().apply(ctx, op)
