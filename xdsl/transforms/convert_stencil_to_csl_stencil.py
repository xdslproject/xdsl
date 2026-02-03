from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from xdsl.builder import ImplicitBuilder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, stencil, tensor, varith
from xdsl.dialects.builtin import (
    DYNAMIC_INDEX,
    AnyTensorType,
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
    ModuleOp,
    TensorType,
)
from xdsl.dialects.csl import csl_stencil
from xdsl.dialects.experimental import dmp
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    OpResult,
    Region,
    SSAValue,
)
from xdsl.irdl import Operand
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
from xdsl.transforms.varith_transformations import (
    ConvertArithToVarithPass,
    ConvertVarithToArithPass,
)
from xdsl.utils.hints import isa


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


def _get_prefetch_buf_idx(op: stencil.ApplyOp) -> int | None:
    # calculate memory cost of all prefetch operands
    def get_prefetch_overhead(o: OpResult):
        assert isa(o.type, TensorType[Attribute])
        return prod(o.type.get_shape())

    candidate_prefetches = [
        (get_prefetch_overhead(o), o)
        for o in op.operands
        if isinstance(o, OpResult) and isinstance(o.op, csl_stencil.PrefetchOp)
    ]
    if len(candidate_prefetches) == 0:
        return

    # select the prefetch with the biggest communication overhead to be fused with matched stencil.apply
    prefetch = max(candidate_prefetches, key=lambda x: x[0])[1]
    return op.operands.index(prefetch)


def _get_apply_op(op: Operation) -> stencil.ApplyOp | None:
    """
    Return the enclosing csl_wrapper.module
    """
    parent_op = op.parent_op()
    while parent_op:
        if isinstance(parent_op, stencil.ApplyOp):
            return parent_op
        parent_op = parent_op.parent_op()
    return None


@dataclass(frozen=True)
class ConvertAccessOpPattern(RewritePattern):
    """
    Rebuilds stencil.access by csl_stencil.access which operates on prefetched accesses.

    stencil.access operates on stencil.temp types found at arg_index
    csl_stencil.access operates on memref< num_neighbors x tensor< buf_size x data_type >> found at last arg index

    Note: This is intended to be called in a nested pattern rewriter, such that the above precondition is met.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.AccessOp, rewriter: PatternRewriter, /):
        assert len(op.offset) == 2
        if isa(op.temp.type, AnyTensorType):
            res_type = TensorType(
                op.temp.type.get_element_type(), op.temp.type.get_shape()[1:]
            )
        else:
            assert isa(op.res.type, AnyTensorType)
            res_type = op.res.type
        rewriter.replace_op(
            op,
            new_access_op := csl_stencil.AccessOp(
                op=op.temp,
                offset=op.offset,
                offset_mapping=op.offset_mapping,
                result_type=res_type,
            ),
        )

        # The stencil-tensorize-z-dimension pass inserts tensor.ExtractSliceOps after stencil.access to remove ghost cells.
        # Since ghost cells are not prefetched, these ops can be removed again. Check if the ExtractSliceOp
        # has no other effect and if so, remove both.
        if (
            isinstance(
                use := new_access_op.result.get_user_of_unique_use(),
                tensor.ExtractSliceOp,
            )
            and use.static_sizes.get_values() == res_type.get_shape()
            and len(use.offsets) == 0
            and len(use.sizes) == 0
            and len(use.strides) == 0
        ):
            rewriter.replace_op(use, [], new_results=[new_access_op.result])


@dataclass
class ConvertSwapToPrefetchPattern(RewritePattern):
    """
    Translates dmp.swap to csl_stencil.prefetch
    """

    num_chunks: int = 1

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.SwapOp, rewriter: PatternRewriter, /):
        # remove op if it contains no swaps
        if len(op.swaps) == 0:
            rewriter.erase_op(op, safe_erase=False)
            return

        assert all(len(swap.size) == 3 for swap in op.swaps), (
            "currently only 3-dimensional stencils are supported"
        )

        assert all(swap.size[:2] == (1, 1) for swap in op.swaps), (
            "invoke dmp to decompose from (x,y,z) to (1,1,z)"
        )

        # check that size is uniform
        uniform_size = op.swaps.data[0].size[2]
        assert all(swap.size[2] == uniform_size for swap in op.swaps), (
            "all swaps need to be of uniform size"
        )

        assert (MemRefType.constr() | stencil.StencilTypeConstr).verifies(
            op.input_stencil.type
        )
        assert isa(
            t_type := op.input_stencil.type.get_element_type(), TensorType[Attribute]
        )
        assert op.strategy.comm_layout() is not None, (
            f"topology on {type(op)} is not given"
        )

        # when translating swaps, remove third dimension
        prefetch_op = csl_stencil.PrefetchOp(
            input_stencil=op.input_stencil,
            topo=op.strategy.comm_layout(),
            num_chunks=IntegerAttr(self.num_chunks, 64),
            swaps=[
                csl_stencil.ExchangeDeclarationAttr(swap.neighbor[:2])
                for swap in op.swaps
            ],
            result_type=TensorType(
                t_type.get_element_type(),
                (len(op.swaps), uniform_size),
            ),
        )

        # if the rewriter needs a result, use `input_stencil` as a drop-in replacement
        # prefetch_op produces a result that needs to be handled separately
        # note, that only un-bufferized dmp.swaps produce a result
        rewriter.replace_op(
            op, prefetch_op, new_results=[op.input_stencil] if op.swapped_values else []
        )

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
            field_block_arg = apply_op.region.block.args[arg_idx]

            # add the prefetched buffer as the last arg to stencil.access
            prefetch_block_arg = apply_op.region.block.insert_arg(
                prefetch_op.result.type, len(apply_op.args)
            )
            field_block_arg.replace_by_if(
                prefetch_block_arg,
                lambda use: isinstance(use.operation, stencil.AccessOp)
                and tuple(use.operation.offset) != (0, 0),
            )

            # rebuild stencil.apply op
            r_types = apply_op.result_types
            assert isa(r_types, Sequence[stencil.TempType[Attribute]])
            new_apply_op = stencil.ApplyOp.build(
                operands=[[*apply_op.args, prefetch_op.result], apply_op.dest],
                regions=[apply_op.detach_region(apply_op.region)],
                result_types=[r_types],
                properties=apply_op.properties,
                attributes=apply_op.attributes,
            )
            rewriter.replace_op(apply_op, new_apply_op)


def split_ops(
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
        elif isinstance(op, arith.ConstantOp):
            a.append(op)
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

    # find constants in `a` needed outside of `a`
    cnst_exports = tuple(
        cnst for cnst in a_exports if isinstance(cnst, arith.ConstantOp)
    )

    # `a` exports one value plus any number of constants - duplicate exported constants and return op split
    if len(a_exports) == 1 + len(cnst_exports):
        recv_chunk_ops, done_exch_ops = list[Operation](), list[Operation]()
        for op in ops:
            if op in a:
                recv_chunk_ops.append(op)
                if op in cnst_exports:
                    assert isinstance(op, arith.ConstantOp)
                    # create a copy of the constant in the second region
                    done_exch_ops.append(cln := op.clone())
                    # rewire ops of the second region to use the copied constant
                    op.result.replace_by_if(
                        cln.result,
                        lambda use: use.operation in b or use.operation in rem,
                    )
            else:
                done_exch_ops.append(op)

        return recv_chunk_ops, done_exch_ops

    # fallback
    # always place `stencil.return` in second block
    return ops[:-1], [ops[-1]]


@dataclass(frozen=True)
class SplitVarithOpPattern(RewritePattern):
    """
    Splits a varith op into two, depending on whether the operands holds stencil accesses to `buf` (only)
    or any other accesses.

    This pass is intended to be run with `buf` set to the block arg indicating data received from neighbours.
    """

    buf: BlockArgument

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: varith.VarithOp, rewriter: PatternRewriter, /):
        if not (apply := _get_apply_op(op)) or not (
            buf_idx := _get_prefetch_buf_idx(apply)
        ):
            return
        buf = apply.region.block.args[buf_idx]
        buf_accesses, others = list[SSAValue](), list[SSAValue]()

        for arg in op.args:
            accs = get_stencil_access_operands(arg)
            (others, buf_accesses)[buf in accs and len(accs) == 1].append(arg)

        if len(others) > 0 and len(buf_accesses) > 0:
            rewriter.replace_op(
                op,
                [
                    n_op := type(op)(*buf_accesses),
                    type(op)(n_op, *others),
                ],
            )


@dataclass(frozen=True)
class ConvertApplyOpPattern(RewritePattern):
    """
    Fuses a `csl_stencil.prefetch` and a `stencil.apply` to build a `csl_stencil.apply`.

    If there are several candidate prefetch ops, the one with the largest result buffer
    size is selected.
    The selection is greedy, and could in the future be expanded into a more global
    selection optimising for minimal prefetch overhead across multiple apply ops.
    """

    num_chunks: int = 1
    """
    Number of chunks into which communication and computation should be split.
    Effectively, the number of times `csl_stencil.apply.receive_chunk` will be executed
    and the tensor sizes it handles.
    Higher values may increase compute overhead but reduce size of communication buffers
    when lowered.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.ApplyOp, rewriter: PatternRewriter, /):
        if not (prefetch_idx := _get_prefetch_buf_idx(op)):
            return

        # select the prefetch with the biggest communication overhead to be fused with matched stencil.apply
        prefetch = op.operands[prefetch_idx]
        assert isinstance(prefetch, OpResult)
        assert isinstance(prefetch.op, csl_stencil.PrefetchOp)
        field_idx = op.operands.index(prefetch.op.input_stencil)
        assert isinstance(prefetch.op, csl_stencil.PrefetchOp)
        assert isa(prefetch.type, TensorType[Attribute])
        field_op_arg = prefetch.op.input_stencil

        # add empty tensor before op to be used as `accumulator`
        # this could potentially be re-used if we have one of the same size lying around
        accumulator = tensor.EmptyOp(
            (),
            TensorType(prefetch.type.get_element_type(), prefetch.type.get_shape()[1:]),
        )
        rewriter.insert_op(accumulator, InsertPoint.before(op))

        # run pass (on this apply's region only) to consume data from `prefetch` accesses first
        # find varith ops and split according to neighbour data
        PatternRewriteWalker(
            SplitVarithOpPattern(op.region.block.args[prefetch_idx]),
            apply_recursively=False,
            listener=rewriter,
        ).rewrite_region(op.region)

        # determine how ops should be split across the two regions
        chunk_region_ops, done_exchange_ops = split_ops(
            list(op.region.block.ops), op.region.block.args[prefetch_idx]
        )

        # fetch what receive_chunk is computing for
        if isinstance(chunk_region_ops[-1], stencil.ReturnOp):
            chunk_res = chunk_region_ops[-1].operands[0]
        else:
            chunk_res = chunk_region_ops[-1].results[0]

        # after region split, check which block args (from the old ops block) are being accessed in each of the new regions
        # ignore accesses block args which already are part of the region's required signature
        chunk_region_used_block_args = sorted(
            set(
                x
                for o in chunk_region_ops
                for x in o.operands
                if isinstance(x, BlockArgument) and x.index != prefetch_idx
            ),
            key=lambda b: b.index,
        )
        done_exchange_used_block_args = sorted(
            set(
                x
                for o in done_exchange_ops
                for x in o.operands
                if isinstance(x, BlockArgument) and x.index != field_idx
            ),
            key=lambda b: b.index,
        )

        # set up region signatures, comprising fixed and optional args - see docs on `csl_stencil.apply` for details
        chunk_region_args = [
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
            # required arg 2: %accumulator
            accumulator.tensor.type,
            # optional args: as needed by the ops
            *[a.type for a in chunk_region_used_block_args],
        ]
        done_exchange_args = [
            # required arg 0: stencil.temp to access own data
            field_op_arg.type,
            # required arg 1: %accumulator
            accumulator.tensor.type,
            # optional args: as needed by the ops
            *[a.type for a in done_exchange_used_block_args],
        ]

        # set up two regions
        receive_chunk = Region(Block(arg_types=chunk_region_args))
        done_exchange = Region(Block(arg_types=done_exchange_args))

        # translate old to new block arg index for optional args
        chunk_region_oprnd_table = dict[Operand, Operand](
            (old, receive_chunk.block.args[idx])
            for idx, old in enumerate(chunk_region_used_block_args, start=3)
        )
        done_exchange_oprnd_table = dict[Operand, Operand](
            (old, done_exchange.block.args[idx])
            for idx, old in enumerate(done_exchange_used_block_args, start=2)
        )

        # add translation from old to new arg index for non-optional args - note, access
        # to accumulator must be handled separately below
        chunk_region_oprnd_table[op.region.block.args[prefetch_idx]] = (
            receive_chunk.block.args[0]
        )
        done_exchange_oprnd_table[op.region.block.args[field_idx]] = (
            done_exchange.block.args[0]
        )
        done_exchange_oprnd_table[chunk_res] = done_exchange.block.args[1]

        # detach ops from old region
        for o in op.region.block.ops:
            op.region.block.detach_op(o)

        # add operations from list to receive_chunk, use translation table to rebuild operands
        for o in chunk_region_ops:
            if isinstance(o, stencil.ReturnOp | csl_stencil.YieldOp):
                break
            o.operands = [chunk_region_oprnd_table.get(x, x) for x in o.operands]
            rewriter.insert_op(o, InsertPoint.at_end(receive_chunk.block))

        # put `chunk_res` into `accumulator` (using tensor.insert_slice) and yield the result
        rewriter.insert_op(
            [
                insert_slice_op := tensor.InsertSliceOp.get(
                    source=chunk_res,
                    dest=receive_chunk.block.args[2],
                    offsets=(receive_chunk.block.args[1],),
                    static_sizes=(prefetch.type.get_shape()[1] // self.num_chunks,),
                ),
                csl_stencil.YieldOp(insert_slice_op.result),
            ],
            InsertPoint.at_end(receive_chunk.block),
        )

        # add operations from list to done_exchange, use translation table to rebuild operands
        for o in done_exchange_ops:
            o.operands = [done_exchange_oprnd_table.get(x, x) for x in o.operands]
            rewriter.insert_op(o, InsertPoint.at_end(done_exchange.block))
            if isinstance(o, stencil.ReturnOp):
                rewriter.replace_op(o, csl_stencil.YieldOp(*o.operands))

        rewriter.replace_op(
            op,
            csl_stencil.ApplyOp(
                operands=[
                    field_op_arg,
                    accumulator,
                    [op.operands[a.index] for a in chunk_region_used_block_args],
                    [op.operands[a.index] for a in done_exchange_used_block_args],
                    op.dest,
                ],
                properties={
                    "swaps": prefetch.op.swaps,
                    "topo": prefetch.op.topo,
                    "num_chunks": IntegerAttr(self.num_chunks, IntegerType(64)),
                    "bounds": op.bounds,
                },
                regions=[
                    receive_chunk,
                    done_exchange,
                ],
                result_types=[op.result_types],
            ),
        )

        if not prefetch.uses:
            rewriter.erase_op(prefetch.op)


class PromoteCoefficients(RewritePattern):
    """
    Promotes constant coefficients to attributes. When a `csl_stencil.access` is immediately multiplied by
    an `arith.constant` as the sole use of the accessed data, the constant is promoted to a coefficient property
    in the `csl_stencil.apply` op.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.AccessOp, rewriter: PatternRewriter, /):
        if (
            not isinstance(apply := op.get_apply(), csl_stencil.ApplyOp)
            or not op.op == apply.receive_chunk.block.args[0]
            or not isinstance(mulf := op.result.get_user_of_unique_use(), arith.MulfOp)
        ):
            return

        coeff = mulf.lhs if op.result == mulf.rhs else mulf.rhs

        if (
            not isinstance(cnst := coeff.owner, arith.ConstantOp)
            or not isinstance(dense := cnst.value, DenseIntOrFPElementsAttr)
            or not dense.is_splat()
        ):
            return

        val = dense.get_attrs()[0]
        assert isinstance(val, FloatAttr)
        apply.add_coeff(op.offset, val)
        rewriter.replace_op(mulf, [], new_results=[op.result])


class TransformPrefetch(RewritePattern):
    """
    Rewrites a prefetch into a communicate-only csl_stencil.apply
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: csl_stencil.PrefetchOp, rewriter: PatternRewriter, /
    ):
        a_buf = tensor.EmptyOp((), op.result.type)
        # because we are building a set of offsets, we are not retaining offset mappings
        offsets = [swap.neighbor for swap in op.swaps]

        assert isa(op.result.type, AnyTensorType)
        chunk_buf_t = TensorType(
            op.result.type.get_element_type(),
            (
                len(op.swaps),
                op.result.type.get_shape()[1] // op.num_chunks.value.data,
            ),
        )
        chunk_t = TensorType(chunk_buf_t.element_type, chunk_buf_t.get_shape()[1:])

        block = Block(arg_types=[chunk_buf_t, builtin.IndexType(), op.result.type])
        block2 = Block(arg_types=[op.input_stencil.type, op.result.type])
        block2.add_op(csl_stencil.YieldOp())

        with ImplicitBuilder(block) as (buf, offset, acc):
            dest = acc
            for i, acc_offset in enumerate(offsets):
                ac_op = csl_stencil.AccessOp(
                    buf, stencil.IndexAttr.get(*acc_offset), chunk_t
                )
                assert isa(ac_op.result.type, AnyTensorType)
                # inserts 1 (see static_sizes) 1d slice into a 2d tensor at offset (i, `offset`) (see static_offsets)
                # where the latter offset is provided dynamically (see offsets)
                dest = tensor.InsertSliceOp.get(
                    source=ac_op.result,
                    dest=dest,
                    static_sizes=[1, *ac_op.result.type.get_shape()],
                    static_offsets=[i, DYNAMIC_INDEX],
                    offsets=[offset],
                ).result
            csl_stencil.YieldOp(dest)

        apply_op = csl_stencil.ApplyOp(
            operands=[op.input_stencil, a_buf, [], [], []],
            regions=[Region(block), Region(block2)],
            properties={
                "swaps": op.swaps,
                "topo": op.topo,
                "num_chunks": op.num_chunks,
            },
            result_types=[[]],
        )

        rewriter.replace_op(op, [a_buf, apply_op], new_results=[a_buf.tensor])


@dataclass(frozen=True)
class ConvertStencilToCslStencilPass(ModulePass):
    name = "convert-stencil-to-csl-stencil"

    # chunks into which to slice communication
    num_chunks: int = 1

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        ConvertArithToVarithPass().apply(ctx, op)
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertSwapToPrefetchPattern(num_chunks=self.num_chunks),
                    ConvertAccessOpPattern(),
                ]
            ),
            walk_reverse=False,
            apply_recursively=False,
        )
        module_pass.rewrite_module(op)
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ConvertApplyOpPattern(num_chunks=self.num_chunks),
                    PromoteCoefficients(),
                    TransformPrefetch(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        ).rewrite_module(op)

        ConvertVarithToArithPass().apply(ctx, op)

        if self.num_chunks > 1:
            BackpropagateStencilShapes().apply(ctx, op)
