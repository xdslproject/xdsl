from dataclasses import dataclass, field
from typing import TypeVar, Iterable, Callable, cast, ClassVar

from xdsl.builder import Builder
from xdsl.passes import ModulePass

from xdsl.utils.hints import isa
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import Rewriter
from xdsl.ir import (
    BlockArgument,
    MLContext,
    Operation,
    SSAValue,
    Block,
    Region,
    OpResult,
)
from xdsl.irdl import Attribute
from xdsl.dialects import builtin, mpi, memref, arith, scf, func
from xdsl.dialects.experimental import stencil, dmp

from xdsl.transforms.experimental.StencilShapeInference import StencilShapeInferencePass

from xdsl.transforms.experimental.dmp.decompositions import (
    DomainDecompositionStrategy,
    HorizontalSlices2D,
    GridSlice2d,
)

_T = TypeVar("_T", bound=Attribute)


@dataclass
class ChangeStoreOpSizes(RewritePattern):
    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        assert all(
            integer_attr.data == 0 for integer_attr in op.lb.array.data
        ), "lb must be 0"
        shape: tuple[int, ...] = tuple(
            (integer_attr.data for integer_attr in op.ub.array.data)
        )
        new_shape = self.strategy.calc_resize(shape)
        op.ub = stencil.IndexAttr.get(*new_shape)


@dataclass
class AddHaloExchangeOps(RewritePattern):
    """
    This rewrite adds a `stencil.halo_exchange` after each `stencil.load` op
    """

    strategy: DomainDecompositionStrategy

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.LoadOp, rewriter: PatternRewriter, /):
        swap_op = dmp.HaloSwapOp.get(op.res)
        swap_op.nodes = self.strategy.comm_layout()
        rewriter.insert_op_after_matched_op(swap_op)


@dataclass
class LowerHaloExchangeToMpi(RewritePattern):
    init: bool

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: dmp.HaloSwapOp, rewriter: PatternRewriter, /):
        assert op.swaps is not None
        assert op.nodes is not None
        exchanges = list(op.swaps)

        assert isa(op.input_stencil.typ, memref.MemRefType[Attribute])

        rewriter.replace_matched_op(
            list(
                generate_mpi_calls_for(
                    op.input_stencil,
                    exchanges,
                    op.input_stencil.typ.element_type,
                    op.nodes,
                    emit_init=self.init,
                )
            ),
            [],
        )


def generate_mpi_calls_for(
    source: SSAValue,
    exchanges: list[dmp.HaloExchangeDecl],
    dtype: Attribute,
    grid: dmp.NodeGrid,
    emit_init: bool = True,
) -> Iterable[Operation]:
    # call mpi init (this will be hoisted to function level)
    if emit_init:
        yield mpi.Init()
    # allocate request array
    # we need two request objects per exchange
    # one for the send, one for the recv
    req_cnt = arith.Constant.from_int_and_width(len(exchanges) * 2, builtin.i32)
    reqs = mpi.AllocateTypeOp.get(mpi.RequestType, req_cnt)
    # get comm rank
    rank = mpi.CommRank.get()
    # define static tag of 0
    # TODO: what is tag?
    tag = arith.Constant.from_int_and_width(0, builtin.i32)

    yield from (req_cnt, reqs, rank, tag)

    recv_buffers: list[tuple[dmp.HaloExchangeDecl, memref.Alloc, SSAValue]] = []

    for i, ex in enumerate(exchanges):
        # TODO: handle multi-d grids
        neighbor_offset = arith.Constant.from_int_and_width(ex.neighbor[0], builtin.i32)
        neighbor_rank = arith.Addi(rank, neighbor_offset)
        yield from (neighbor_offset, neighbor_rank)

        # generate a temp buffer to store the data in
        alloc_outbound = memref.Alloc.get(dtype, 64, [ex.elem_count])
        alloc_outbound.memref.name_hint = f"send_buff_ex{i}"
        alloc_inbound = memref.Alloc.get(dtype, 64, [ex.elem_count])
        alloc_inbound.memref.name_hint = f"recv_buff_ex{i}"
        yield from (alloc_outbound, alloc_inbound)

        # boundary condition:
        # TODO: handle non-1d layouts
        assert len(ex.neighbor) == 1
        bound = arith.Constant.from_int_and_width(
            0 if ex.neighbor[0] < 0 else grid.as_tuple()[0], builtin.i32
        )
        # comparison == true <=> we have a valid dest rank
        comparison = "sge" if ex.neighbor[0] < 0 else "slt"

        cond_val = arith.Cmpi.get(neighbor_rank, bound, comparison)
        yield from (bound, cond_val)

        recv_buffers.append((ex, alloc_inbound, cond_val.result))

        # get two unique indices
        cst_i = arith.Constant.from_int_and_width(i, builtin.i32)
        cst_in = arith.Constant.from_int_and_width(i + len(exchanges), builtin.i32)
        yield from (cst_i, cst_in)
        # from these indices, get request objects
        req_send = mpi.VectorGetOp.get(reqs, cst_i)
        req_recv = mpi.VectorGetOp.get(reqs, cst_in)
        yield from (req_send, req_recv)

        def then() -> Iterable[Operation]:
            # copy source area to outbound buffer
            yield from generate_memcpy(source, ex.source_area(), alloc_outbound.memref)
            # get ptr, count, dtype
            unwrap_out = mpi.UnwrapMemrefOp.get(alloc_outbound)
            unwrap_out.ptr.name_hint = f"send_buff_ex{i}_ptr"
            yield unwrap_out

            # isend call
            yield mpi.Isend.get(
                unwrap_out.ptr,
                unwrap_out.len,
                unwrap_out.typ,
                neighbor_rank,
                tag,
                req_send,
            )

            # get ptr for receive buffer
            unwrap_in = mpi.UnwrapMemrefOp.get(alloc_inbound)
            unwrap_in.ptr.name_hint = f"recv_buff_ex{i}_ptr"
            yield unwrap_in
            # Irecv call
            yield mpi.Irecv.get(
                unwrap_in.ptr,
                unwrap_in.len,
                unwrap_in.typ,
                neighbor_rank,
                tag,
                req_recv,
            )
            yield scf.Yield.get()

        def else_() -> Iterable[Operation]:
            # set the request object to MPI_REQUEST_NULL s.t. they are ignored
            # in the waitall call
            yield mpi.NullRequestOp.get(req_send)
            yield mpi.NullRequestOp.get(req_recv)
            yield scf.Yield.get()

        yield scf.If.get(
            cond_val,
            [],
            Region([Block(then())]),
            Region([Block(else_())]),
        )

    # wait for all calls to complete
    yield mpi.Waitall.get(reqs.result, req_cnt.result)

    # start shuffling data into the main memref again
    for ex, buffer, cond_val in recv_buffers:
        yield scf.If.get(
            cond_val,
            [],
            Region(
                [
                    Block(
                        list(
                            generate_memcpy(
                                source,
                                ex,
                                buffer.memref,
                                reverse=True,
                            )
                        )
                        + [scf.Yield.get()]
                    )
                ]
            ),
            Region([Block([scf.Yield.get()])]),
        )


def generate_memcpy(
    source: SSAValue, ex: dmp.HaloExchangeDecl, dest: SSAValue, reverse: bool = False
) -> list[Operation]:
    """
    This function generates a memcpy routine to copy over the parts
    specified by the `ex` from `source` into `dest`.

    If reverse=True, it instead copy from `dest` into the parts of
    `source` as specified by `ex`

    """
    assert ex.dim == 2, "Cannot handle non-2d case of memcpy yet!"

    idx = builtin.IndexType()

    x0 = arith.Constant.from_int_and_width(ex.offset[0], idx)
    x0.result.name_hint = "x0"
    y0 = arith.Constant.from_int_and_width(ex.offset[1], idx)
    y0.result.name_hint = "y0"
    x_len = arith.Constant.from_int_and_width(ex.size[0], idx)
    x_len.result.name_hint = "x_len"
    y_len = arith.Constant.from_int_and_width(ex.size[1], idx)
    y_len.result.name_hint = "y_len"
    cst0 = arith.Constant.from_int_and_width(0, idx)
    cst1 = arith.Constant.from_int_and_width(1, idx)

    @Builder.implicit_region([idx, idx])
    def loop_body(args: tuple[BlockArgument, ...]):
        """
        Loop body of the scf.parallel() that iterates the following domain:
        i = (0->x_len), y = (0->y_len)
        """
        i, j = args
        i.name_hint = "i"
        j.name_hint = "j"

        # x = i + x0
        # y = i + y0
        # TODO: proper fix this (probs in HaloDimsHelper)
        x_ = arith.Addi(i, x0)
        x = arith.Addi(x_, cst1)
        y_ = arith.Addi(j, y0)
        y = arith.Addi(y_, cst1)

        x.result.name_hint = "x"
        y.result.name_hint = "y"

        # linearized_idx = (j * x_len) + i
        dest_idx = arith.Muli(j, x_len)
        linearized_idx = arith.Addi(dest_idx, i)
        linearized_idx.result.name_hint = "linearized_idx"

        if reverse:
            load = memref.Load.get(dest, [linearized_idx])
            memref.Store.get(load, source, [x, y])
        else:
            load = memref.Load.get(source, [x, y])
            memref.Store.get(load, dest, [linearized_idx])

        scf.Yield.get()

    loop = scf.ParallelOp.get(
        [cst0, cst0],
        [x_len, y_len],
        [cst1, cst1],
        loop_body,
        [],
    )

    return [
        x0,
        y0,
        x_len,
        y_len,
        cst0,
        cst1,
        loop,
    ]


class MpiLoopInvariantCodeMotion:
    """
    THIS IS NOT A REWRITE PATTERN!

    This is a two-stage rewrite that modifies operations in a manner
    that is incompatible with the PatternRewriter!

    It implements a custom rewrite_module() method directly
    on the class.

    This rewrite moves all memref.allo, mpi.comm.rank, mpi.allocate
    and mpi.unwrap_memref ops and moves them "up" until it hits a
    func.func, and then places them *before* the op they appear in.
    """

    seen_ops: set[Operation]
    has_init: set[func.FuncOp]

    def __init__(self):
        self.seen_ops = set()
        self.has_init = set()

    def rewrite(
        self,
        op: memref.Alloc
        | mpi.CommRank
        | mpi.AllocateTypeOp
        | mpi.UnwrapMemrefOp
        | mpi.Init,
        rewriter: Rewriter,
        /,
    ):
        if op in self.seen_ops:
            return
        self.seen_ops.add(op)

        # memref unwraps can always be moved to their allocation
        if isinstance(op, mpi.UnwrapMemrefOp) and isinstance(
            op.ref.owner, memref.Alloc
        ):
            op.detach()
            rewriter.insert_op_after(op.ref.owner, op)
            return

        base = op
        parent = op.parent_op()
        # walk upwards until we hit a function
        while parent is not None and not isinstance(parent, func.FuncOp):
            base = parent
            parent = base.parent_op()

        # check that we did not run into "nowhere"
        assert parent is not None, "Expected MPI to be inside a func.FuncOp!"
        assert isinstance(parent, func.FuncOp)  # this must be true now

        # check that we "ascended"
        if base == op:
            return

        if not can_loop_invariant_code_move(op):
            return

        # if we move an mpi.init, generate a finalize()!
        if isinstance(op, mpi.Init):
            # ignore multiple inits
            if parent in self.has_init:
                rewriter.erase_op(op)
                return
            self.has_init.add(parent)
            # add a finalize() call to the end of the function
            block = parent.regions[0].blocks[-1]
            return_op = block.last_op
            assert return_op is not None
            rewriter.insert_op_before(return_op, mpi.Finalize())

        ops = list(collect_args_recursive(op))
        for found_op in ops:
            found_op.detach()
            rewriter.insert_op_before(base, found_op)

    def get_matcher(
        self,
        worklist: list[
            memref.Alloc
            | mpi.CommRank
            | mpi.AllocateTypeOp
            | mpi.UnwrapMemrefOp
            | mpi.Init
        ],
    ) -> Callable[[Operation], None]:
        """
        Returns a match() function that adds methods to a worklist
        if they satisfy some criteria.
        """

        def match(op: Operation):
            if isinstance(
                op,
                (
                    memref.Alloc,
                    mpi.CommRank,
                    mpi.AllocateTypeOp,
                    mpi.UnwrapMemrefOp,
                    mpi.Init,
                ),
            ):
                worklist.append(op)

        return match

    def rewrite_module(self, op: builtin.ModuleOp):
        """
        Apply the rewrite to a module.

        We do a two-stage rewrite because we are modifying
        the operations we loop on them, which would throw of `op.walk`.
        """
        # collect all ops that should be rewritten
        worklist: list[
            memref.Alloc
            | mpi.CommRank
            | mpi.AllocateTypeOp
            | mpi.UnwrapMemrefOp
            | mpi.Init
        ] = list()
        matcher = self.get_matcher(worklist)
        for o in op.walk():
            matcher(o)

        # rewrite ops
        rewriter = Rewriter()
        for matched_op in worklist:
            self.rewrite(matched_op, rewriter)


_LOOP_INVARIANT_OPS = (arith.Constant, arith.Addi, arith.Muli)


def can_loop_invariant_code_move(op: Operation):
    """
    This function walks the def-use chain up to see if all the args are
    "constant enough" to move outside the loop.

    This check is very conservative, but that means it definitely works!
    """

    for arg in op.operands:
        if not isinstance(arg, OpResult):
            print("{} is not opresult".format(arg))
            return False
        if not isinstance(arg.owner, _LOOP_INVARIANT_OPS):
            print("{} is not loop invariant".format(arg))
            return False
        if not can_loop_invariant_code_move(arg.owner):
            return False
    return True


def collect_args_recursive(op: Operation) -> Iterable[Operation]:
    """
    Collect the def-use chain "upwards" of an operation.
    Check with can_loop_invariant_code_move prior to using this!
    """
    for arg in op.operands:
        assert isinstance(arg, OpResult)
        yield from collect_args_recursive(arg.owner)
    yield op


@dataclass
class DmpSwapShapeInference:
    """
    Not a rewrite pattern, as it's a bit more involved.

    This is applied after stencil shape inference has run. It will find the
    HaloSwapOps again, and use the results of the shape inference pass
    to attach the swap declarations.
    """

    strategy: DomainDecompositionStrategy
    rewriter: Rewriter = field(default_factory=Rewriter)

    def match_and_rewrite(self, op: dmp.HaloSwapOp):
        core_lb: stencil.IndexAttr | None = None
        core_ub: stencil.IndexAttr | None = None

        for use in op.input_stencil.uses:
            if not isinstance(use.operation, stencil.ApplyOp):
                continue
            assert use.operation.res
            res_typ = cast(stencil.TempType[Attribute], use.operation.res[0].typ)
            assert isinstance(res_typ.bounds, stencil.StencilBoundsAttr)
            core_lb = res_typ.bounds.lb
            core_ub = res_typ.bounds.ub
            break

        # this shouldn't have changed since the op was created!
        temp = op.input_stencil.typ
        assert isa(temp, stencil.TempType[Attribute])
        assert isinstance(temp.bounds, stencil.StencilBoundsAttr)
        buff_lb = temp.bounds.lb
        buff_ub = temp.bounds.ub

        # fun fact: pyright does not understand this:
        # assert None not in (core_lb, core_ub, buff_lb, buff_ub)
        assert core_lb is not None
        assert core_ub is not None
        assert buff_lb is not None
        assert buff_ub is not None

        op.swaps = builtin.ArrayAttr(
            self.strategy.halo_exchange_defs(
                dmp.HaloShapeInformation.from_index_attrs(
                    buff_lb=buff_lb,
                    core_lb=core_lb,
                    buff_ub=buff_ub,
                    core_ub=core_ub,
                )
            )
        )

    def apply(self, module: builtin.ModuleOp):
        for op in module.walk():
            if isinstance(op, dmp.HaloSwapOp):
                self.match_and_rewrite(op)


@dataclass
class DmpDecompositionPass(ModulePass):
    """
    Represents a pass that takes a strategy as input
    """

    STRATEGIES: ClassVar[dict[str, type[DomainDecompositionStrategy]]] = {
        "2d-horizontal": HorizontalSlices2D,
        "2d-grid": GridSlice2d,
    }

    name = "dmp-decompose-2d"

    strategy: str

    slices: list[int]
    """
    Number of slices to decompose the input into
    """

    def get_strategy(self) -> DomainDecompositionStrategy:
        if self.strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self.STRATEGIES[self.strategy](self.slices)


@dataclass
class GlobalStencilToLocalStencil2DHorizontal(DmpDecompositionPass):
    """
    Decompose a stencil to apply to a local domain.

    This pass *replaces* stencil shape inference in a
    pass pipeline!
    """

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        strategy = self.get_strategy()

        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ChangeStoreOpSizes(strategy),
                    AddHaloExchangeOps(strategy),
                ]
            ),
            apply_recursively=False,
        ).rewrite_module(op)

        # run the shape inference pass
        StencilShapeInferencePass().apply(ctx, op)

        DmpSwapShapeInference(strategy).apply(op)


@dataclass
class LowerHaloToMPI(ModulePass):
    name = "dmp-to-mpi"

    mpi_init: bool = True

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerHaloExchangeToMpi(
                        self.mpi_init,
                    ),
                ]
            )
        ).rewrite_module(op)
        MpiLoopInvariantCodeMotion().rewrite_module(op)
