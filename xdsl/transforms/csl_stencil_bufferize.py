from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, bufferization, func, linalg, memref, stencil, tensor
from xdsl.dialects.builtin import (
    AnyTensorType,
    AnyTensorTypeConstr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    FunctionType,
    ModuleOp,
    TensorType,
    i64,
)
from xdsl.dialects.csl import csl_stencil
from xdsl.ir import Attribute, Block, BlockArgument, Operation, Region, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa
from xdsl.utils.isattr import isattr


def tensor_to_memref_type(t: TensorType[Attribute]) -> memref.MemRefType[Attribute]:
    """Type conversion from tensor to memref."""
    return memref.MemRefType(t.get_element_type(), t.get_shape())


def to_memref_op(op: SSAValue) -> bufferization.ToMemrefOp:
    """Creates a `bufferization.to_memref` operation."""
    assert isa(op.type, TensorType[Attribute])
    r_type = memref.MemRefType(
        op.type.get_element_type(), op.type.get_shape()
    )  # todo set strided+offset here?
    return bufferization.ToMemrefOp(operands=[op], result_types=[r_type])


def to_tensor_op(
    op: SSAValue, writable: bool = False, restrict: bool = True
) -> bufferization.ToTensorOp:
    """Creates a `bufferization.to_tensor` operation."""
    assert isa(op.type, memref.MemRefType[Attribute])
    return bufferization.ToTensorOp(op, restrict, writable)


class StencilTypeConversion(TypeConversionPattern):
    """
    Converts from tensorised stencil.field types to memref by extracting the element type which is a tensor
    and converting it to memref.

    For instance:
        `!stencil.field<[-1,1023]x[-1,511]xtensor<512xf32>` to `memref<512xf32>`
    """

    @attr_type_rewrite_pattern
    def convert_type(
        self, typ: stencil.FieldType[TensorType[Attribute]]
    ) -> memref.MemRefType[Attribute]:
        # todo should this convert to `memref` or `stencil.field<..xmemref<..>>`?
        return tensor_to_memref_type(typ.get_element_type())


@dataclass(frozen=True)
class ApplyOpBufferize(RewritePattern):
    """
    Bufferizes csl_stencil.apply, rewriting args and block args, changing them from tensor to memref types.
    For each converted arg, creates a `bufferization.to_memref` before the apply op.
    For each converted block arg, creates a `bufferization.to_tensor` at the start of the block.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        if isa(op.accumulator.type, memref.MemRefType[Attribute]):
            return

        # convert args
        buf_args: list[SSAValue] = []
        to_memrefs: list[Operation] = [buf_iter_arg := to_memref_op(op.accumulator)]
        for arg in [*op.args_rchunk, *op.args_dexchng]:
            if isa(arg.type, TensorType[Attribute]):
                to_memrefs.append(new_arg := to_memref_op(arg))
                buf_args.append(new_arg.memref)
            else:
                buf_args.append(arg)

        # create new op
        buf_apply_op = csl_stencil.ApplyOp(
            operands=[
                op.field,
                buf_iter_arg.memref,
                op.args_rchunk,
                op.args_dexchng,
                op.dest,
            ],
            result_types=op.res.types or [[]],
            regions=[
                self._get_empty_bufferized_region(op.receive_chunk.block.args),
                self._get_empty_bufferized_region(op.done_exchange.block.args),
            ],
            properties=op.properties,
            attributes=op.attributes,
        )

        # insert to_tensor ops and create arg mappings for block inlining
        chunk_region_arg_mapping: Sequence[SSAValue] = []
        for idx, (old_arg, arg) in enumerate(
            zip(op.receive_chunk.block.args, buf_apply_op.receive_chunk.block.args)
        ):
            # arg0 has special meaning and does not need a `to_tensor` op
            if isattr(old_arg.type, TensorType) and idx != 0:
                rewriter.insert_op(
                    # ensure iter_arg is writable
                    t := to_tensor_op(arg, writable=idx == 2),
                    InsertPoint.at_end(buf_apply_op.receive_chunk.block),
                )
                chunk_region_arg_mapping.append(t.tensor)
            else:
                chunk_region_arg_mapping.append(arg)

        done_exchange_arg_mapping: Sequence[SSAValue] = []
        for idx, (old_arg, arg) in enumerate(
            zip(op.done_exchange.block.args, buf_apply_op.done_exchange.block.args)
        ):
            if isattr(old_arg.type, TensorType):
                rewriter.insert_op(
                    # ensure iter_arg is writable
                    t := to_tensor_op(arg, writable=idx == 1),
                    InsertPoint.at_end(buf_apply_op.done_exchange.block),
                )
                done_exchange_arg_mapping.append(t.tensor)
            else:
                done_exchange_arg_mapping.append(arg)

        # inline blocks from old into new regions
        rewriter.inline_block(
            op.receive_chunk.block,
            InsertPoint.at_end(buf_apply_op.receive_chunk.block),
            chunk_region_arg_mapping,
        )

        rewriter.inline_block(
            op.done_exchange.block,
            InsertPoint.at_end(buf_apply_op.done_exchange.block),
            done_exchange_arg_mapping,
        )

        # insert new op
        rewriter.replace_matched_op(new_ops=[*to_memrefs, buf_apply_op])

    @staticmethod
    def _get_empty_bufferized_region(args: Sequence[BlockArgument]) -> Region:
        """Helper function to create a new region with bufferized arg types."""
        return Region(
            Block(
                arg_types=[
                    (
                        tensor_to_memref_type(arg.type)
                        if isattr(arg.type, AnyTensorTypeConstr)
                        else arg.type
                    )
                    for arg in args
                ]
            )
        )


@dataclass(frozen=True)
class AccessOpBufferize(RewritePattern):
    """
    Bufferizes AccessOp.

    The type conversion pass creates the scenario that some `csl_stencil.access` ops are equal input and output types,
    for instance, `(memref<512xf32>) -> memref<512xf32>`. This only happens for ops accessing own data. In this case,
    the access op has no effect and can safely be folded away.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.AccessOp, rewriter: PatternRewriter, /):
        if not isa(op.result.type, TensorType[Attribute]):
            return
        r_type = tensor_to_memref_type(op.result.type)

        # accesses to own data that (after bufferization) have the same input and output type can be safely folded away
        if op.op.type == r_type and all(o == 0 for o in op.offset):
            rewriter.replace_matched_op(to_tensor_op(op.op))
            return

        rewriter.replace_matched_op(
            [
                access := csl_stencil.AccessOp(
                    op.op,
                    op.offset,
                    r_type,
                    op.offset_mapping,
                ),
                to_tensor_op(access.result),
            ]
        )


@dataclass(frozen=True)
class YieldOpBufferize(RewritePattern):
    """Bufferizes YieldOp."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.YieldOp, rewriter: PatternRewriter, /):
        to_memrefs: list[Operation] = []
        args: list[SSAValue] = []
        for arg in op.arguments:
            if isa(arg.type, TensorType[Attribute]):
                to_memrefs.append(new_arg := to_memref_op(arg))
                args.append(new_arg.memref)
            else:
                args.append(arg)

        if len(to_memrefs) == 0:
            return

        rewriter.replace_matched_op([*to_memrefs, csl_stencil.YieldOp(*args)])


@dataclass(frozen=True)
class FuncOpBufferize(RewritePattern):
    """
    Replace the function_type and let a separate type conversion pass handle the block args.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        function_type = FunctionType.from_lists(
            [
                (
                    tensor_to_memref_type(t.get_element_type())
                    if isa(t, stencil.FieldType[TensorType[Attribute]])
                    else t
                )
                for t in op.function_type.inputs
            ],
            [
                (
                    tensor_to_memref_type(t.get_element_type())
                    if isa(t, stencil.FieldType[TensorType[Attribute]])
                    else t
                )
                for t in op.function_type.outputs
            ],
        )
        if function_type == op.function_type:
            return
        rewriter.replace_matched_op(
            func.FuncOp.build(
                operands=op.operands,
                result_types=op.result_types,
                regions=[op.detach_region(op.body)],
                properties={**op.properties, "function_type": function_type},
                attributes=op.attributes.copy(),
            )
        )


@dataclass(frozen=True)
class ArithConstBufferize(RewritePattern):
    """
    Bufferize arith tensor constants to prevent mlir bufferize from promoting them to globals.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: arith.Constant, rewriter: PatternRewriter, /):
        if not isa(op.result.type, TensorType[Attribute]):
            return
        assert isinstance(op.value, DenseIntOrFPElementsAttr)
        assert isa(op.value.type, TensorType[Attribute])
        typ = DenseIntOrFPElementsAttr(
            [tensor_to_memref_type(op.value.type), op.value.data]
        )
        rewriter.replace_matched_op(
            [
                c := arith.Constant(typ),
                to_tensor_op(c.result),
            ]
        )


@dataclass(frozen=True)
class LinalgAccumulatorInjection(RewritePattern):
    """
    Injects the `accumulator` block argument for linalg ops within the csl_stencil.apply regions
    into the `outs` argument. This typically reduces the overhead of bufferization.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: linalg.NamedOpBase | linalg.Generic, rewriter: PatternRewriter, /
    ):
        # check if there is an output to inject the accumulator into
        if len(op.outputs) != 1 or not isa(
            target_t := op.outputs[0].type, AnyTensorType
        ):
            return

        # find parent `csl_stencil.apply` and which of the regions `op` is in
        apply, region = self.get_apply_and_region(op)
        if not apply or not region:
            return

        # retrieve the correct accumulator block arg
        if region == apply.receive_chunk:
            acc_block_arg = region.block.args[2]
        elif region == apply.done_exchange:
            acc_block_arg = region.block.args[1]
        else:
            raise ValueError("Invalid region")

        # fetch the bufferization of the accumulator block arg
        acc_uses = [
            use.operation
            for use in acc_block_arg.uses
            if isinstance(use.operation, bufferization.ToTensorOp)
        ]
        if len(acc_uses) < 1:
            return
        acc_bufferization = acc_uses[0]

        # in the `chunk_recieved` region, fetch or create a down-sized chunk of the accumulator
        if acc_bufferization.tensor.type != target_t and region == apply.receive_chunk:
            # check if we can find an `extract_slice` to the desired size
            extract_slice = None
            for use in acc_bufferization.tensor.uses:
                if (
                    isinstance(use.operation, tensor.ExtractSliceOp)
                    and use.operation.result.type == target_t
                ):
                    extract_slice = use.operation
                    break

            # create `extract_slice` op if none exists
            if not extract_slice:
                extract_slice = tensor.ExtractSliceOp(
                    operands=[acc_bufferization, [region.block.args[1]], [], []],
                    result_types=[target_t],
                    properties={
                        "static_offsets": DenseArrayBase.from_list(
                            i64, (memref.Subview.DYNAMIC_INDEX,)
                        ),
                        "static_sizes": DenseArrayBase.from_list(
                            i64, target_t.get_shape()
                        ),
                        "static_strides": DenseArrayBase.from_list(i64, (1,)),
                    },
                )
                rewriter.insert_op(extract_slice, InsertPoint.after(acc_bufferization))

            # use the `extract_slice` op fetched or created when rebuilding `op`
            acc_bufferization = extract_slice

        # check if `op` can be rebuild and needs to be rebuilt
        if (
            acc_bufferization.results[0].type == target_t
            and acc_bufferization.results[0] != op.outputs[0]
        ):
            rewriter.replace_matched_op(
                type(op).build(
                    operands=[op.inputs, acc_bufferization],
                    result_types=op.result_types,
                    properties=op.properties,
                    attributes=op.attributes,
                    regions=[op.detach_region(r) for r in op.regions],
                ),
            )

    @staticmethod
    def get_apply_and_region(
        op: Operation,
    ) -> tuple[csl_stencil.ApplyOp, Region] | tuple[None, None]:
        p_region = op.parent_region()
        apply = None
        while (
            p_region
            and (apply := p_region.parent_op())
            and not isinstance(apply, csl_stencil.ApplyOp)
        ):
            p_region = apply.parent_region()
        if not isinstance(apply, csl_stencil.ApplyOp) or not p_region:
            return None, None
        return apply, p_region


@dataclass(frozen=True)
class CslStencilBufferize(ModulePass):
    """
    Bufferizes the csl_stencil dialect.

    Attempts to inject `csl_stencil.apply.recv_chunk_cb.accumulator` into linalg compute ops `outs` within that region
    for improved bufferization. Ideally be run after `--lift-arith-to-linalg`.
    """

    name = "csl-stencil-bufferize"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        module_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    StencilTypeConversion(),
                    ApplyOpBufferize(),
                    AccessOpBufferize(),
                    YieldOpBufferize(),
                    FuncOpBufferize(),
                    ArithConstBufferize(),
                    LinalgAccumulatorInjection(),
                ]
            )
        )
        module_pass.rewrite_module(op)
