from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import arith, bufferization, func, linalg, memref, stencil, tensor
from xdsl.dialects.builtin import (
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
        for arg in op.args:
            if isa(arg.type, TensorType[Attribute]):
                to_memrefs.append(new_arg := to_memref_op(arg))
                buf_args.append(new_arg.memref)
            else:
                buf_args.append(arg)

        # create new op
        buf_apply_op = csl_stencil.ApplyOp(
            operands=[op.field, buf_iter_arg.memref, op.args, op.dest],
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

        assert isa(typ := op.receive_chunk.block.args[0].type, TensorType[Attribute])
        chunk_type = TensorType(typ.get_element_type(), typ.get_shape()[1:])

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

        self._inject_iter_arg_into_linalg_outs(
            buf_apply_op, rewriter, chunk_type, chunk_region_arg_mapping[2]
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

    @staticmethod
    def _inject_iter_arg_into_linalg_outs(
        op: csl_stencil.ApplyOp,
        rewriter: PatternRewriter,
        chunk_type: TensorType[Attribute],
        iter_arg: SSAValue,
    ):
        """
        Finds a linalg op with `chunk_type` shape in `outs` and injects
        an extracted slice of `iter_arg`. This is a work-around for the
        way bufferization works, causing it to use `iter_arg` as an accumulator
        and avoiding having an extra alloc + memref.copy.
        """
        linalg_op: linalg.NamedOpBase | None = None
        for curr_op in op.receive_chunk.block.ops:
            if (
                isinstance(curr_op, linalg.NamedOpBase)
                and len(curr_op.outputs) > 0
                and curr_op.outputs.types[0] == chunk_type
            ):
                linalg_op = curr_op
                break

        if linalg_op is None:
            return

        rewriter.replace_op(
            linalg_op,
            [
                extract_slice_op := tensor.ExtractSliceOp(
                    operands=[iter_arg, [op.receive_chunk.block.args[1]], [], []],
                    result_types=[chunk_type],
                    properties={
                        "static_offsets": DenseArrayBase.from_list(
                            i64, (memref.Subview.DYNAMIC_INDEX,)
                        ),
                        "static_sizes": DenseArrayBase.from_list(
                            i64, chunk_type.get_shape()
                        ),
                        "static_strides": DenseArrayBase.from_list(i64, (1,)),
                    },
                ),
                type(linalg_op).build(
                    operands=[linalg_op.inputs, extract_slice_op.results],
                    result_types=linalg_op.result_types,
                    properties=linalg_op.properties,
                    attributes=linalg_op.attributes,
                    regions=[linalg_op.detach_region(r) for r in linalg_op.regions],
                ),
            ],
        )

    @staticmethod
    def _build_extract_slice(
        op: csl_stencil.ApplyOp, to_tensor: bufferization.ToTensorOp, offset: SSAValue
    ) -> tensor.ExtractSliceOp:
        """
        Helper function to create an early tensor.extract_slice in the apply.recv_chunk_cb region needed for bufferization.
        """

        # this is the unbufferized `tensor<(neighbours)x(ZDim)x(type)>` value
        assert isa(typ := op.receive_chunk.block.args[0].type, TensorType[Attribute])

        return tensor.ExtractSliceOp(
            operands=[to_tensor.tensor, [offset], [], []],
            result_types=[TensorType(typ.get_element_type(), typ.get_shape()[1:])],
            properties={
                "static_offsets": DenseArrayBase.from_list(
                    i64, (memref.Subview.DYNAMIC_INDEX,)
                ),
                "static_sizes": DenseArrayBase.from_list(i64, typ.get_shape()[1:]),
                "static_strides": DenseArrayBase.from_list(i64, (1,)),
            },
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
                ]
            )
        )
        module_pass.rewrite_module(op)
