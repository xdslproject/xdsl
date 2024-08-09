from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import MLContext
from xdsl.dialects import bufferization, func, memref, stencil
from xdsl.dialects.builtin import FunctionType, ModuleOp, TensorType
from xdsl.dialects.csl import csl_stencil
from xdsl.ir import Attribute, Block, Operation, Region, SSAValue
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


def tensor_to_memref(t: TensorType[Attribute]) -> memref.MemRefType[Attribute]:
    return memref.MemRefType(t.get_element_type(), t.get_shape())


def get_to_memref(op: SSAValue) -> bufferization.ToMemrefOp:
    assert isa(op.type, TensorType[Attribute])
    r_type = memref.MemRefType(
        op.type.get_element_type(), op.type.get_shape()
    )  # todo strided+offset
    return bufferization.ToMemrefOp(operands=[op], result_types=[r_type])


def get_to_tensor(
    op: SSAValue, writable: bool = False, restrict: bool = True
) -> bufferization.ToTensorOp:
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
        # todo should this convert to memref?
        return tensor_to_memref(typ.get_element_type())


@dataclass(frozen=True)
class ApplyOpBufferize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.ApplyOp, rewriter: PatternRewriter, /):
        if isa(op.iter_arg.type, memref.MemRefType[Attribute]):
            return
        buf_args: Sequence[SSAValue] = []
        to_memrefs: Sequence[Operation] = [buf_iter_arg := get_to_memref(op.iter_arg)]

        for arg in op.args:
            if isa(arg.type, TensorType[Attribute]):
                to_memrefs.append(new_arg := get_to_memref(arg))
                buf_args.append(new_arg.memref)
            else:
                buf_args.append(arg)

        chunk_reduce_translate_idxs: Sequence[int] = []
        chunk_reduce_args: Sequence[Attribute] = []
        for idx, arg in enumerate(op.chunk_reduce.block.args):
            if isa(arg.type, TensorType[Attribute]):
                if idx != 0:
                    chunk_reduce_translate_idxs.append(idx)
                chunk_reduce_args.append(tensor_to_memref(arg.type))
            else:
                chunk_reduce_args.append(arg.type)

        post_process_translate_idxs: Sequence[int] = []
        post_process_args: Sequence[Attribute] = []
        for idx, arg in enumerate(op.post_process.block.args):
            if isa(arg.type, TensorType[Attribute]):
                post_process_translate_idxs.append(idx)
                post_process_args.append(tensor_to_memref(arg.type))
            else:
                post_process_args.append(arg.type)

        buf_apply_op = csl_stencil.ApplyOp(
            operands=[op.communicated_stencil, buf_iter_arg.memref, op.args, op.dest],
            result_types=[t.type for t in op.results] or [[]],
            regions=[
                Region(Block(arg_types=chunk_reduce_args)),
                Region(Block(arg_types=post_process_args)),
            ],
            properties=op.properties,
            attributes=op.attributes,
        )

        chunk_reduce_arg_mapping: Sequence[SSAValue] = []
        for idx, arg in enumerate(buf_apply_op.chunk_reduce.block.args):
            if idx in chunk_reduce_translate_idxs:
                rewriter.insert_op(
                    t := get_to_tensor(arg),
                    InsertPoint.at_end(buf_apply_op.chunk_reduce.block),
                )
                chunk_reduce_arg_mapping.append(t.tensor)
            else:
                chunk_reduce_arg_mapping.append(arg)

        post_process_arg_mapping: Sequence[SSAValue] = []
        for idx, arg in enumerate(buf_apply_op.post_process.block.args):
            if idx in post_process_translate_idxs:
                rewriter.insert_op(
                    t := get_to_tensor(arg),
                    InsertPoint.at_end(buf_apply_op.post_process.block),
                )
                post_process_arg_mapping.append(t.tensor)
            else:
                post_process_arg_mapping.append(arg)

        rewriter.inline_block(
            op.chunk_reduce.block,
            InsertPoint.at_end(buf_apply_op.chunk_reduce.block),
            chunk_reduce_arg_mapping,
        )

        rewriter.inline_block(
            op.post_process.block,
            InsertPoint.at_end(buf_apply_op.post_process.block),
            post_process_arg_mapping,
        )

        rewriter.replace_matched_op(new_ops=[*to_memrefs, buf_apply_op])


@dataclass(frozen=True)
class AccessOpBufferize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.AccessOp, rewriter: PatternRewriter, /):
        if not isa(op.result.type, TensorType[Attribute]):
            return
        rewriter.replace_matched_op(
            [
                access := csl_stencil.AccessOp(
                    op.op,
                    op.offset,
                    tensor_to_memref(op.result.type),
                    op.offset_mapping,
                ),
                get_to_tensor(access.result),
            ]
        )


@dataclass(frozen=True)
class YieldOpBufferize(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: csl_stencil.YieldOp, rewriter: PatternRewriter, /):
        to_memrefs: list[Operation] = []
        args: list[SSAValue] = []
        for arg in op.arguments:
            if isa(arg.type, TensorType[Attribute]):
                to_memrefs.append(new_arg := get_to_memref(arg))
                args.append(new_arg.memref)
            else:
                args.append(arg)

        if len(to_memrefs) == 0:
            return

        rewriter.replace_matched_op([*to_memrefs, csl_stencil.YieldOp(*args)])


@dataclass(frozen=True)
class FuncOpBufferize(RewritePattern):
    """
    Replace the function_type and let the type conversion pass handle the block args.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        op.function_type = FunctionType.from_lists(
            [
                (
                    tensor_to_memref(t.get_element_type())
                    if isa(t, stencil.FieldType[TensorType[Attribute]])
                    else t
                )
                for t in op.function_type.inputs
            ],
            [
                (
                    tensor_to_memref(t.get_element_type())
                    if isa(t, stencil.FieldType[TensorType[Attribute]])
                    else t
                )
                for t in op.function_type.outputs
            ],
        )


@dataclass(frozen=True)
class CslStencilBufferize(ModulePass):
    """
    Bufferizes the csl_stencil dialect
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
                ]
            )
        )
        module_pass.rewrite_module(op)
