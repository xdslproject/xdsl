from typing import cast

from attr import dataclass

from xdsl.dialects import memref
from xdsl.dialects.builtin import (
    AffineMapAttr,
    MemRefType,
    ModuleOp,
    StringAttr,
    TensorType,
    UnitAttr,
)
from xdsl.dialects.linalg import Generic, IteratorTypeAttr, YieldOp
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    CastOp,
    FieldType,
    LoadOp,
    ReturnOp,
    StencilBoundsAttr,
    StencilType,
    StoreOp,
    TempType,
)
from xdsl.dialects.tensor import EmptyOp, ExtractSliceOp, InsertSliceOp
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    MLContext,
    Operation,
    Region,
    SSAValue,
)
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    result_def,
)
from xdsl.passes import ModulePass, PipelinePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    TypeConversionPattern,
    attr_type_rewrite_pattern,
    op_type_rewrite_pattern,
)
from xdsl.transforms.mlir_opt import MLIROptPass

# TODO: PASS: convert-stencil-to-tensor,mlir-opt[eliminate-empty-tensors,cse,one-shot-bufferize,canonicalize,convert-linalg-to-parallel-loops]


def stencil_type_to_memref(field: StencilType[Attribute]):
    return MemRefType(field.get_element_type(), field.get_shape())


def stencil_type_to_tensor(field: StencilType[Attribute]):
    return TensorType(field.get_element_type(), field.get_shape())


class StencilFieldConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: FieldType[Attribute]) -> MemRefType[Attribute]:
        return stencil_type_to_memref(typ)


class StencilTempConversion(TypeConversionPattern):
    @attr_type_rewrite_pattern
    def convert_type(self, typ: TempType[Attribute]) -> TensorType[Attribute]:
        return stencil_type_to_tensor(typ)


@irdl_op_definition
class toTensor(IRDLOperation):
    name = "bufferization.to_tensor"

    memref = operand_def(MemRefType)
    tensor = result_def(TensorType)
    writable = opt_prop_def(UnitAttr)
    restrict = opt_prop_def(UnitAttr)


def bufferization_to_tensor(memref: SSAValue):
    memref_t = cast(StencilType[Attribute], memref.type)
    return toTensor(
        operands=[memref],
        properties={"writable": UnitAttr(), "restrict": UnitAttr()},
        result_types=[stencil_type_to_tensor(memref_t)],
    )


class LoadOpToSubviewCopy(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LoadOp, rewriter: PatternRewriter, /):
        to_tensor = bufferization_to_tensor(op.field)
        field_t = cast(StencilType[Attribute], op.field.type)
        temp_t = cast(StencilType[Attribute], op.res.type)
        assert isinstance(field_t.bounds, StencilBoundsAttr)
        assert isinstance(temp_t.bounds, StencilBoundsAttr)
        offsets = tuple(
            -flb + tlb for flb, tlb in zip(field_t.bounds.lb, temp_t.bounds.lb)
        )
        sizes = temp_t.get_shape()
        extract = extract_slice(to_tensor.tensor, offsets, sizes)
        rewriter.replace_matched_op((to_tensor, extract))


class StoreOpToSubviewCopy(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        field_t = cast(FieldType[Attribute], op.field.type)
        temp_t = cast(TempType[Attribute], op.temp.type)
        assert isinstance(field_t.bounds, StencilBoundsAttr)

        to_tensor = bufferization_to_tensor(op.field)
        match op.field.owner:
            case Operation():
                rewriter.insert_op_after(
                    to_tensor,
                    op.field.owner,
                )
            case Block():
                rewriter.insert_op_at_start(to_tensor, op.field.owner)

        offsets = tuple(-lb for lb in field_t.bounds.lb)
        sizes = temp_t.get_shape()
        strides = (1,) * temp_t.get_num_dims()
        insert = InsertSliceOp.from_static_parameters(
            op.temp,
            to_tensor.tensor,
            offsets,
            sizes,
            strides,
        )

        rewriter.replace_matched_op((insert), new_results=())


@dataclass(frozen=True)
class ReturnOpToYield(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ReturnOp, rewriter: PatternRewriter, /):
        rewriter.replace_matched_op(YieldOp(*op.operands))


@dataclass(frozen=True)
class CastOpToCast(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CastOp, rewriter: PatternRewriter, /):
        t = cast(FieldType[Attribute], op.result.type)
        rewriter.replace_matched_op(
            memref.Cast.get(op.field, stencil_type_to_memref(t))
        )


def extract_slice(tensor: SSAValue, offsets: tuple[int, ...], sizes: tuple[int, ...]):
    # TODO Implement and use tensor.extract_slice
    t = cast(TempType[Attribute], tensor.type)

    extract_slice = ExtractSliceOp.from_static_parameters(
        tensor,
        stencil_type_to_tensor(t),
        offsets,
        sizes,
        strides=(1,) * len(sizes),
    )
    return extract_slice


@dataclass(frozen=True)
class ApplyOpToGeneric(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        output = op.res[0]
        output_type = cast(TempType[Attribute], output.type)
        shape = output_type.get_shape()
        # Get accesses to generate inputs and replace by block arguments
        inputs: list[SSAValue] = []
        new_block = Block()
        extract_slices: list[Operation] = []
        indexing_maps: list[AffineMapAttr] = []
        for a in op.walk():
            if isinstance(a, AccessOp):
                if a.offset_mapping is not None:
                    raise NotImplementedError(
                        "Offset mapping not implemented in the tensor lowering."
                    )
                block_arg = a.temp
                assert isinstance(block_arg, BlockArgument)
                operand = op.args[block_arg.index]

                temp_t = cast(TempType[Attribute], a.temp.type)
                assert isinstance(temp_t.bounds, StencilBoundsAttr)

                offsets = tuple(-lb + o for lb, o in zip(temp_t.bounds.lb, a.offset))

                extract = extract_slice(operand, offsets, shape)

                indexing_maps.append(AffineMapAttr(AffineMap.identity(len(a.offset))))

                extract_slices.append(extract)
                out_t = cast(TensorType[Attribute], extract.result.type)

                inputs.append(extract.result)
                new_block.insert_arg(out_t.get_element_type(), len(new_block.args))
                rewriter.replace_op(a, [], [new_block.args[-1]])
        for a in op.region.block.args:
            for u in tuple(a.uses):
                u.operation.operands[u.index] = op.args[u.index]

        for o in op.region.block.ops:
            rewriter.insert_op_at_end(op.region.block.detach_op(o), new_block)

        output_types = tuple(
            stencil_type_to_tensor(cast(TempType[Attribute], r.type)) for r in op.res
        )
        outputs_ops = tuple(EmptyOp([], t) for t in output_types)

        for ot in output_types:
            indexing_maps.append(AffineMapAttr(AffineMap.identity(ot.get_num_dims())))
            new_block.insert_arg(ot.get_element_type(), len(new_block.args))

        op.get_rank()

        generic = Generic(
            inputs,
            tuple(o.tensor for o in outputs_ops),
            Region(new_block),
            indexing_maps,
            [IteratorTypeAttr.parallel()] * op.get_rank(),
            output_types,
            StringAttr("apply"),
        )

        rewriter.replace_matched_op(
            (*extract_slices, *outputs_ops, generic),
        )

        pass


@dataclass(frozen=True)
class ConvertStencilToLinalg(ModulePass):
    name = "convert-stencil-to-tensor"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        ctx.get_optional_op("bufferization.materialize_in_destination")
        the_one_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    StoreOpToSubviewCopy(),
                    LoadOpToSubviewCopy(),
                    CastOpToCast(),
                    ApplyOpToGeneric(),
                    ReturnOpToYield(),
                ]
            ),
            walk_reverse=True,
        )
        the_one_pass.rewrite_module(op)
        type_conversion_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    StencilTempConversion(recursive=True),
                    StencilFieldConversion(recursive=True),
                ]
            )
        )
        type_conversion_pass.rewrite_module(op)


class ConvertStencilToTensorCOmpat(ModulePass):
    name = "convert-stencil-to-tensor-compat"

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
        pipeline = PipelinePass(
            (
                ConvertStencilToLinalg(),
                MLIROptPass(
                    arguments=(
                        "--allow-unregistered-dialect",
                        "--mlir-print-op-generic",
                        "-p",
                        "builtin.module(eliminate-empty-tensors,cse,one-shot-bufferize,canonicalize,convert-linalg-to-parallel-loops)",
                    )
                ),
            )
        )
        pipeline.apply(ctx, op)
