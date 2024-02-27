from typing import cast

from attr import dataclass

from xdsl.dialects import linalg, memref
from xdsl.dialects.builtin import (
    AffineMapAttr,
    DenseArrayBase,
    MemRefType,
    ModuleOp,
    TensorType,
    UnitAttr,
    i32,
)
from xdsl.dialects.linalg import Generic, IteratorTypeAttr, YieldOp
from xdsl.dialects.memref import Subview
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    CastOp,
    FieldType,
    ReturnOp,
    StencilBoundsAttr,
    StencilType,
    StoreOp,
    TempType,
)
from xdsl.dialects.tensor import EmptyOp
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    MLContext,
    Operation,
    Region,
    SSAValue,
)
from xdsl.ir.affine import AffineExpr, AffineMap
from xdsl.irdl import (
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    result_def,
)
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

# TODO: PASS: convert-stencil-to-tensor,mlir-opt[eliminate-empty-tensors,one-shot-bufferize,cse,canonicalize,convert-linalg-to-parallel-loops]


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


class Mat(Operation):
    name = "bufferization.materialize_in_destination"

    traits = frozenset()


@irdl_op_definition
class toTensor(IRDLOperation):
    name = "bufferization.to_tensor"

    memref = operand_def(MemRefType)
    tensor = result_def(TensorType)
    writable = opt_prop_def(UnitAttr)
    restrict = opt_prop_def(UnitAttr)


class Slice(Operation):
    name = "tensor.insert_slice"

    traits = frozenset()


class StoreOpToSubviewCopy(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: StoreOp, rewriter: PatternRewriter, /):
        field_t = cast(FieldType[Attribute], op.field.type)
        temp_t = cast(TempType[Attribute], op.temp.type)
        assert isinstance(field_t.bounds, StencilBoundsAttr)

        to_tensor = toTensor(
            operands=[op.field],
            properties={"writable": UnitAttr(), "restrict": UnitAttr()},
            result_types=[stencil_type_to_tensor(field_t)],
        )
        rewriter.insert_op_after(
            to_tensor,
            op.field.owner,
        )
        subview = Subview.from_static_parameters(
            op.field,
            stencil_type_to_memref(field_t),
            tuple(-lb for lb in field_t.bounds.lb),
            temp_t.get_shape(),
            (1,) * temp_t.get_num_dims(),
        )
        subview.properties["operandSegmentSizes"] = DenseArrayBase.from_list(
            i32, (1, 1, 0, 0, 0)
        )
        insert = Slice(
            operands=[op.temp, to_tensor.tensor],
            result_types=[stencil_type_to_tensor(field_t)],
            properties=subview.properties,
        )

        # c = Mat(
        #     operands=[op.temp, subview.result],
        #     properties={"writable": UnitAttr(), "restrict": UnitAttr()},
        # )

        # c = copy(op.temp, subview.result)
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
    t = cast(TensorType[Attribute], tensor.type)
    block = Block([], arg_types=[t.get_element_type()] * 2)
    block.add_op(linalg.YieldOp(block.args[0]))

    map = AffineMap(
        len(offsets),
        0,
        tuple(AffineExpr.dimension(i) + o for i, o in enumerate(offsets)),
    )
    output_type = TensorType(t.get_element_type(), sizes)
    empty = EmptyOp([], output_type)
    extract_slice = Generic(
        [tensor],
        [empty.tensor],
        Region(block),
        [AffineMapAttr(map)],
        [IteratorTypeAttr.parallel()] * t.get_num_dims(),
        [output_type],
    )
    return empty, extract_slice


def copy(input: SSAValue, output: SSAValue):
    t = cast(TensorType[Attribute], input.type)
    block = Block([], arg_types=[t.get_element_type()] * 2)
    block.add_op(linalg.YieldOp(block.args[0]))
    return Generic(
        [input],
        [output],
        Region(block),
        [
            AffineMapAttr(AffineMap.identity(t.get_num_dims())),
            AffineMapAttr(AffineMap.identity(t.get_num_dims())),
        ],
        [IteratorTypeAttr.parallel()] * t.get_num_dims(),
        [],
    )


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

                extract = extract_slice(operand, tuple(a.offset), shape)

                indexing_maps.append(AffineMapAttr(AffineMap.identity(len(a.offset))))

                extract_slices += list(extract)
                inputs.append(extract[-1].res[0])
                new_block.insert_arg(inputs[-1].type, len(new_block.args))
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

        # Create a generic op with the stencil inputs
        generic = Generic(
            inputs,
            tuple(o.tensor for o in outputs_ops),
            Region(new_block),
            indexing_maps,
            [IteratorTypeAttr.parallel()] * op.get_rank(),
            output_types,
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
                    CastOpToCast(),
                    ApplyOpToGeneric(),
                    ReturnOpToYield(),
                    StencilTempConversion(recursive=True),
                    StencilFieldConversion(recursive=True),
                ]
            ),
            walk_reverse=True,
        )
        the_one_pass.rewrite_module(op)
