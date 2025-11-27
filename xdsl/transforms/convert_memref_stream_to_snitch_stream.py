from typing import Any, cast

from xdsl.backend.riscv.lowering.utils import (
    cast_operands_to_regs,
    move_to_unallocated_regs,
)
from xdsl.context import Context
from xdsl.dialects import (
    builtin,
    memref,
    memref_stream,
    riscv,
    riscv_snitch,
    snitch,
    snitch_stream,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    Float16Type,
    Float32Type,
    Float64Type,
    IntAttr,
    MemRefType,
    ModuleOp,
    UnrealizedConversionCastOp,
    VectorType,
)
from xdsl.ir import Attribute, AttributeCovT, Operation
from xdsl.ir.affine import AffineMap
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.exceptions import DiagnosticException


def snitch_stream_element_type_is_valid(attr: Attribute) -> bool:
    """
    An override of the helper to account for Snitch packed SIMD.
    """
    if isinstance(attr, VectorType):
        attr = cast(VectorType[Any], attr)
        match attr.element_type, attr.element_count():
            case Float64Type(), 1:
                return True
            case Float32Type(), 2:
                return True
            case Float16Type(), 4:
                return True
            case _:
                # TODO: handle fp8
                return False
    else:
        return isinstance(attr, Float64Type)


class ReadOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.ReadOp, rewriter: PatternRewriter
    ) -> None:
        stream_type = op.stream.type
        assert isinstance(stream_type, memref_stream.ReadableStreamType)
        value_type = cast(
            memref_stream.ReadableStreamType[Attribute], stream_type
        ).element_type
        if not snitch_stream_element_type_is_valid(value_type):
            raise DiagnosticException(
                f"Invalid snitch stream element type {value_type}"
            )
        register_type = riscv.Registers.UNALLOCATED_FLOAT

        new_stream = UnrealizedConversionCastOp.get(
            (op.stream,), (snitch.ReadableStreamType(register_type),)
        )
        new_op = riscv_snitch.ReadOp(new_stream.results[0])
        if op.res.has_one_use():
            new_mv = ()
            new_vals = (new_op.res,)
        else:
            new_mv, new_vals = move_to_unallocated_regs(
                (new_op.res,),
                (value_type,),
            )
        new_res = UnrealizedConversionCastOp.get(
            new_vals,
            (value_type,),
        )

        rewriter.replace_op(
            op,
            (new_stream, new_op, *new_mv, new_res),
        )


class WriteOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.WriteOp, rewriter: PatternRewriter
    ) -> None:
        stream_type = op.stream.type
        assert isinstance(stream_type, memref_stream.WritableStreamType)
        value_type = cast(
            memref_stream.WritableStreamType[Attribute], stream_type
        ).element_type
        if not snitch_stream_element_type_is_valid(value_type):
            raise DiagnosticException(
                f"Invalid snitch stream element type {value_type}"
            )
        register_type = riscv.Registers.UNALLOCATED_FLOAT

        new_stream = UnrealizedConversionCastOp.get(
            (op.stream,), (snitch.WritableStreamType(register_type),)
        )
        cast_op = UnrealizedConversionCastOp.get((op.value,), (register_type,))
        if isinstance(defining_op := op.value.owner, Operation) and (
            defining_op.parent_region() is op.parent_region()
            and not isinstance(defining_op, memref_stream.ReadOp)
        ):
            move_ops = ()
            new_values = cast_op.results
        else:
            move_ops = (
                riscv.FMvDOp(cast_op.results[0], rd=riscv.Registers.UNALLOCATED_FLOAT),
            )
            new_values = move_ops[0].results
        new_write = riscv_snitch.WriteOp(new_values[0], new_stream.results[0])

        rewriter.replace_op(
            op,
            (new_stream, cast_op, *move_ops, new_write),
        )


class StreamOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.StreamingRegionOp, rewriter: PatternRewriter
    ) -> None:
        operand_types = tuple(
            cast(memref.MemRefType, value_type)
            for value in op.operands
            if isinstance(value_type := value.type, memref.MemRefType)
        )
        stride_patterns = tuple(
            snitch_stream.StridePattern(
                ArrayAttr(ub.value for ub in pattern.ub),
                ArrayAttr(
                    IntAttr(stride)
                    for stride in strides_for_affine_map(
                        pattern.index_map.data, memref_type
                    )
                ),
            ).simplified()
            for pattern, memref_type in zip(op.patterns, operand_types, strict=True)
        )
        if len(set(stride_patterns)) == 1:
            stride_patterns = (stride_patterns[0],)
        new_operands = cast_operands_to_regs(rewriter)
        new_inputs = new_operands[: len(op.inputs)]
        new_outputs = new_operands[len(op.inputs) :]
        freg = riscv.Registers.UNALLOCATED_FLOAT

        rewriter.replace_op(
            op,
            new_op := snitch_stream.StreamingRegionOp(
                new_inputs,
                new_outputs,
                ArrayAttr(stride_patterns),
                rewriter.move_region_contents_to_new_regions(op.body),
            ),
        )

        new_body = new_op.body.block

        input_stream_types = (snitch.ReadableStreamType(freg),) * len(op.inputs)
        output_stream_types = (snitch.WritableStreamType(freg),) * len(op.outputs)
        stream_types = input_stream_types + output_stream_types
        for i in reversed(range(len(stream_types))):
            arg = new_body.args[i]
            stream_type = stream_types[i]
            rewriter.insert_op(
                cast_op := builtin.UnrealizedConversionCastOp.get((arg,), (arg.type,)),
                InsertPoint.at_start(new_body),
            )
            arg.replace_by_if(
                cast_op.results[0], lambda use: use.operation is not cast_op
            )
            rewriter.replace_value_with_new_type(arg, stream_type)


def strides_for_affine_map(
    affine_map: AffineMap, memref_type: MemRefType[AttributeCovT]
) -> list[int]:
    """
    Given an iteration space represented as an affine map (for indexing) and a shape (for
    bounds), returns the corresponding iteration strides for each dimension.

    The affine map must not have symbols.
    """
    if affine_map.num_symbols:
        raise ValueError("Cannot create strides for affine map with symbols")

    # only static memref shapes are supported for now:
    static_shapes = (shape != -1 for shape in memref_type.get_shape())
    if not all(static_shapes):
        raise ValueError("Cannot create strides for a memref with dynamic shapes")

    offset_map = memref_type.get_affine_map_in_bytes()
    composed = offset_map.compose(affine_map)

    zeros = [0] * composed.num_dims
    # composed map can have symbols for dynamic offset, just set them to 0
    symbols = [0] * composed.num_symbols

    result: list[int] = []

    # subtract the static offset from each result
    offset = composed.eval(zeros, symbols)[0]

    for i in range(composed.num_dims):
        zeros[i] = 1
        result.append(composed.eval(zeros, symbols)[0] - offset)
        zeros[i] = 0

    return result


class ConvertMemRefStreamToSnitchStreamPass(ModulePass):
    """
    Converts memref_stream `read` and `write` operations to the snitch_stream equivalents.

    Care needs to be taken to preserve the semantics of the program.
    In assembly, the reads and writes are implicit, by using a register.
    In IR, they are modeled by `read` and `write` ops, which are not printed at the
    assembly level.

    To preserve semantics, additional move ops are inserted in the following cases:
     - reading form a stream: if the value read has multiple uses,
     - writing to a stream: if the value is defined by an operation outside of the
     streaming region or if the defining operation is a stream read.
    """

    name = "convert-memref-stream-to-snitch-stream"

    def apply(self, ctx: Context, op: ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    ReadOpLowering(),
                    WriteOpLowering(),
                    StreamOpLowering(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        ).rewrite_module(op)
