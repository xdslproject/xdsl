import operator
from functools import reduce
from typing import cast

from xdsl.backend.riscv.lowering.convert_memref_to_riscv import element_size_for_type
from xdsl.backend.riscv.lowering.utils import (
    cast_operands_to_regs,
    move_to_unallocated_regs,
    register_type_for_type,
)
from xdsl.context import MLContext
from xdsl.dialects import (
    builtin,
    memref,
    memref_stream,
    riscv,
    riscv_snitch,
    snitch_stream,
    stream,
)
from xdsl.dialects.builtin import (
    ArrayAttr,
    IntAttr,
    MemRefType,
    ModuleOp,
    UnrealizedConversionCastOp,
)
from xdsl.ir import Attribute, AttributeCovT, Operation
from xdsl.ir.affine import AffineExpr, AffineMap
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


class ReadOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.ReadOp, rewriter: PatternRewriter
    ) -> None:
        stream_type = op.stream.type
        assert isinstance(stream_type, stream.ReadableStreamType)
        value_type = cast(
            stream.ReadableStreamType[Attribute], stream_type
        ).element_type
        register_type = register_type_for_type(value_type).unallocated()

        new_stream = UnrealizedConversionCastOp.get(
            (op.stream,), (stream.ReadableStreamType(register_type),)
        )
        new_op = riscv_snitch.ReadOp(new_stream.results[0])
        if len(op.res.uses) == 1:
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

        rewriter.replace_matched_op(
            (new_stream, new_op, *new_mv, new_res),
        )


class WriteOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.WriteOp, rewriter: PatternRewriter
    ) -> None:
        stream_type = op.stream.type
        assert isinstance(stream_type, stream.WritableStreamType)
        value_type = cast(
            stream.WritableStreamType[Attribute], stream_type
        ).element_type
        register_type = register_type_for_type(value_type).unallocated()

        new_stream = UnrealizedConversionCastOp.get(
            (op.stream,), (stream.WritableStreamType(register_type),)
        )
        cast_op = UnrealizedConversionCastOp.get((op.value,), (register_type,))
        if isinstance(defining_op := op.value.owner, Operation) and (
            defining_op.parent_region() is op.parent_region()
            and not isinstance(defining_op, memref_stream.ReadOp)
        ):
            move_ops = ()
            new_values = cast_op.results
        else:
            move_ops, new_values = move_to_unallocated_regs(
                cast_op.results, (value_type,)
            )
        new_write = riscv_snitch.WriteOp(new_values[0], new_stream.results[0])

        rewriter.replace_matched_op(
            (new_stream, cast_op, *move_ops, new_write),
        )


class StreamOpLowering(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: memref_stream.StreamingRegionOp, rewriter: PatternRewriter
    ) -> None:
        operand_types = tuple(
            cast(memref.MemRefType[Attribute], value_type)
            for value in op.operands
            if isinstance(value_type := value.type, memref.MemRefType)
        )
        el_types = tuple(operand_type.element_type for operand_type in operand_types)
        if not all(el_type == builtin.f64 for el_type in el_types):
            # Only support f64 streams for now
            return
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
        freg = riscv.FloatRegisterType.unallocated()

        rewriter.replace_matched_op(
            new_op := snitch_stream.StreamingRegionOp(
                new_inputs,
                new_outputs,
                ArrayAttr(stride_patterns),
                rewriter.move_region_contents_to_new_regions(op.body),
            )
        )

        new_body = new_op.body.block

        input_stream_types = (stream.ReadableStreamType(freg),) * len(op.inputs)
        output_stream_types = (stream.WritableStreamType(freg),) * len(op.outputs)
        stream_types = input_stream_types + output_stream_types
        for i in reversed(range(len(stream_types))):
            arg = new_body.args[i]
            stream_type = stream_types[i]
            rewriter.insert_op(
                cast_op := builtin.UnrealizedConversionCastOp.get((arg,), (arg.type,)),
                InsertPoint.at_start(new_body),
            )
            arg.replace_by(cast_op.results[0])
            cast_op.operands = (arg,)
            rewriter.modify_block_argument_type(arg, stream_type)


def strides_map_from_memref_type(memref_type: MemRefType[AttributeCovT]) -> AffineMap:
    """
    Given a memref returns the map from indices to an offset in bytes in memory. The
    resulting map has one result expression.

    e.g.:
    ```
    my_list = [1, 2, 3, 4, 5, 6]
    shape = [2, 3]
    for i in range(2):
        for j in range(3):
            k = i * 3 + j
            el = my_list[k]
            print(el) # -> 1, 2, 3, 4, 5, 6

    map = strides_map_from_memref_type(MemRefType(f64, [3, 1]))

    for i in range(2):
        for j in range(3):
            k = map.eval(i, j) // 8 # factor of 8 since 8 bytes in f64
            el = my_list[k]
            print(el) # -> 1, 2, 3, 4, 5, 6
    ```
    """
    maybe_strides = tuple(memref_type.get_strides())
    for stride in maybe_strides:
        if stride is None:
            raise DiagnosticException(
                f"Cannot get strides from dynamic layout {memref_type}"
            )

    strides = cast(tuple[int, ...], maybe_strides)

    if not strides:
        raise DiagnosticException(
            f"Unsupported empty shape in memref of type {memref_type}"
        )

    factor = element_size_for_type(memref_type.element_type)

    return AffineMap(
        len(strides),
        0,
        (
            reduce(
                operator.add,
                (
                    AffineExpr.dimension(i) * stride * factor
                    for i, stride in enumerate(strides)
                ),
            ),
        ),
    )


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
    offset_map = strides_map_from_memref_type(memref_type)
    composed = offset_map.compose(affine_map)

    zeros = [0] * composed.num_dims

    result: list[int] = []

    for i in range(composed.num_dims):
        zeros[i] = 1
        result.append(composed.eval(zeros, ())[0])
        zeros[i] = 0

    return result


class ConvertMemrefStreamToSnitchStreamPass(ModulePass):
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

    def apply(self, ctx: MLContext, op: ModuleOp) -> None:
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
