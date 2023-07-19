from dataclasses import dataclass

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.ir import MLContext, Operation, OpResult
from xdsl.dialects.builtin import i32, f64, i64
from xdsl.dialects.func import FuncOp, Call
from xdsl.dialects import arith, builtin
from xdsl.dialects.arith import Constant

from xdsl.dialects.stencil import ExternalLoadOp, AccessOp, ApplyOp

from xdsl.dialects.experimental.hls import HLSStream, HLSStreamType, HLSStreamRead

from xdsl.dialects.memref import MemRefType

from xdsl.passes import ModulePass

from xdsl.utils.hints import isa

from xdsl.dialects.llvm import (
    LLVMPointerType,
    LLVMStructType,
    LLVMArrayType,
    GEPOp,
    LoadOp,
)

from xdsl.ir.core import BlockArgument

IN = 0
OUT = 1


# TODO docstrings and comments
@dataclass
class StencilAccessToGEP(RewritePattern):
    def __init__(self, op: builtin.ModuleOp):
        self.module = op

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        x = op.offset.array.data[0].data
        y = op.offset.array.data[1].data
        z = op.offset.array.data[2].data

        stream = op.parent_block().args[1]

        gep = GEPOp.get(stream, [0, 0], result_type=LLVMPointerType.typed(f64))
        gep = GEPOp.get(stream, [0, x, y, z], result_type=LLVMPointerType.typed(f64))

        load = LoadOp.get(gep, f64)

        rewriter.replace_matched_op([gep, load])


@dataclass
class StencilExternalLoadToHLSExternalLoad(RewritePattern):
    module: builtin.ModuleOp
    shift_streams: list
    load_data_declaration: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        assert isinstance(op.field, OpResult)
        field = op.field

        # Find the llvm.ptr to external memory that genrates the argument to the stencil.external_load. For PSyclone, this is
        # an argument to the parent function. TODO: this might need to be tested and generalised for other codes. Also, we are
        # considering that the function argument will be the second to insertvalue, but we're walking up trhough the second to
        # avoid bumping into arith.constants (see the mlir ssa).
        new_op = field
        func_arg = None

        while not isa(func_arg, BlockArgument):
            assert isinstance(new_op.owner, Operation)
            func_arg = new_op.owner.operands[-1]
            new_op = new_op.owner.operands[0]

        if isa(func_arg.typ, LLVMPointerType):
            func_arg_elem_type = func_arg.typ.type
        else:
            func_arg_elem_type = func_arg.typ

        stencil_type = LLVMStructType.from_type_list(
            [
                LLVMArrayType.from_size_and_type(
                    3,
                    LLVMArrayType.from_size_and_type(
                        3, LLVMArrayType.from_size_and_type(3, f64)
                    ),
                )
            ]
        )

        data_type = func_arg_elem_type
        struct_data_type = LLVMStructType.from_type_list([func_arg_elem_type])
        struct_stencil_type = LLVMStructType.from_type_list([stencil_type])

        assert isinstance(field.typ, MemRefType)
        shape = field.typ.get_shape()
        shape_x = Constant.from_int_and_width(shape[0], i32)
        shape_y = Constant.from_int_and_width(shape[1], i32)
        shape_z = Constant.from_int_and_width(shape[2], i32)

        two_int = Constant.from_int_and_width(2, i32)
        shift_shape_x = arith.Subi(shape_x, two_int)
        data_stream = HLSStream.get(data_type)
        stencil_stream = HLSStream.get(stencil_type)

        inout = op.attributes["inout"].value.data

        data_stream.attributes["inout"] = op.attributes["inout"]
        stencil_stream.attributes["inout"] = op.attributes["inout"]

        threedload_call = Call.get(
            "load_data", [func_arg, data_stream, shape_x, shape_y, shape_z], []
        )

        shift_buffer_call = Call.get(
            "shift_buffer",
            [data_stream, stencil_stream, shift_shape_x, shape_y, shape_z],
            [],
        )

        self.shift_streams.append(stencil_stream)

        if inout is IN:
            rewriter.insert_op_before_matched_op(
                [
                    data_stream,
                    stencil_stream,
                    shape_x,
                    shape_y,
                    shape_z,
                    two_int,
                    shift_shape_x,
                    threedload_call,
                    shift_buffer_call,
                ]
            )
        elif inout is OUT:
            rewriter.insert_op_before_matched_op(
                [
                    stencil_stream,
                ]
            )

        if not self.load_data_declaration:
            load_data_func = FuncOp.external(
                "load_data",
                [
                    func_arg.typ,
                    LLVMPointerType.typed(
                        LLVMStructType.from_type_list([data_stream.elem_type])
                    ),
                    i32,
                    i32,
                    i32,
                ],
                [],
            )
            self.module.body.block.add_op(load_data_func)
            shift_buffer_func = FuncOp.external(
                "shift_buffer",
                [
                    LLVMPointerType.typed(
                        LLVMStructType.from_type_list([data_stream.elem_type])
                    ),
                    LLVMPointerType.typed(
                        LLVMStructType.from_type_list([stencil_stream.elem_type])
                    ),
                    i32,
                    i32,
                    i32,
                ],
                [],
            )
            self.module.body.block.add_op(shift_buffer_func)

            self.load_data_declaration = True


@dataclass
class ApplyOpToHLS(RewritePattern):
    module: builtin.ModuleOp
    shift_streams: list

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        # Insert the HLS stream operands and their corresponding block arguments for reading from the shift buffer and writing
        # to external memory
        for stream in self.shift_streams:
            rewriter.insert_block_argument(
                op.region.block, len(op.region.block.args), stream.results[0].typ
            )

            old_operands_lst = [old_operand for old_operand in op.operands]
            op.operands = old_operands_lst + [stream.results[0]]

        # Indices of the streams to read. Used to locate the corresponding block argument
        indices_stream_to_read = []
        i = 0
        for _operand in op.operands:
            if (
                isinstance(_operand.op, HLSStream)
                and _operand.op.attributes["inout"].value.data is IN
            ):
                i += 1
                indices_stream_to_read.append(i)

        for arg_index in indices_stream_to_read:
            stream_to_read = op.region.block.args[arg_index]

            read_op = HLSStreamRead(stream_to_read)

            rewriter.insert_op_at_start(read_op, op.region.block)


@dataclass
class HLSConvertStencilToLLMLIRPass(ModulePass):
    name = "hls-convert-stencil-to-ll-mlir"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        load_data_declaration: bool = False
        module: builtin.ModuleOp = op
        shift_streams = []

        hls_pass = PatternRewriteWalker(
            # GreedyRewritePatternApplier([StencilExternalLoadToHLSExternalLoad(op), StencilAccessToGEP(op)]),
            GreedyRewritePatternApplier(
                [
                    StencilExternalLoadToHLSExternalLoad(module, shift_streams),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        hls_pass.rewrite_module(op)

        adapt_stencil_pass = PatternRewriteWalker(
            # GreedyRewritePatternApplier([StencilExternalLoadToHLSExternalLoad(op), StencilAccessToGEP(op)]),
            GreedyRewritePatternApplier([ApplyOpToHLS(module, shift_streams)]),
            apply_recursively=False,
            walk_reverse=True,
        )
        adapt_stencil_pass.rewrite_module(op)
