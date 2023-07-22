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
from xdsl.dialects import arith, builtin, scf
from xdsl.dialects.arith import Constant

from xdsl.dialects.stencil import ExternalLoadOp, AccessOp, ApplyOp, ReturnOp

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
    AllocaOp,
)

from xdsl.ir.core import BlockArgument, Block, Region
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import prepare_apply_body

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

        qualify_apply_op_with_shapes(op.parent_op(), shape_x, shape_y, shape_z)

        two_int = Constant.from_int_and_width(2, i32)
        shift_shape_x = arith.Subi(shape_x, two_int)
        data_stream = HLSStream.get(data_type)
        stencil_stream = HLSStream.get(stencil_type)
        copy_stencil_stream = HLSStream.get(stencil_type)

        one_int = Constant.from_int_and_width(1, i32)
        four_int = Constant.from_int_and_width(4, i32)
        copy_shift_x = arith.Subi(shape_x, four_int)
        copy_shift_y = arith.Subi(shape_y, four_int)
        copy_shift_z = arith.Subi(shape_z, one_int)
        prod_x_y = arith.Muli(copy_shift_x, copy_shift_y)
        copy_n = arith.Muli(prod_x_y, copy_shift_z)

        inout = op.attributes["inout"].value.data

        data_stream.attributes["inout"] = op.attributes["inout"]
        stencil_stream.attributes["inout"] = op.attributes["inout"]
        copy_stencil_stream.attributes["inout"] = op.attributes["inout"]

        threedload_call = Call.get(
            "load_data", [func_arg, data_stream, shape_x, shape_y, shape_z], []
        )

        shift_buffer_call = Call.get(
            "shift_buffer",
            [data_stream, stencil_stream, shift_shape_x, shape_y, shape_z],
            [],
        )

        duplicateStream_call = Call.get(
            "duplicateStream",
            [stencil_stream, copy_stencil_stream, copy_n],
            [],
        )

        if inout is IN:
            rewriter.insert_op_before_matched_op(
                [
                    data_stream,
                    stencil_stream,
                    copy_stencil_stream,
                    shape_x,
                    shape_y,
                    shape_z,
                    two_int,
                    shift_shape_x,
                    threedload_call,
                    shift_buffer_call,
                    one_int,
                    four_int,
                    copy_shift_x,
                    copy_shift_y,
                    copy_shift_z,
                    prod_x_y,
                    copy_n,
                    duplicateStream_call,
                ]
            )
            self.shift_streams.append(copy_stencil_stream)
        elif inout is OUT:
            rewriter.insert_op_before_matched_op(
                [
                    stencil_stream,
                ]
            )
            self.shift_streams.append(stencil_stream)

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
            duplicateStream_func = FuncOp.external(
                "duplicateStream",
                [
                    LLVMPointerType.typed(
                        LLVMStructType.from_type_list([stencil_stream.elem_type])
                    ),
                    LLVMPointerType.typed(
                        LLVMStructType.from_type_list([stencil_stream.elem_type])
                    ),
                    i32,
                ],
                [],
            )
            self.module.body.block.add_op(duplicateStream_func)

            self.load_data_declaration = True


def qualify_apply_op_with_shapes(
    stencil_func: FuncOp,
    shape_x: arith.Constant,
    shape_y: arith.Constant,
    shape_z: arith.Constant,
):
    block = stencil_func.body.block

    for op in block.ops:
        if isinstance(op, ApplyOp):
            op.attributes["shape_x"] = shape_x.value
            op.attributes["shape_y"] = shape_y.value
            op.attributes["shape_z"] = shape_z.value


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
                indices_stream_to_read.append(i)
            i += 1

        for arg_index in indices_stream_to_read:
            stream_to_read = op.region.block.args[arg_index]

            read_op = HLSStreamRead(stream_to_read)

            rewriter.insert_op_at_start(read_op, op.region.block)

        # Transform ApplyOp into for loops
        get_number_chunks = FuncOp.external(
            "get_number_chunks",
            [i32, LLVMPointerType.typed(i32)],
            [i32],
        )

        get_chunk_size = FuncOp.external(
            "get_chunk_size",
            [i32, i32, i32, i32],
            [i32],
        )

        res_type = op.res[0].typ
        # Get this apply's ReturnOp
        body_block = op.region.blocks[0]
        return_op = next(o for o in body_block.ops if isinstance(o, ReturnOp))
        rewriter.erase_op(return_op)

        body = prepare_apply_body(op, rewriter)
        body.block.add_op(scf.Yield.get())
        dim = res_type.get_num_dims()

        size_x = Constant.from_int_and_width(
            op.attributes["shape_x"].value.data, builtin.IndexType()
        )
        size_y = Constant.from_int_and_width(
            op.attributes["shape_y"].value.data, builtin.IndexType()
        )
        size_z = Constant.from_int_and_width(
            op.attributes["shape_z"].value.data, builtin.IndexType()
        )
        one_int = Constant.from_int_and_width(1, i32)
        two = Constant.from_int_and_width(2, builtin.IndexType())
        zero = Constant.from_int_and_width(0, builtin.IndexType())
        one = Constant.from_int_and_width(1, builtin.IndexType())

        size_x_2 = arith.Subi(size_x, two)
        size_y_1 = arith.Subi(size_y, one)

        lower_x = Constant.from_int_and_width(2, builtin.IndexType())
        lower_y = Constant.from_int_and_width(1, builtin.IndexType())
        lower_z = Constant.from_int_and_width(1, builtin.IndexType())
        upper_x = size_x_2
        upper_y = size_y_1
        upper_z = Constant.from_int_and_width(
            op.attributes["shape_z"].value.data, builtin.IndexType()
        )

        p_remainder = AllocaOp.get(one_int, i32)

        call_get_number_chunks = Call.get(
            "get_number_chunks", [upper_x, p_remainder], [builtin.IndexType()]
        )

        lower_chunks = zero
        upper_chunks = call_get_number_chunks

        lowerBounds = [lower_chunks, lower_x, lower_y, lower_z]
        upperBounds = [upper_chunks, upper_x, upper_y, upper_z]

        # The for loop for the y index receives its trip variable from the get_chunk_size function, since the chunking
        # is happening in the y axis. TODO: this is currently intended for the 3D case. It should be extended to the
        # 1D and 2D cases as well.
        y_for_op = None

        current_region = body
        for i in range(1, dim + 1):
            for_op = scf.For.get(
                lb=lowerBounds[-i],
                ub=upperBounds[-i],
                step=one,
                iter_args=[],
                body=current_region,
            )
            block = Block(
                ops=[for_op, scf.Yield.get()], arg_types=[builtin.IndexType()]
            )
            current_region = Region(block)

            if i == 2:
                y_for_op = for_op

        p = scf.ParallelOp.get(
            lowerBounds=[lowerBounds[0]],
            upperBounds=[upperBounds[0]],
            steps=[one],
            body=current_region,
        )

        chunk_num = p.body.block.args[0]

        MAX_Y_SIZE = 16
        max_chunk_length = Constant.from_int_and_width(MAX_Y_SIZE, i32)

        remainder = LoadOp.get(p_remainder)

        call_get_chunk_size = Call.get(
            "get_chunk_size",
            [chunk_num, call_get_number_chunks, max_chunk_length, remainder],
            [builtin.IndexType()],
        )

        p.body.block.insert_op_before(call_get_chunk_size, p.body.block.first_op)

        old_operands_lst = [old_operand for old_operand in y_for_op.operands]
        y_for_op.operands = [call_get_chunk_size.res[0]] + old_operands_lst[1:]

        rewriter.insert_op_before_matched_op(
            [
                size_x,
                size_y,
                one_int,
                one,
                two,
                max_chunk_length,
                p_remainder,
                remainder,
                *lowerBounds,
                *upperBounds,
                p,
            ]
        )


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
