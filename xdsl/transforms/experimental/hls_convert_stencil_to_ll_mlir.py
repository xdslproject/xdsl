from dataclasses import dataclass

from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    GreedyRewritePatternApplier,
    op_type_rewrite_pattern,
)
from xdsl.ir import MLContext, Operation, OpResult
from xdsl.dialects.builtin import i32, f64, i64, ArrayAttr, DenseArrayBase
from xdsl.dialects.func import FuncOp, Call
from xdsl.dialects import arith, builtin, scf, stencil
from xdsl.dialects.arith import Constant

from xdsl.dialects.stencil import (
    ExternalLoadOp,
    ExternalStoreOp,
    AccessOp,
    ApplyOp,
    ReturnOp,
)

from xdsl.dialects.experimental.hls import (
    HLSStream,
    HLSStreamType,
    HLSStreamRead,
    HLSStreamWrite,
)

from xdsl.dialects.memref import MemRefType

from xdsl.passes import ModulePass

from xdsl.utils.hints import isa

from xdsl.dialects.llvm import (
    LLVMPointerType,
    LLVMStructType,
    LLVMArrayType,
    GEPOp,
    LoadOp,
    StoreOp,
    AllocaOp,
    ExtractValueOp,
)

from xdsl.ir.core import BlockArgument, Block, Region
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import (
    prepare_apply_body,
    AccessOpToMemref,
    LoadOpToMemref,
    CastOpToMemref,
    TrivialExternalLoadOpCleanup,
    TrivialExternalStoreOpCleanup,
)

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
    out_data_streams: list
    out_global_mem: list
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

        if op.attributes["inout"].value.data is OUT:
            self.out_global_mem.append(func_arg)

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

        if len(shape) < 3:
            return

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

        # We need to indicate that this is a stencil stream and not a data stream. TODO: make this more elegant
        stencil_stream.attributes["stencil"] = op.attributes["inout"]
        copy_stencil_stream.attributes["stencil"] = op.attributes["inout"]

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
            out_data_stream = HLSStream.get(data_type)
            out_data_stream.attributes["inout"] = op.attributes["inout"]
            out_data_stream.attributes["data"] = op.attributes["inout"]
            rewriter.insert_op_before_matched_op(
                [
                    out_data_stream,
                    # stencil_stream
                ]
            )
            self.out_data_streams.append(out_data_stream)
            # self.shift_streams.append(stencil_stream)

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
    out_data_streams: list
    out_global_mem: list

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        # Insert the HLS stream operands and their corresponding block arguments for reading from the shift buffer and writing
        # to external memory

        # We replace by streams only the 3D temps. The rest should be left as is
        operand_stream = dict()

        current_stream = 0
        new_operands_lst = []

        for i in range(len(op.operands)):
            operand = op.operands[i]
            n_dims = len(operand.typ.bounds.lb)

            if n_dims == 3:
                stream = self.shift_streams[current_stream]
                rewriter.modify_block_argument_type(
                    op.region.block.args[i], stream.results[0].typ
                )

                new_operands_lst.append(stream.results[0])
                current_stream += 1
            else:
                new_operands_lst.append(operand)

        op.operands = new_operands_lst

        indices_stream_to_read = []
        indices_stream_to_write = []
        i = 0
        for _operand in op.operands:
            if (
                isinstance(_operand.op, HLSStream)
                and "stencil" in _operand.op.attributes
                and _operand.op.attributes["inout"].value.data is IN
            ):
                indices_stream_to_read.append(i)
            # if (
            #    isinstance(_operand.op, HLSStream)
            #    and "data" in _operand.op.attributes
            #    and _operand.op.attributes["inout"].value.data is OUT
            # ):
            #    indices_stream_to_write.append(i)
            i += 1

        # Get this apply's ReturnOp
        body_block = op.region.blocks[0]
        return_op = next(o for o in body_block.ops if isinstance(o, ReturnOp))

        stencil_return_vals = [val for val in return_op.arg]

        alloca_size = Constant.from_int_and_width(1, i32)

        global_mem_idx = 0
        # for arg_index_read, arg_index_write in zip(
        #    indices_stream_to_read, indices_stream_to_write
        # ):

        p_func_arg_addr_lst = []
        for arg_index_read in indices_stream_to_read:
            # We store the address of the output array outside the loop
            func_arg_datatype = self.out_global_mem[global_mem_idx].typ
            p_func_arg_addr = AllocaOp.get(alloca_size, func_arg_datatype)
            store_func_arg_addr = StoreOp.get(
                self.out_global_mem[global_mem_idx], p_func_arg_addr
            )

            # Inside the loop, we do the pointer arithmetic to write on the next array element in every teration
            dummy_element = Constant.from_float_and_width(5.0, f64)

            copy_func_arg = LoadOp.get(p_func_arg_addr)
            write_op = StoreOp.get(stencil_return_vals[0], copy_func_arg)
            # write_op = StoreOp.get(dummy_element, copy_func_arg)
            incr_copy_func_arg_addr = GEPOp.get(
                copy_func_arg, [1], result_type=func_arg_datatype
            )
            update_copy_func_arg_addr = StoreOp.get(
                incr_copy_func_arg_addr, p_func_arg_addr
            )

            p_func_arg_addr_lst.append(p_func_arg_addr)

            stream_to_read = op.region.block.args[arg_index_read]
            # stream_to_write = op.region.block.args[arg_index_write]

            read_op = HLSStreamRead(stream_to_read)
            read_elem = read_op.res

            # TODO: operate on the elements of the stencil block read_elem. For now we are providing a dummy double

            # write_op = HLSStreamWrite(dummy_element, stream_to_write)

            # rewriter.insert_op_at_start([dummy_element, write_op], op.region.block)
            # TODO: this stream will be passsed to the write_data intrinsic, but for now we are just going to write to memory directly here.
            global_mem_idx += 1
            rewriter.insert_op_at_end(
                [
                    copy_func_arg,
                    write_op,
                    incr_copy_func_arg_addr,
                    update_copy_func_arg_addr,
                ],
                op.region.block,
            )

            rewriter.insert_op_at_start(read_op, op.region.block)

        # Transform ApplyOp into for loops
        get_number_chunks = FuncOp.external(
            "get_number_chunks",
            [builtin.IndexType(), LLVMPointerType.typed(i32)],
            [builtin.IndexType()],
        )

        get_chunk_size = FuncOp.external(
            "get_chunk_size",
            [builtin.IndexType(), builtin.IndexType(), i32, i32],
            [builtin.IndexType()],
        )

        self.module.body.block.add_op(get_number_chunks)
        self.module.body.block.add_op(get_chunk_size)

        res_type = op.res[0].typ

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

        # MAX_Y_SIZE = 16
        MAX_Y_SIZE = 8  # TODO: we use this size because our Y dimension is under 16 and that way the kernel doesn't finish. It's not a problem for size > 16
        max_chunk_length = Constant.from_int_and_width(MAX_Y_SIZE, i32)

        remainder = LoadOp.get(p_remainder)

        call_get_chunk_size = Call.get(
            "get_chunk_size",
            [chunk_num, call_get_number_chunks, max_chunk_length, remainder],
            [builtin.IndexType()],
        )

        p.body.block.insert_op_before(call_get_chunk_size, p.body.block.first_op)
        chunk_size_y_1 = arith.Subi(call_get_chunk_size, one)
        p.body.block.insert_op_after(chunk_size_y_1, call_get_chunk_size)

        old_operands_lst = [old_operand for old_operand in y_for_op.operands]
        y_for_op.operands = (
            [old_operands_lst[0]] + [chunk_size_y_1.results[0]] + old_operands_lst[2:]
        )

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
                upper_x,
                *lowerBounds,
                upper_chunks,
                upper_y,
                upper_z,
                alloca_size,
                *p_func_arg_addr_lst,
                store_func_arg_addr,
                p,
            ]
        )


@dataclass
class StencilExternalStoreToHLSWriteData(RewritePattern):
    module: builtin.ModuleOp
    out_data_streams: list
    write_data_declaration: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalStoreOp, rewriter: PatternRewriter, /):
        field = op.field
        temp = op.temp

        # Find the llvm.ptr to external memory that genrates the argument to the stencil.external_load. For PSyclone, this is
        # an argument to the parent function. TODO: this might need to be tested and generalised for other codes. Also, we are
        # considering that the function argument will be the second to insertvalue, but we're walking up trhough the second to
        # avoid bumping into arith.constants (see the mlir ssa).
        new_op = temp
        func_arg = None

        while not isa(func_arg, BlockArgument):
            assert isinstance(new_op.owner, Operation)
            func_arg = new_op.owner.operands[-1]
            new_op = new_op.owner.operands[0]

        if isa(func_arg.typ, LLVMPointerType):
            func_arg_elem_type = func_arg.typ.type
        else:
            func_arg_elem_type = func_arg.typ

        stream = self.out_data_streams[0]
        stream_type = stream.results[0].typ
        elem_type = stream.elem_type

        p_elem_type = LLVMPointerType.typed(LLVMStructType.from_type_list([elem_type]))

        shape = op.field.typ.shape

        shape_x = Constant.from_int_and_width(shape.data[0].value.data, i32)
        shape_y = Constant.from_int_and_width(shape.data[1].value.data, i32)
        shape_z = Constant.from_int_and_width(shape.data[2].value.data, i32)

        if not self.write_data_declaration:
            write_data = FuncOp.external(
                "write_data",
                [p_elem_type, LLVMPointerType.typed(f64), i32, i32, i32],
                [],
            )

            self.module.body.block.add_op(write_data)
            write_data_declaration = True

        call_write_data = Call.get(
            "write_data", [stream, func_arg, shape_x, shape_y, shape_z], []
        )

        rewriter.insert_op_after_matched_op(
            [shape_x, shape_y, shape_z, call_write_data]
        )


@dataclass
class StencilAccessOpToReadBlockOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        result_hls_read = None

        replace_access = False

        for use in op.temp.uses:
            if isinstance(use.operation, HLSStreamRead):
                hls_read = use.operation

                result_hls_read = hls_read.results[0]
                replace_access = True

        if replace_access:
            access_idx = []
            for idx in op.offset.array.data:
                access_idx.append(idx.data + 1)

            access_idx_array = DenseArrayBase.create_dense_int_or_index(
                i64, [0] + access_idx
            )

            stencil_value = ExtractValueOp(access_idx_array, result_hls_read, f64)

            rewriter.replace_matched_op(stencil_value)


# Copied from convert_stencil_to_ll_mlir
@dataclass
class StencilStoreToSubview(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        stores = [o for o in op.walk() if isinstance(o, StoreOp)]

        for store in stores:
            field = store.field
            assert isa(field.type, FieldType[Attribute])
            assert isa(field.type.bounds, StencilBoundsAttr)
            temp = store.temp
            assert isa(temp.type, TempType[Attribute])
            offsets = [i for i in -field.type.bounds.lb]
            sizes = [i for i in temp.type.get_shape()]
            subview = memref.Subview.from_static_parameters(
                field,
                StencilToMemRefType(field.type),
                offsets,
                sizes,
                [1] * len(sizes),
            )
            name = None
            if subview.source.name_hint:
                name = subview.source.name_hint + "_storeview"
            subview.result.name_hint = name
            if isinstance(field.owner, Operation):
                rewriter.insert_op_after(subview, field.owner)
            else:
                rewriter.insert_op_at_start(subview, field.owner)

            rewriter.erase_op(store)


class TrivialStoreOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


class TrivialApplyOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        rewriter.erase_matched_op()


@dataclass
class HLSConvertStencilToLLMLIRPass(ModulePass):
    name = "hls-convert-stencil-to-ll-mlir"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        load_data_declaration: bool = False
        module: builtin.ModuleOp = op
        shift_streams = []
        out_data_streams = []
        out_global_mem = []

        hls_pass = PatternRewriteWalker(
            # GreedyRewritePatternApplier([StencilExternalLoadToHLSExternalLoad(op), StencilAccessToGEP(op)]),
            GreedyRewritePatternApplier(
                [
                    StencilExternalLoadToHLSExternalLoad(
                        module, shift_streams, out_data_streams, out_global_mem
                    ),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        hls_pass.rewrite_module(op)

        adapt_stencil_pass = PatternRewriteWalker(
            # GreedyRewritePatternApplier([StencilExternalLoadToHLSExternalLoad(op), StencilAccessToGEP(op)]),
            GreedyRewritePatternApplier(
                [
                    ApplyOpToHLS(
                        module, shift_streams, out_data_streams, out_global_mem
                    ),
                    StencilAccessOpToReadBlockOp(),
                    StencilStoreToSubview(),
                    CastOpToMemref(),
                    LoadOpToMemref(),
                    AccessOpToMemref(),
                    TrivialExternalLoadOpCleanup(),
                    TrivialExternalStoreOpCleanup(),
                    TrivialStoreOpCleanup(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        adapt_stencil_pass.rewrite_module(op)

        clean_apply_pass = PatternRewriteWalker(
            # GreedyRewritePatternApplier([StencilExternalLoadToHLSExternalLoad(op), StencilAccessToGEP(op)]),
            GreedyRewritePatternApplier(
                [
                    TrivialApplyOpCleanup(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        clean_apply_pass.rewrite_module(op)

        write_data_pass = PatternRewriteWalker(
            # GreedyRewritePatternApplier([StencilExternalLoadToHLSExternalLoad(op), StencilAccessToGEP(op)]),
            GreedyRewritePatternApplier(
                [StencilExternalStoreToHLSWriteData(module, out_data_streams)]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        # write_data_pass.rewrite_module(op)
