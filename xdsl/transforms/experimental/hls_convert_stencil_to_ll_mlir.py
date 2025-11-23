import typing
from dataclasses import dataclass, field

from xdsl.builder import Builder
from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, llvm, memref, scf, stencil
from xdsl.dialects.arith import ConstantOp
from xdsl.dialects.builtin import (
    DenseArrayBase,
    IndexType,
    IntAttr,
    MemRefType,
    f64,
    i32,
    i64,
)
from xdsl.dialects.experimental.hls import (
    HLSExtractStencilValueOp,
    HLSStreamOp,
    HLSStreamReadOp,
    HLSStreamType,
    HLSStreamWriteOp,
    HLSYieldOp,
    PragmaDataflowOp,
    PragmaPipelineOp,
)
from xdsl.dialects.func import CallOp, FuncOp
from xdsl.dialects.llvm import (
    AllocaOp,
    InsertValueOp,
    LLVMArrayType,
    LLVMPointerType,
    LLVMStructType,
    LoadOp,
    UndefOp,
)
from xdsl.dialects.stencil import (
    AccessOp,
    ApplyOp,
    ExternalLoadOp,
    ExternalStoreOp,
    FieldType,
    ReturnOp,
    StencilBoundsAttr,
    TempType,
)
from xdsl.ir import (
    Attribute,
    Block,
    BlockArgument,
    Operation,
    OpResult,
    Region,
    SSAValue,
)
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.transforms.experimental.convert_stencil_to_ll_mlir import (
    AccessOpToMemRef,
    CastOpToMemRef,
    LoadOpToMemRef,
    StencilToMemRefType,
    TrivialExternalLoadOpCleanup,
    TrivialExternalStoreOpCleanup,
    prepare_apply_body,
)
from xdsl.utils.hints import isa

IN = 0
OUT = 1


# def add_pragma_interface(func_arg: BlockArgument, inout: int, kernel: FuncOp):
#    func_call = None
#    if inout is IN:
#        func_call = Call("IN", func_arg, [])
#    elif inout is OUT:
#        func_call = Call("OUT", func_arg, [])
#
#    kernel.body.block.insert_op_before(typing.cast(Call, func_call), typing.cast(Operation, kernel.body.block.first_op))


def gen_duplicate_loop(
    input_stream: HLSStreamOp,
    duplicate_stream_lst: list[HLSStreamOp],
    n: arith.IndexCastOp,
):
    ii = ConstantOp.from_int_and_width(1, i32)

    @Builder.region([IndexType()])
    def for_body(builder: Builder, args: tuple[BlockArgument, ...]):
        hls_pipeline_op = PragmaPipelineOp(ii)

        builder.insert(hls_pipeline_op)
        hls_read = HLSStreamReadOp(input_stream.results[0])
        builder.insert(hls_read)

        for duplicate_stream in duplicate_stream_lst:
            hls_write = HLSStreamWriteOp(hls_read, duplicate_stream)
            hls_write.attributes["duplicate"] = IntAttr(1)
            builder.insert(hls_write)

        yield_op = scf.YieldOp()
        builder.insert(yield_op)

    lb = ConstantOp.from_int_and_width(0, IndexType())
    ub = n
    step = ConstantOp.from_int_and_width(1, IndexType())

    for_duplicate = scf.ForOp(lb, ub, step, [], for_body)

    return [ii, lb, ub, step, for_duplicate]


@dataclass
class StencilExternalLoadToHLSExternalLoad(RewritePattern):
    module: builtin.ModuleOp
    shift_streams: list[list[HLSStreamOp]]
    out_data_streams: list[HLSStreamOp]
    out_global_mem: list[BlockArgument]
    load_data_declaration: bool = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
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

        func_arg_elem_type = func_arg.type

        # add_pragma_interface(func_arg, op.attributes["inout"].data, op.parent_op())
        assert isa(op.attributes["inout"], IntAttr)
        inout = op.attributes["inout"].data

        if op.attributes["inout"].data is OUT:
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

        LLVMStructType.from_type_list([func_arg_elem_type])
        LLVMStructType.from_type_list([stencil_type])

        field_type = field.type
        assert isinstance(field_type, MemRefType)
        shape = field_type.get_shape()

        if len(shape) < 3:
            return

        shape_x = ConstantOp.from_int_and_width(shape[0], i32)
        shape_y = ConstantOp.from_int_and_width(shape[1], i32)
        shape_z = ConstantOp.from_int_and_width(shape[2], i32)

        qualify_apply_op_with_shapes(
            typing.cast(FuncOp, op.parent_op()), shape_x, shape_y, shape_z
        )

        two_int = ConstantOp.from_int_and_width(2, i32)
        shift_shape_x = arith.SubiOp(shape_x, two_int)
        # TODO: generalise this
        data_stream = HLSStreamOp.get(f64)
        stencil_stream = HLSStreamOp.get(stencil_type)

        copy_stencil_stream_lst: list[HLSStreamOp] = []
        # TODO: we are generating 3 copies for now. This is what we need for pw_advection, but should be generalised for codes
        # with a different number of components
        n_components = 0
        for _op in typing.cast(FuncOp, op.parent_op()).body.blocks[0].ops:
            if isinstance(_op, ApplyOp):
                apply_op = _op
                for op_in_apply in apply_op.region.blocks[0].ops:
                    if isinstance(op_in_apply, stencil.ReturnOp):
                        return_op = op_in_apply
                        n_components = len(return_op.arg)

        copy_stencil_stream = HLSStreamOp.get(stencil_type)

        one_int = ConstantOp.from_int_and_width(1, i32)
        four_int = ConstantOp.from_int_and_width(4, i32)
        copy_shift_x = arith.SubiOp(shape_x, four_int)
        copy_shift_y = arith.SubiOp(shape_y, four_int)
        copy_shift_z = arith.SubiOp(shape_z, one_int)
        prod_x_y = arith.MuliOp(copy_shift_x, copy_shift_y)
        copy_n = arith.MuliOp(prod_x_y, copy_shift_z)

        inout = op.attributes["inout"].data

        data_stream.attributes["inout"] = op.attributes["inout"]
        stencil_stream.attributes["inout"] = op.attributes["inout"]
        copy_stencil_stream.attributes["inout"] = op.attributes["inout"]

        # We need to indicate that this is a stencil stream and not a data stream. TODO: make this more elegant
        stencil_stream.attributes["stencil"] = op.attributes["inout"]
        copy_stencil_stream.attributes["stencil"] = op.attributes["inout"]

        for _ in range(n_components):
            copy_stencil_stream_lst.append(copy_stencil_stream)

        threedload_call = CallOp(
            "dummy_load_data", [func_arg, data_stream, shape_x, shape_y, shape_z], []
        )

        @Builder.region
        def load_data_region(builder: Builder):
            yield_op = HLSYieldOp.get()
            builder.insert(threedload_call)
            builder.insert(yield_op)

        load_data_dataflow = PragmaDataflowOp(load_data_region)

        shift_buffer_call = CallOp(
            "shift_buffer",
            [data_stream, stencil_stream, shift_shape_x, shape_y, shape_z],
            [],
        )

        @Builder.region
        def shift_buffer_region(builder: Builder):
            yield_op = HLSYieldOp.get()
            builder.insert(shift_buffer_call)
            builder.insert(yield_op)

        shift_buffer_dataflow = PragmaDataflowOp(shift_buffer_region)

        n_idx = arith.IndexCastOp(copy_n, IndexType())

        duplicate_loop = gen_duplicate_loop(
            stencil_stream, copy_stencil_stream_lst, n_idx
        )

        @Builder.region
        def duplicateStream_region(builder: Builder):
            for dup_op in duplicate_loop:
                builder.insert(dup_op)
            yield_op = HLSYieldOp.get()
            builder.insert(yield_op)

        duplicateStream_dataflow = PragmaDataflowOp(duplicateStream_region)

        ndims = len(field_type.get_shape())
        if inout is IN and ndims == 3:
            rewriter.insert_op(
                [
                    data_stream,
                    stencil_stream,
                    *copy_stencil_stream_lst,
                    shape_x,
                    shape_y,
                    shape_z,
                    two_int,
                    shift_shape_x,
                    # threedload_call,
                    load_data_dataflow,
                    shift_buffer_dataflow,
                    one_int,
                    four_int,
                    copy_shift_x,
                    copy_shift_y,
                    copy_shift_z,
                    prod_x_y,
                    copy_n,
                    duplicateStream_dataflow,
                ]
            )
            self.shift_streams.append(copy_stencil_stream_lst)
        elif inout is OUT:
            out_data_stream = HLSStreamOp.get(f64)
            out_data_stream.attributes["inout"] = op.attributes["inout"]
            out_data_stream.attributes["data"] = op.attributes["inout"]
            rewriter.insert_op(
                [
                    out_data_stream,
                ]
            )
            self.out_data_streams.append(out_data_stream)

        if not self.load_data_declaration:
            shift_buffer_func = FuncOp.external(
                "shift_buffer",
                [
                    LLVMPointerType(),
                    LLVMPointerType(),
                    i32,
                    i32,
                    i32,
                ],
                [],
            )
            self.module.body.block.add_op(shift_buffer_func)

            self.load_data_declaration = True


def qualify_apply_op_with_shapes(
    stencil_func: FuncOp,
    shape_x: arith.ConstantOp,
    shape_y: arith.ConstantOp,
    shape_z: arith.ConstantOp,
):
    block = stencil_func.body.block

    for op in block.ops:
        if isinstance(op, ApplyOp):
            op.attributes["shape_x"] = shape_x.value
            op.attributes["shape_y"] = shape_y.value
            op.attributes["shape_z"] = shape_z.value


def add_read_write_ops(
    indices_stream_to_read: list[int],
    indices_stream_to_write: list[int],
    op: ApplyOp,
    rewriter: PatternRewriter,
    boilerplate: list[Operation],
):
    body_block = op.region.blocks[0]
    return_op = next(o for o in body_block.ops if isinstance(o, ReturnOp))
    stencil_return_vals: list[SSAValue] = [val for val in return_op.arg]

    ConstantOp.from_int_and_width(1, i32)

    stencil_idx = 0
    for arg_index_write in indices_stream_to_write:
        stream_to_write: BlockArgument = op.region.block.args[arg_index_write]
        write_op = HLSStreamWriteOp(stencil_return_vals[stencil_idx], stream_to_write)

        rewriter.insert_op(write_op, InsertPoint.at_end(op.region.block))

    for arg_index_read in indices_stream_to_read:
        stream_to_read = op.region.block.args[arg_index_read]

        read_op = HLSStreamReadOp(stream_to_read)
        read_op.attributes["write_data"] = IntAttr(1)

        rewriter.insert_op(read_op, InsertPoint.at_start(op.region.block))


def transform_apply_into_loop(
    op: ApplyOp, rewriter: PatternRewriter, ndim: int, boilerplate: list[Operation]
):
    dim: int = ndim
    assert dim == 3
    body = prepare_apply_body(op)

    assert isinstance(op.attributes["shape_x"], builtin.IntegerAttr)
    assert isinstance(op.attributes["shape_y"], builtin.IntegerAttr)
    assert isinstance(op.attributes["shape_z"], builtin.IntegerAttr)
    size_x = ConstantOp.from_int_and_width(
        op.attributes["shape_x"].value.data, builtin.IndexType()
    )
    size_y = ConstantOp.from_int_and_width(
        op.attributes["shape_y"].value.data, builtin.IndexType()
    )
    ConstantOp.from_int_and_width(
        op.attributes["shape_z"].value.data, builtin.IndexType()
    )
    one_int = ConstantOp.from_int_and_width(1, i32)
    two = ConstantOp.from_int_and_width(2, builtin.IndexType())
    zero = ConstantOp.from_int_and_width(0, builtin.IndexType())
    one = ConstantOp.from_int_and_width(1, builtin.IndexType())

    size_x_2 = arith.SubiOp(size_x, two)
    size_y_1 = arith.SubiOp(size_y, one)

    lower_x = ConstantOp.from_int_and_width(2, builtin.IndexType())
    lower_y = ConstantOp.from_int_and_width(1, builtin.IndexType())
    lower_z = ConstantOp.from_int_and_width(1, builtin.IndexType())
    upper_x = size_x_2
    upper_y = size_y_1
    upper_z = ConstantOp.from_int_and_width(
        op.attributes["shape_z"].value.data, builtin.IndexType()
    )

    p_remainder = AllocaOp(one_int, i32)

    call_get_number_chunks = CallOp(
        "get_number_chunks", [size_y, p_remainder], [builtin.IndexType()]
    )

    lower_chunks = zero
    upper_chunks = call_get_number_chunks

    lowerBounds = [lower_chunks, lower_x, lower_y, lower_z]
    upperBounds = [upper_chunks, upper_x, upper_y, upper_z]

    # The for loop for the y index receives its trip variable from the get_chunk_size function, since the chunking
    # is happening in the y axis. TODO: this is currently intended for the 3D case. It should be extended to the
    # 1D and 2D cases as well.
    y_for_op: scf.ForOp

    # Pipeline the loop
    ii = ConstantOp.from_int_and_width(1, i32)
    hls_pipeline_op = PragmaPipelineOp(ii)

    # current_region = for_body
    current_block = body
    for_op_lst: list[scf.ForOp] = []
    for i in range(1, dim + 1):
        for_op = scf.ForOp(
            lb=lowerBounds[-i],
            ub=upperBounds[-i],
            step=one,
            iter_args=[],
            body=current_block,
        )
        for_op_lst.append(for_op)
        current_block = Block(
            ops=[for_op, scf.YieldOp()], arg_types=[builtin.IndexType()]
        )

        # if i == 2:
        #    y_for_op = for_op

        if i == 1:
            for_op.body.blocks[0].insert_op_before(
                hls_pipeline_op, typing.cast(Operation, for_op.body.blocks[0].first_op)
            )
            for_op.body.blocks[0].insert_op_before(
                ii, typing.cast(Operation, for_op.body.blocks[0].first_op)
            )

    y_for_op = for_op_lst[1]
    p = scf.ParallelOp(
        lower_bounds=[lowerBounds[0]],
        upper_bounds=[upperBounds[0]],
        steps=[one],
        body=Region(current_block),
    )

    chunk_num = p.body.block.args[0]

    MAX_Y_SIZE = 16
    max_chunk_length = ConstantOp.from_int_and_width(MAX_Y_SIZE, i32)

    remainder = LoadOp(p_remainder, result_type=i32)

    call_get_chunk_size = CallOp(
        "get_chunk_size",
        [chunk_num, call_get_number_chunks, max_chunk_length, remainder],
        [builtin.IndexType()],
    )

    p.body.block.insert_op_before(
        call_get_chunk_size, typing.cast(Operation, p.body.block.first_op)
    )
    chunk_size_y_1 = arith.SubiOp(call_get_chunk_size, one)
    p.body.block.insert_op_after(chunk_size_y_1, call_get_chunk_size)

    old_operands_lst = [old_operand for old_operand in y_for_op.operands]
    y_for_op.operands = (
        [old_operands_lst[0]] + [chunk_size_y_1.results[0]] + old_operands_lst[2:]
    )

    p.attributes["compute_loop"] = IntAttr(1)

    @Builder.region
    def p_region(builder: Builder):
        builder.insert(p)
        builder.insert(HLSYieldOp.get())

    p_dataflow = PragmaDataflowOp(p_region)

    boilerplate += [
        size_x,
        size_y,
        one_int,
        one,
        two,
        max_chunk_length,
        p_remainder,
        upper_x,
        *lowerBounds,
        upper_chunks,
        upper_y,
        upper_z,
        remainder,
    ]

    return p_dataflow


@dataclass
class ApplyOpToHLS(RewritePattern):
    module: builtin.ModuleOp
    shift_streams: list[list[HLSStreamOp]]
    out_data_streams: list[HLSStreamOp]
    out_global_mem: list[BlockArgument]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        # We qualify the parent function as a kernel for futher processing
        typing.cast(Operation, op.parent_op()).attributes["kernel"] = IntAttr(1)

        body_block = op.region.blocks[0]
        return_op = next(o for o in body_block.ops if isinstance(o, ReturnOp))
        n_components = len(return_op.arg)
        apply_clones_lst: list[ApplyOp] = [op.clone() for _ in range(n_components)]

        # We replace the temp arguments by HLS streams. Only for the 3D temps
        for k in range(len(apply_clones_lst)):
            # Insert the HLS stream operands and their corresponding block arguments for
            # reading from the shift buffer and writing to external memory.
            # We replace by streams only the 3D temps.
            # The rest should be left as is operand_stream = dict()
            current_stream = 0

            new_operands_lst: list[OpResult] = []
            apply_clone = apply_clones_lst[k]

            for i in range(len(apply_clone.operands)):
                operand: OpResult = typing.cast(OpResult, apply_clone.operands[i])
                operand_type = operand.type
                assert isinstance(operand_type, TempType)
                assert isinstance(operand_type.bounds, StencilBoundsAttr)
                n_dims = len(operand_type.bounds.lb)

                if n_dims == 3:
                    stream = self.shift_streams[current_stream][k]
                    rewriter.replace_value_with_new_type(
                        apply_clone.region.block.args[i], stream.results[0].type
                    )

                    assert isa(stream.results, list[OpResult])
                    new_operands_lst.append(stream.results[0])
                    current_stream += 1
                else:
                    new_operands_lst.append(operand)

            apply_clone.operands = new_operands_lst + [
                self.out_data_streams[k].results[0]
            ]

        indices_stream_to_read: list[int] = []
        indices_stream_to_write: list[int] = []
        i = 0
        apply_clone = apply_clones_lst[-1]
        for _operand in apply_clone.operands:
            assert isinstance(_operand, BlockArgument) or isinstance(_operand, OpResult)
            if isinstance(_operand, OpResult) and isinstance(_operand.op, HLSStreamOp):
                assert isa(_operand.op.attributes["inout"], IntAttr)
                if (
                    "stencil" in _operand.op.attributes
                    and _operand.op.attributes["inout"].data is IN
                ):
                    indices_stream_to_read.append(i)
                if (
                    "data" in _operand.op.attributes
                    and _operand.op.attributes["inout"].data is OUT
                ):
                    indices_stream_to_write.append(i)
            i += 1

        for write_idx in indices_stream_to_write:
            apply_clone.region.blocks[0].insert_arg(
                self.out_data_streams[0].results[0].type, write_idx
            )

        # Get this apply's ReturnOp
        body_block = op.region.blocks[0]
        return_op = next(o for o in body_block.ops if isinstance(o, ReturnOp))

        # We are going to split the apply by the operations conducive to each returned value
        boilerplate: list[list[Operation]] = [[] for _ in range(len(return_op.arg))]
        new_apply_lst: list[ApplyOp] = []
        new_return_component_lst: list[ReturnOp] = []

        component_idx = 0
        k = 0
        new_apply: ApplyOp
        for component in return_op.arg:
            assert isinstance(component, OpResult)
            new_apply = apply_clones_lst[k]
            k += 1

            component_operations: dict[int, Operation] = dict()
            operation_indices: set[int] = set()
            component_operations[
                typing.cast(Block, return_op.parent_block()).get_operation_index(
                    return_op
                )
            ] = return_op
            operation_indices.add(
                typing.cast(Block, return_op.parent_block()).get_operation_index(
                    return_op
                )
            )
            collectComponentOperations(
                component, component_operations, operation_indices
            )

            new_apply_block = new_apply.region.blocks[0]
            new_return_op = next(
                o for o in new_apply_block.ops if isinstance(o, ReturnOp)
            )

            new_return_op.detach()
            new_return_op.erase()

            for operation in reversed(new_apply_block.ops):
                op_index = new_apply_block.get_operation_index(operation)
                if op_index not in operation_indices:
                    operation.detach()
                    operation.erase()

            new_component = typing.cast(Operation, new_apply_block.last_op).results[0]
            new_return_component = ReturnOp.get([new_component])
            new_apply_block.add_op(new_return_component)

            add_read_write_ops(
                indices_stream_to_read,
                indices_stream_to_write,
                new_apply,
                rewriter,
                boilerplate[component_idx],
            )
            new_apply_lst.append(new_apply)
            new_return_component_lst.append(new_return_component)
            component_idx += 1

        get_number_chunks = FuncOp.external(
            "get_number_chunks",
            [builtin.IndexType(), LLVMPointerType()],
            [builtin.IndexType()],
        )

        get_chunk_size = FuncOp.external(
            "get_chunk_size",
            [builtin.IndexType(), builtin.IndexType(), i32, i32],
            [builtin.IndexType()],
        )

        self.module.body.block.add_op(get_number_chunks)
        self.module.body.block.add_op(get_chunk_size)

        ndims: int = op.res[0].type.get_num_dims()

        rewriter.erase_op(return_op)
        for new_return_component in new_return_component_lst:
            rewriter.erase_op(new_return_component)

        p_dataflow_lst: list[PragmaDataflowOp] = []
        component_idx = 0
        for new_apply in new_apply_lst:
            p_dataflow = transform_apply_into_loop(
                new_apply, rewriter, ndims, boilerplate[component_idx]
            )
            p_dataflow_lst.append(p_dataflow)
            component_idx += 1

        operations_to_insert: list[Operation] = []
        for i in range(n_components):
            operations_to_insert += boilerplate[i] + [p_dataflow_lst[i]]

        rewriter.insert_op(operations_to_insert)

        rewriter.replace_op(op, new_apply_lst[-1])


def collectComponentOperations(
    op: OpResult,
    component_operations: dict[int, Operation],
    operation_indices: set[int],
):
    parent_op = op.owner

    for operand in parent_op.operands:
        if not isinstance(operand, BlockArgument):
            assert isinstance(operand, OpResult)
            block_index = typing.cast(
                Block, operand.op.parent_block()
            ).get_operation_index(operand.op)
            component_operations[block_index] = operand.op
            operation_indices.add(block_index)
            collectComponentOperations(operand, component_operations, operation_indices)


def get_number_external_stores(op: FuncOp):
    external_stores_lst = [
        o for o in op.body.blocks[0].ops if isinstance(o, ExternalStoreOp)
    ]

    return len(external_stores_lst)


@dataclass
class StencilExternalStoreToHLSWriteData(RewritePattern):
    module: builtin.ModuleOp
    out_data_streams: list[HLSStreamType]
    write_data_declaration: bool = False
    func_args_lst: list[BlockArgument] = field(default_factory=list[BlockArgument])
    n_args: int = 0
    total_args: int | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalStoreOp, rewriter: PatternRewriter, /):
        if self.total_args is None:
            self.total_args = get_number_external_stores(
                typing.cast(func.FuncOp, op.parent_op())
            )
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

        self.func_args_lst.append(func_arg)
        self.n_args += 1

        if self.n_args == self.total_args:
            write_data_func_name = f"write_data_{self.total_args}"
            packed_type = LLVMPointerType()
            assert isinstance(self.out_data_streams[0], HLSStreamOp)
            out_data_stream_type = LLVMPointerType()
            write_data_func_args_lst = (
                self.total_args * [out_data_stream_type]
                + self.total_args * [packed_type]
                + 3 * [i32]
            )
            write_data_func = FuncOp.external(
                write_data_func_name,
                write_data_func_args_lst,
                [],
            )
            self.module.body.block.add_op(write_data_func)

            func_arg = None

            while not isa(func_arg, BlockArgument):
                assert isinstance(new_op.owner, Operation)
                func_arg = new_op.owner.operands[-1]

            op_field_type = op.field.type
            assert isinstance(op_field_type, MemRefType)
            shape = op_field_type.shape
            shape_x = ConstantOp.from_int_and_width(shape.data[0].data, i32)
            shape_y = ConstantOp.from_int_and_width(shape.data[1].data, i32)
            shape_z = ConstantOp.from_int_and_width(shape.data[2].data, i32)

            call_write_data = CallOp(
                write_data_func_name,
                [
                    *[
                        typing.cast(Operation, stream).results[0]
                        for stream in self.out_data_streams
                    ],
                    *reversed(self.func_args_lst),
                    shape_x,
                    shape_y,
                    shape_z,
                ],
                [],
            )

            @Builder.region
            def write_data_df_region(builder: Builder):
                builder.insert(call_write_data)
                hls_yield_op = HLSYieldOp.get()
                builder.insert(hls_yield_op)

            write_data_dataflow = PragmaDataflowOp(write_data_df_region)

            rewriter.insert_op(
                [shape_x, shape_y, shape_z, write_data_dataflow], InsertPoint.after(op)
            )


@dataclass
class StencilAccessOpToReadBlockOp(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: AccessOp, rewriter: PatternRewriter, /):
        result_hls_read: OpResult | None = None

        replace_access = False

        for use in op.temp.uses:
            if (
                isinstance(use.operation, HLSStreamReadOp)
                and use.operation.parent_op() == op.parent_op()
            ):
                hls_read = use.operation

                assert isinstance(hls_read.results[0], OpResult)
                result_hls_read = hls_read.results[0]
                replace_access = True

        if replace_access:
            access_idx: list[int] = []
            assert isa(op.offset.array, builtin.ArrayAttr[IntAttr])
            for idx in op.offset.array.data:
                access_idx.append(idx.data + 1)

            access_idx_array = DenseArrayBase.from_list(i64, [0] + access_idx)

            assert isinstance(result_hls_read, OpResult)
            stencil_value = HLSExtractStencilValueOp(
                access_idx_array, result_hls_read, f64
            )

            rewriter.replace_op(op, stencil_value)


# Copied from convert_stencil_to_ll_mlir
@dataclass
class StencilStoreToSubview(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: FuncOp, rewriter: PatternRewriter, /):
        stores = [o for o in op.walk() if isinstance(o, stencil.StoreOp)]

        for store in stores:
            field = store.field
            assert isa(field.type, FieldType[Attribute])
            assert isa(field.type.bounds, StencilBoundsAttr)
            temp = store.temp
            assert isa(temp.type, TempType[Attribute])
            offsets = [i for i in -field.type.bounds.lb]
            sizes = [i for i in temp.type.get_shape()]
            subview = memref.SubviewOp.from_static_parameters(
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
                rewriter.insert_op(subview, InsertPoint.after(field.owner))
            else:
                rewriter.insert_op(subview, InsertPoint.at_start(field.owner))

            rewriter.erase_op(store)


@dataclass
class TrivialStoreOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: stencil.StoreOp, rewriter: PatternRewriter, /):
        rewriter.erase_op(op)


@dataclass
class TrivialApplyOpCleanup(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ApplyOp, rewriter: PatternRewriter, /):
        rewriter.erase_op(op)


@dataclass
class QualifyAllArgumentsAsOut(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
        op.attributes["inout"] = IntAttr(IN)


@dataclass
class GetInoutAttributeFromExternalStore(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalStoreOp, rewriter: PatternRewriter, /):
        for use in op.field.uses:
            if isinstance(use.operation, ExternalLoadOp):
                use.operation.attributes["inout"] = IntAttr(OUT)


def get_number_input_stencils(op: FuncOp):
    # ndims = len(field.typ.get_shape())
    def dim(o: ExternalLoadOp):
        assert isinstance(o.field, OpResult)
        op_field_type = o.field.type
        assert isinstance(op_field_type, memref.MemRefType)
        return len(op_field_type.get_shape())

    external_load_lst = [
        o
        for o in op.body.blocks[0].ops
        if isinstance(o, ExternalLoadOp) and dim(o) == 3
    ]

    n = sum(
        [
            1
            for o in external_load_lst
            if typing.cast(IntAttr, o.attributes["inout"]).data == IN
        ]
    )

    return n


@dataclass
class GroupLoadsUnderSameDataflow(RewritePattern):
    module: builtin.ModuleOp
    first_load: CallOp | None = None
    # sizes: IntegerType | None = None
    sizes: list[OpResult | SSAValue] = field(default_factory=list[OpResult | SSAValue])
    in_module_load_all_data_func: FuncOp | None = None
    n_current_load: int = 0
    n_input: int = -1

    data_arrays: list[BlockArgument] = field(default_factory=list[BlockArgument])
    data_streams: list[OpResult] = field(default_factory=list[OpResult])

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: CallOp, rewriter: PatternRewriter, /):
        if op.callee.root_reference.data == "dummy_load_data":
            self.n_input = get_number_input_stencils(
                typing.cast(
                    func.FuncOp,
                    typing.cast(PragmaDataflowOp, op.parent_op()).parent_op(),
                )
            )
        if (
            op.callee.root_reference.data == "dummy_load_data"
            and self.n_current_load < self.n_input
        ):
            self.n_current_load += 1

            self.data_arrays.append(typing.cast(BlockArgument, op.operands[0]))
            self.data_streams.append(typing.cast(OpResult, op.operands[1]))

            # We are using the same sizes for all the load_data operations. We remove the duplicates
            if self.first_load is None:
                self.first_load = op
                self.sizes += list(
                    op.operands[2:]
                )  # [operand for operand in op.operands[2:]]
            else:
                parent_dataflow = typing.cast(PragmaDataflowOp, op.parent_op())
                rewriter.erase_op(op)
                parent_dataflow.detach()
                parent_dataflow.erase()

            if self.n_current_load == self.n_input:
                assert isinstance(self.first_load, CallOp)
                assert isinstance(self.first_load.arguments[1].type, HLSStreamType)
                load_data_func_name = f"load_data_{self.n_input}"
                hls_stream_type = LLVMPointerType()
                load_data_args_lst = (
                    self.n_input * [self.first_load.arguments[0].type]
                    + self.n_input * [hls_stream_type]
                    + 3 * [i32]
                )
                load_data_func = FuncOp.external(
                    load_data_func_name,
                    load_data_args_lst,
                    [],
                )

                self.module.body.block.add_op(load_data_func)

                load_all_data_args = self.data_arrays + self.data_streams + self.sizes
                assert isa(load_all_data_args, list[SSAValue])

                call_load_all_data = CallOp(
                    load_data_func_name,
                    load_all_data_args,
                    [],
                )

                parent_dataflow = typing.cast(
                    PragmaDataflowOp, self.first_load.parent_op()
                )

                rewriter.replace_op(self.first_load, call_load_all_data)

                for data_stream in self.data_streams:
                    data_stream.op.detach()
                    rewriter.insert_op(
                        data_stream.op, InsertPoint.before(parent_dataflow)
                    )


@dataclass
class PackData(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ExternalLoadOp, rewriter: PatternRewriter, /):
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

        field_type = field.type
        assert isinstance(field_type, memref.MemRefType)
        shape = field_type.get_shape()
        ndims = len(shape)

        arg_idx = func_arg.index
        parent_func = typing.cast(func.FuncOp, op.parent_op())

        assert isa(op.attributes["inout"], IntAttr)
        if op.attributes["inout"].data == OUT or (
            op.attributes["inout"].data == IN and ndims == 3
        ):
            # TODO: this should be generalised by packaging the original type instead of
            # f64. We would need intrinsics to deal with the different types
            packed_type = LLVMPointerType()
            parent_func.replace_argument_type(arg_idx, packed_type, rewriter)

            for use in func_arg.uses:
                if isinstance(use.operation, InsertValueOp):
                    insertvalue = use.operation

                    assert isinstance(insertvalue.container, OpResult)
                    container_op = insertvalue.container.op
                    if isinstance(container_op, UndefOp):
                        # We mark the UndefOp to update its type in the next pass and
                        # also update the type returned by the insertvalue operation
                        # that uses it

                        container_op.attributes["replace"] = IntAttr(0)

                        # Update the return types of the chain of insertvalues used to generatate the field structure
                        assert isinstance(insertvalue.res.type, llvm.LLVMStructType)
                        field_struct = insertvalue.res.type

                        old_type = list(field_struct.types.data)
                        new_type = old_type
                        new_type[0] = packed_type
                        new_type[1] = packed_type
                        struct_new_type = LLVMStructType.from_type_list(new_type)

                        current_insertvalue = insertvalue
                        rewriter.replace_value_with_new_type(
                            current_insertvalue.res, struct_new_type
                        )
                        update_types_insertvalue(
                            current_insertvalue, struct_new_type, rewriter
                        )


def update_types_insertvalue(
    op: InsertValueOp, new_type: llvm.LLVMStructType, rewriter: PatternRewriter
):
    for use in op.res.uses:
        if isinstance(use.operation, InsertValueOp):
            rewriter.replace_value_with_new_type(use.operation.res, new_type)
            update_types_insertvalue(use.operation, new_type, rewriter)


@dataclass
class PackDataInStencilField(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: UndefOp, rewriter: PatternRewriter, /):
        if "replace" in op.attributes:
            # packed_type = LLVMPointerType.typed(LLVMArrayType.from_size_and_type(8, f64))
            packed_type = LLVMPointerType()
            assert isinstance(op.res.type, llvm.LLVMStructType)
            field_struct = op.res.type

            old_type = list(field_struct.types.data)
            new_type = old_type
            new_type[0] = packed_type
            new_type[1] = packed_type

            struct_new_type = LLVMStructType.from_type_list(new_type)

            new_container_op = UndefOp(struct_new_type)

            rewriter.replace_op(op, new_container_op)


# We create copies for all the coefficients. We create more than one copy where necesssary
@dataclass
class GetRepeatedCoefficients(RewritePattern):
    original_memref_lst: list[memref.CastOp]
    clone_memref_lst: list[memref.AllocaOp]

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: memref.SubviewOp, rewriter: PatternRewriter, /):
        assert isinstance(op.source, OpResult)
        assert isinstance(op.source.op, memref.CastOp)
        cast = op.source.op  # original memref
        cast_dest_type = cast.dest.type
        assert isinstance(cast_dest_type, memref.MemRefType)
        cast.clone()

        dim = len(cast_dest_type.shape.data)
        cast_dest_type.shape.data[0].data
        if dim == 1:
            uses_copy = set(op.results[0].uses)
            for use in uses_copy:
                if isinstance(use.operation, memref.LoadOp):
                    memref_copy = memref.AllocaOp.get(
                        return_type=f64, shape=cast_dest_type.shape
                    )
                    use.operation.operands[0] = memref_copy.results[0]
                    rewriter.insert_op(memref_copy)

                    self.original_memref_lst.append(cast)
                    self.clone_memref_lst.append(memref_copy)
                elif isinstance(use.operation, stencil.ApplyOp):
                    op.results[0].remove_use(use)

            rewriter.erase_op(op)


@dataclass
class MakeLocaCopiesOfCoefficients(RewritePattern):
    original_memref_lst: list[memref.CastOp]
    clone_memref_lst: list[memref.AllocaOp]
    inserted_already = False

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: PragmaDataflowOp, rewriter: PatternRewriter, /):
        if (
            "compute_loop"
            in typing.cast(Operation, op.body.blocks[0].first_op).attributes
            and not self.inserted_already
            and len(self.original_memref_lst) > 0
        ):
            original_memref_lst_dest_type = self.original_memref_lst[0].dest.type
            assert isinstance(original_memref_lst_dest_type, memref.MemRefType)
            dim = original_memref_lst_dest_type.shape.data[0].data

            lb = ConstantOp.from_int_and_width(0, IndexType())
            ub = ConstantOp.from_int_and_width(dim, IndexType())
            step = ConstantOp.from_int_and_width(1, IndexType())

            ii = ConstantOp.from_int_and_width(1, i32)

            @Builder.region([IndexType()])
            def for_body(builder: Builder, args: tuple[BlockArgument, ...]):
                hls_pipeline_op = PragmaPipelineOp(ii)
                builder.insert(hls_pipeline_op)
                for i in range(len(self.original_memref_lst)):
                    load_op = memref.LoadOp.get(self.original_memref_lst[i], [args[0]])
                    store_op = memref.StoreOp.get(
                        load_op, self.clone_memref_lst[i], [args[0]]
                    )
                    builder.insert(load_op)
                    builder.insert(store_op)

                yield_op = scf.YieldOp()
                builder.insert(yield_op)

            for_local_copies = scf.ForOp(lb, ub, step, [], for_body)
            rewriter.insert_op([lb, ub, step, ii, for_local_copies])

            self.inserted_already = True


@dataclass
class TrivialCleanUpAuxAttributes(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ParallelOp, rewriter: PatternRewriter, /):
        if "compute_loop" in op.attributes:
            del op.attributes["compute_loop"]


@dataclass
class QualifyInterfacesPass(RewritePattern):
    module: builtin.ModuleOp
    declared_coeff_func: bool = False
    called_coeff_func: bool = False
    interface_coeff_func_name = "_maxi_coeff"

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: func.FuncOp, rewriter: PatternRewriter, /):
        arg_idx = 0

        if "kernel" in op.attributes:
            del op.attributes["kernel"]

            for _ in op.function_type.inputs:
                call_interface_func = llvm.CallOp(
                    self.interface_coeff_func_name, op.body.blocks[0].args[arg_idx]
                )
                rewriter.insert_op(
                    call_interface_func, InsertPoint.at_start(op.body.blocks[0])
                )
                self.called_coeff_func = True

                arg_idx += 1

        if self.called_coeff_func and not self.declared_coeff_func:
            interface_coeff_func_type = llvm.LLVMFunctionType([], None, True)
            interface_coeff_func = llvm.FuncOp(
                self.interface_coeff_func_name,
                interface_coeff_func_type,
                llvm.LinkageAttr("external"),
            )
            self.module.body.block.add_op(interface_coeff_func)
            self.declared_coeff_func = True


@dataclass(frozen=True)
class HLSConvertStencilToLLMLIRPass(ModulePass):
    name = "hls-convert-stencil-to-ll-mlir"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        module: builtin.ModuleOp = op
        shift_streams: list[list[HLSStreamOp]] = []
        out_data_streams: list[HLSStreamOp] = []
        out_global_mem: list[BlockArgument] = []

        inout_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    QualifyAllArgumentsAsOut(),
                    GetInoutAttributeFromExternalStore(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=False,
        )
        inout_pass.rewrite_module(op)

        pack_data_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    PackData(),
                    PackDataInStencilField(),
                ]
            ),
            apply_recursively=True,
            walk_reverse=True,
        )
        pack_data_pass.rewrite_module(op)

        hls_pass = PatternRewriteWalker(
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
            GreedyRewritePatternApplier(
                [
                    ApplyOpToHLS(
                        module, shift_streams, out_data_streams, out_global_mem
                    ),
                    StencilAccessOpToReadBlockOp(),
                    StencilStoreToSubview(),
                    CastOpToMemRef(),
                    LoadOpToMemRef(),
                    AccessOpToMemRef(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        adapt_stencil_pass.rewrite_module(op)

        write_data_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    StencilExternalStoreToHLSWriteData(module, []),
                    # TrivialExternalLoadOpCleanup(),
                    TrivialExternalStoreOpCleanup(),
                    TrivialStoreOpCleanup(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=True,
        )
        write_data_pass.rewrite_module(op)

        grouploads_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier([GroupLoadsUnderSameDataflow(op)]),
            apply_recursively=False,
            walk_reverse=False,
        )
        grouploads_pass.rewrite_module(op)

        cleanup_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    TrivialExternalLoadOpCleanup(),
                    TrivialExternalStoreOpCleanup(),
                    TrivialStoreOpCleanup(),
                ]
            ),
            apply_recursively=False,
            walk_reverse=False,
        )
        cleanup_pass.rewrite_module(op)

        clean_apply_pass = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [TrivialApplyOpCleanup()]  # , GroupLoadsUnderSameDataflow(op)]
            ),
            apply_recursively=False,
            walk_reverse=False,
        )
        clean_apply_pass.rewrite_module(op)

        original_memref_lst: list[memref.CastOp] = []
        clone_memref_lst: list[memref.AllocaOp] = []

        get_repeated = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    GetRepeatedCoefficients(original_memref_lst, clone_memref_lst),
                ]
            ),
            apply_recursively=True,
            walk_reverse=True,
        )
        get_repeated.rewrite_module(op)

        make_local_copies = PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [MakeLocaCopiesOfCoefficients(original_memref_lst, clone_memref_lst)]
            ),
            apply_recursively=True,
            walk_reverse=False,
        )
        make_local_copies.rewrite_module(op)

        final_cleanup = PatternRewriteWalker(
            GreedyRewritePatternApplier([TrivialCleanUpAuxAttributes()]),
            apply_recursively=True,
            walk_reverse=False,
        )
        final_cleanup.rewrite_module(op)

        interfaces_pass = PatternRewriteWalker(
            QualifyInterfacesPass(op),
            apply_recursively=True,
            walk_reverse=False,
        )
        interfaces_pass.rewrite_module(op)
