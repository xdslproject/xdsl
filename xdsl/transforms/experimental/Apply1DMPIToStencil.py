from xdsl.context import Context
from xdsl.dialects import arith, builtin, llvm, memref, mpi, scf, stencil
from xdsl.ir import Operation, TypeAttribute
from xdsl.irdl import Operand
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint
from xdsl.utils.hints import isa

AnyNumericType = builtin.AnyFloat | builtin.IntegerType


class ApplyMPIToExternalLoad(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: stencil.ExternalLoadOp, rewriter: PatternRewriter, /
    ):
        assert isa(op.field.type, memref.MemRefType[AnyNumericType])
        memref_type: memref.MemRefType[AnyNumericType] = op.field.type
        if len(memref_type.shape) <= 1:
            return
        mpi_operations: list[Operation] = []

        # Rank and size
        comm_size_op = mpi.CommSizeOp()
        comm_rank_op = mpi.CommRankOp()
        mpi_operations += [comm_size_op, comm_rank_op]

        # Constants we need to use
        zero = arith.ConstantOp.from_int_and_width(0, 32)
        one = arith.ConstantOp.from_int_and_width(1, 32)
        one_i64 = arith.ConstantOp.from_int_and_width(1, 64)
        two = arith.ConstantOp.from_int_and_width(2, 32)
        two_i64 = arith.ConstantOp.from_int_and_width(2, 64)
        three = arith.ConstantOp.from_int_and_width(3, 32)
        four = arith.ConstantOp.from_int_and_width(4, 32)
        eight_i64 = arith.ConstantOp.from_int_and_width(8, 64)
        mpi_operations += [zero, one, one_i64, two, two_i64, three, four, eight_i64]

        # The underlying datatype we use in communications and size in dimension zero
        element_type: TypeAttribute = memref_type.element_type
        datatype_op = mpi.GetDtypeOp(element_type)
        int_attr: builtin.IntegerAttr[builtin.IndexType] = builtin.IntegerAttr(
            0, builtin.IndexType()
        )
        dim_zero_const = arith.ConstantOp(int_attr, builtin.IndexType())
        dim_zero_size_op = memref.DimOp.from_source_and_index(op.field, dim_zero_const)
        dim_zero_i32_op = arith.IndexCastOp(dim_zero_size_op, builtin.i32)
        dim_zero_i64_op = arith.IndexCastOp(dim_zero_size_op, builtin.i64)

        index_memref = memref.ExtractAlignedPointerAsIndexOp.get(op.field)
        index_memref_i64 = arith.IndexCastOp(index_memref, builtin.i64)

        mpi_operations += [
            datatype_op,
            dim_zero_const,
            dim_zero_size_op,
            dim_zero_i32_op,
            dim_zero_i64_op,
            index_memref,
            index_memref_i64,
        ]

        # Four request handles, one for send, one for recv
        alloc_request_op = mpi.AllocateTypeOp(mpi.RequestType, four)
        mpi_operations += [alloc_request_op]

        # Comparison for top and bottom ranks
        compare_top_op = arith.CmpiOp(comm_rank_op, zero, "sgt")
        size_minus_one = arith.SubiOp(comm_size_op, one)
        compare_bottom_op = arith.CmpiOp(comm_rank_op, size_minus_one, "slt")
        mpi_operations += [compare_top_op, size_minus_one, compare_bottom_op]

        # MPI Request look ups (we need these regardless)
        alloc_lookup_op_zero = mpi.VectorGetOp(alloc_request_op, zero)
        alloc_lookup_op_one = mpi.VectorGetOp(alloc_request_op, one)
        alloc_lookup_op_two = mpi.VectorGetOp(alloc_request_op, two)
        alloc_lookup_op_three = mpi.VectorGetOp(alloc_request_op, three)
        mpi_request_null = arith.ConstantOp.from_int_and_width(0x2C000000, builtin.i32)
        mpi_operations += [
            alloc_lookup_op_zero,
            alloc_lookup_op_one,
            alloc_lookup_op_two,
            alloc_lookup_op_three,
            mpi_request_null,
        ]

        # Send and recv my first row of data to rank -1
        rank_m1_op = arith.SubiOp(comm_rank_op, one)

        add_offset = arith.MuliOp(dim_zero_i64_op, eight_i64)
        added_ptr = arith.AddiOp(index_memref_i64, add_offset)
        send_ptr = llvm.IntToPtrOp(added_ptr)

        mpi_send_top_op = mpi.IsendOp(
            send_ptr,
            dim_zero_i32_op,
            datatype_op,
            rank_m1_op,
            zero,
            alloc_lookup_op_zero,
        )

        recv_ptr = llvm.IntToPtrOp(index_memref_i64)
        mpi_recv_top_op = mpi.IrecvOp(
            recv_ptr,
            dim_zero_i32_op,
            datatype_op,
            rank_m1_op,
            zero,
            alloc_lookup_op_one,
        )

        # Else set empty request handles
        zero_conv = builtin.UnrealizedConversionCastOp.get(
            [alloc_lookup_op_zero], [llvm.LLVMPointerType()]
        )
        null_req_zero = llvm.StoreOp(mpi_request_null, zero_conv)
        one_conv = builtin.UnrealizedConversionCastOp.get(
            [alloc_lookup_op_one], [llvm.LLVMPointerType()]
        )
        null_req_one = llvm.StoreOp(mpi_request_null, one_conv)

        top_halo_exhange = scf.IfOp(
            compare_top_op,
            [],
            [
                rank_m1_op,
                add_offset,
                added_ptr,
                send_ptr,
                mpi_send_top_op,
                recv_ptr,
                mpi_recv_top_op,
                scf.YieldOp(),
            ],
            [zero_conv, null_req_zero, one_conv, null_req_one, scf.YieldOp()],
        )
        mpi_operations += [top_halo_exhange]

        # Send and recv my last row of data to rank +1
        rank_p1_op = arith.AddiOp(comm_rank_op, one)

        # Need to multiple row by column -2 (for data row)
        col_row_b_send = arith.SubiOp(dim_zero_i64_op, two_i64)
        element_b_send = arith.MuliOp(col_row_b_send, dim_zero_i64_op)
        add_offset_b_send = arith.MuliOp(element_b_send, eight_i64)
        added_ptr_b_send = arith.AddiOp(index_memref_i64, add_offset_b_send)
        ptr_b_send = llvm.IntToPtrOp(added_ptr_b_send)

        mpi_send_bottom_op = mpi.IsendOp(
            ptr_b_send,
            dim_zero_i32_op,
            datatype_op,
            rank_p1_op,
            zero,
            alloc_lookup_op_two,
        )

        # Now do the recv

        col_row_b_recv = arith.SubiOp(dim_zero_i64_op, one_i64)
        element_b_recv = arith.MuliOp(col_row_b_recv, dim_zero_i64_op)
        add_offset_b_recv = arith.MuliOp(element_b_recv, eight_i64)
        added_ptr_b_recv = arith.AddiOp(index_memref_i64, add_offset_b_recv)
        ptr_b_recv = llvm.IntToPtrOp(added_ptr_b_recv)

        mpi_recv_bottom_op = mpi.IrecvOp(
            ptr_b_recv,
            dim_zero_i32_op,
            datatype_op,
            rank_p1_op,
            zero,
            alloc_lookup_op_three,
        )

        # Else set empty request handles
        two_conv = builtin.UnrealizedConversionCastOp.get(
            [alloc_lookup_op_two], [llvm.LLVMPointerType()]
        )
        null_req_two = llvm.StoreOp(mpi_request_null, two_conv)
        three_conv = builtin.UnrealizedConversionCastOp.get(
            [alloc_lookup_op_three], [llvm.LLVMPointerType()]
        )
        null_req_three = llvm.StoreOp(mpi_request_null, three_conv)

        bottom_halo_exhange = scf.IfOp(
            compare_bottom_op,
            [],
            [
                rank_p1_op,
                col_row_b_send,
                element_b_send,
                add_offset_b_send,
                added_ptr_b_send,
                ptr_b_send,
                mpi_send_bottom_op,
                col_row_b_recv,
                element_b_recv,
                add_offset_b_recv,
                added_ptr_b_recv,
                ptr_b_recv,
                mpi_recv_bottom_op,
                scf.YieldOp(),
            ],
            [two_conv, null_req_two, three_conv, null_req_three, scf.YieldOp()],
        )

        mpi_operations += [bottom_halo_exhange]
        req_ops: Operand = alloc_request_op.results[0]

        wait_op = mpi.WaitallOp(req_ops, four.results[0])
        mpi_operations += [wait_op]

        rewriter.insert_op(mpi_operations, InsertPoint.after(op))


def Apply1DMpi(ctx: Context, module: builtin.ModuleOp):
    applyMPI = ApplyMPIToExternalLoad()
    walker1 = PatternRewriteWalker(
        GreedyRewritePatternApplier(
            [
                applyMPI,
            ]
        ),
        apply_recursively=True,
    )
    walker1.rewrite_module(module)
