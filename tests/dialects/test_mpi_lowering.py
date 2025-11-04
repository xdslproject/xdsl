from xdsl.dialects import arith, builtin, func, llvm, mpi
from xdsl.dialects.builtin import i32, i64
from xdsl.ir import Attribute, Operation, OpResult
from xdsl.irdl import (
    IRDLOperation,
    Operand,
    irdl_op_definition,
    var_result_def,
)
from xdsl.pattern_rewriter import PatternRewriteWalker
from xdsl.transforms import lower_mpi

info = lower_mpi.MpiLibraryInfo()


def extract_func_call(ops: list[Operation], name: str = "MPI_") -> func.CallOp | None:
    for op in ops:
        if isinstance(op, func.CallOp) and op.callee.string_value().startswith(name):
            return op


def check_emitted_function_signature(
    ops: list[Operation],
    name: str,
    types: tuple[type[Attribute] | None, ...],
):
    call = extract_func_call(ops, name)
    assert call is not None, f"Missing {func.CallOp.name} op to {name} in output!"
    assert len(call.arguments) == len(types)
    for arg, arg_type in zip(call.arguments, types):
        # check that the argument type is correct (if constraint present)
        if arg_type is not None:
            assert isinstance(arg.type, arg_type), (
                f"Expected argument to be of type {arg_type} (got {arg.type} instead)"
            )


@irdl_op_definition
class CreateTestValsOp(IRDLOperation):
    name = "testing.test"
    result = var_result_def()

    @staticmethod
    def get(*types: Attribute):
        return CreateTestValsOp.build(result_types=[list(types)])


def test_lower_mpi_init():
    ops, result = lower_mpi.LowerMpiInit(info).lower(mpi.InitOp.build())

    assert len(result) == 0
    assert len(ops) == 2

    nullop, call = ops

    assert isinstance(call, func.CallOp)
    assert isinstance(nullop, llvm.ZeroOp)
    assert call.callee.string_value() == "MPI_Init"
    assert len(call.arguments) == 2
    assert all(arg == nullop.res for arg in call.arguments)


def test_lower_mpi_finalize():
    ops, result = lower_mpi.LowerMpiFinalize(info).lower(mpi.FinalizeOp.build())

    assert len(result) == 0
    assert len(ops) == 1

    (call,) = ops

    assert isinstance(call, func.CallOp)
    assert call.callee.string_value() == "MPI_Finalize"
    assert len(call.arguments) == 0


def test_lower_mpi_wait_no_status():
    (request,) = CreateTestValsOp.get(mpi.RequestType()).results

    ops, result = lower_mpi.LowerMpiWait(info).lower(mpi.WaitOp(request))

    assert len(result) == 0
    call = extract_func_call(ops)
    assert call is not None
    assert call.callee.string_value() == "MPI_Wait"
    assert len(call.arguments) == 2


def test_lower_mpi_wait_with_status():
    (request,) = CreateTestValsOp.get(mpi.RequestType()).results

    ops, result = lower_mpi.LowerMpiWait(info).lower(
        mpi.WaitOp(request, ignore_status=False)
    )

    assert len(result) == 1
    assert result[0] is not None
    assert isinstance(result[0].type, llvm.LLVMPointerType)
    call = extract_func_call(ops)
    assert call is not None
    assert call.callee.string_value() == "MPI_Wait"
    assert len(call.arguments) == 2
    assert isinstance(call.arguments[1], OpResult)
    assert isinstance(call.arguments[1].owner, llvm.AllocaOp)


def test_lower_mpi_comm_rank():
    ops, result = lower_mpi.LowerMpiCommRank(info).lower(mpi.CommRankOp())

    assert len(result) == 1
    assert result[0] is not None
    assert result[0].type == i32

    # check signature of emitted function call
    # int MPI_Comm_rank(MPI_Comm comm, int *rank)
    check_emitted_function_signature(
        ops,
        "MPI_Comm_rank",
        (None, llvm.LLVMPointerType),
    )


def test_lower_mpi_comm_size():
    ops, result = lower_mpi.LowerMpiCommSize(info).lower(mpi.CommSizeOp())

    assert len(result) == 1
    assert result[0] is not None
    assert result[0].type == i32

    # check signature of emitted function call
    # int MPI_Comm_size(MPI_Comm comm, int *size)
    check_emitted_function_signature(
        ops,
        "MPI_Comm_size",
        (None, llvm.LLVMPointerType),
    )


def test_lower_mpi_send():
    buff, size, dtype, dest, tag = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType(), i32, i32
    ).results

    ops, result = lower_mpi.LowerMpiSend(info).lower(
        mpi.SendOp(buff, size, dtype, dest, tag)
    )
    """
    Check for function with signature like:
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)
    """

    # send has no return
    assert len(result) == 0

    check_emitted_function_signature(
        ops,
        "MPI_Send",
        (llvm.LLVMPointerType, type(i32), None, type(i32), type(i32), None),
    )


def test_lower_mpi_isend():
    ptr, count, dtype, dest, tag, req = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType(), i32, i32, mpi.RequestType()
    ).results

    ops, result = lower_mpi.LowerMpiIsend(info).lower(
        mpi.IsendOp(ptr, count, dtype, dest, tag, req)
    )
    """
    Check for function with signature like:
    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
    """

    # send has no return
    assert len(result) == 0

    check_emitted_function_signature(
        ops,
        "MPI_Isend",
        (
            llvm.LLVMPointerType,
            type(i32),
            None,
            type(i32),
            type(i32),
            None,
            mpi.RequestType,
        ),
    )


def test_lower_mpi_recv_no_status():
    buff, count, dtype, source, tag = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType(), i32, i32
    ).results
    """
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
    """

    ops, result = lower_mpi.LowerMpiRecv(info).lower(
        mpi.RecvOp(buff, count, dtype, source, tag, ignore_status=True)
    )

    assert len(result) == 0

    check_emitted_function_signature(
        ops,
        "MPI_Recv",
        (
            llvm.LLVMPointerType,
            type(i32),
            None,
            type(i32),
            type(i32),
            None,
            llvm.LLVMPointerType,
        ),
    )


def test_lower_mpi_recv_with_status():
    buff, count, dtype, source, tag = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType(), i32, i32
    ).results
    """
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
    """

    ops, result = lower_mpi.LowerMpiRecv(info).lower(
        mpi.RecvOp(buff, count, dtype, source, tag, ignore_status=False)
    )

    assert len(result) == 1

    check_emitted_function_signature(
        ops,
        "MPI_Recv",
        (
            llvm.LLVMPointerType,
            type(i32),
            None,
            type(i32),
            type(i32),
            None,
            llvm.LLVMPointerType,
        ),
    )


def test_lower_mpi_irecv():
    ptr, count, dtype, source, tag, req = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType(), i32, i32, mpi.RequestType()
    ).results
    """
    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
            int source, int tag, MPI_Comm comm, MPI_Request *request)
    """

    ops, result = lower_mpi.LowerMpiIrecv(info).lower(
        mpi.IrecvOp(ptr, count, dtype, source, tag, req)
    )

    # recv has no results
    assert len(result) == 0

    check_emitted_function_signature(
        ops,
        "MPI_Irecv",
        (
            llvm.LLVMPointerType,
            type(i32),
            None,
            type(i32),
            type(i32),
            None,
            mpi.RequestType,
        ),
    )


def test_lower_mpi_reduce():
    ptr, count, dtype, root = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType(), i32
    ).results
    """
    int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
    """

    ops, result = lower_mpi.LowerMpiReduce(info).lower(
        mpi.ReduceOp(ptr, ptr, count, dtype, mpi.MpiOp.MPI_SUM, root)
    )

    # reduce has no results
    assert len(result) == 0

    check_emitted_function_signature(
        ops,
        "MPI_Reduce",
        (
            llvm.LLVMPointerType,
            llvm.LLVMPointerType,
            type(i32),
            None,
            None,
            type(i32),
            None,
        ),
    )


def test_lower_mpi_all_reduce_no_send_buffer():
    ptr, count, dtype = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType()
    ).results
    """
    int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    """

    ops, result = lower_mpi.LowerMpiAllreduce(info).lower(
        mpi.AllreduceOp(None, ptr, count, dtype, mpi.MpiOp.MPI_SUM)
    )

    # allreduce has no results
    assert len(result) == 0

    check_emitted_function_signature(
        ops,
        "MPI_Allreduce",
        (
            type(i64),
            llvm.LLVMPointerType,
            type(i32),
            None,
            None,
            None,
        ),
    )


def test_mpi_waitall():
    _, count, _ = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType()
    ).results

    dummy = arith.ConstantOp.from_int_and_width(4, 32)
    alloc_request_op = mpi.AllocateTypeOp(mpi.RequestType, dummy)
    req_op: Operand = alloc_request_op.results[0]
    waitall = mpi.WaitallOp(req_op, count)

    assert waitall.operands[0] == req_op
    assert waitall.operands[1] == count
    assert len(waitall.results) == 0
    # TODO: enhance this test with a lowering


def test_lower_mpi_bcast():
    ptr, count, dtype, root = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType(), i32
    ).results
    """
    int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
              MPI_Comm comm)
    """

    ops, result = lower_mpi.LowerMpiBcast(info).lower(
        mpi.BcastOp(ptr, count, dtype, root)
    )

    # bcast has no results
    assert len(result) == 0

    check_emitted_function_signature(
        ops,
        "MPI_Bcast",
        (
            llvm.LLVMPointerType,
            type(i32),
            None,
            type(i32),
            None,
        ),
    )


def test_lower_mpi_allocate():
    (count,) = CreateTestValsOp.get(i32).results
    op = mpi.AllocateTypeOp(mpi.RequestType, count)

    ops, res = lower_mpi.LowerMpiAllocateType(info).lower(op)

    assert len(res) == 1
    assert len(ops) == 1
    assert isinstance(ops[0], llvm.AllocaOp)


def test_lower_mpi_vec_get():
    mod = builtin.ModuleOp(
        [
            count := CreateTestValsOp.get(i32),
            vec := mpi.AllocateTypeOp(mpi.RequestType, count),
            get := mpi.VectorGetOp(vec, count),
        ]
    )
    # we have to apply this rewrite to that the argument type of the `get`
    # becomes an llvm.ptr
    PatternRewriteWalker(lower_mpi.LowerMpiAllocateType(info)).rewrite_module(mod)

    ops, res = lower_mpi.LowerMpiVectorGet(info).lower(get)

    assert len(res) == 1
    assert res[0] is not None
    assert isinstance(res[0].type, llvm.LLVMPointerType)
    assert len(ops) > 0


def test_lower_mpi_gather():
    ptr, count, dtype, root = CreateTestValsOp.get(
        llvm.LLVMPointerType(), i32, mpi.DataType(), i32
    ).results

    ops, result = lower_mpi.LowerMpiGatherOp(info).lower(
        mpi.GatherOp(ptr, count, dtype, ptr, count, dtype, root)
    )

    # gather has no results
    assert len(result) == 0

    check_emitted_function_signature(
        ops,
        "MPI_Gather",
        (
            llvm.LLVMPointerType,
            type(i32),
            None,
            llvm.LLVMPointerType,
            type(i32),
            None,
            type(i32),
            None,
        ),
    )


def test_mpi_type_conversion():
    """
    Test that each builtin datatype is correctly mapped to an MPI datatype

    """
    info = lower_mpi.MpiLibraryInfo(
        MPI_UNSIGNED_CHAR=1,
        MPI_UNSIGNED_SHORT=2,
        MPI_UNSIGNED=3,
        MPI_UNSIGNED_LONG_LONG=4,
        MPI_CHAR=5,
        MPI_SHORT=6,
        MPI_INT=7,
        MPI_LONG_LONG_INT=8,
    )

    lowering = lower_mpi.LowerMpiRecv(info)

    from xdsl.dialects.builtin import IntegerType, Signedness, f32, f64, i32, i64

    u64 = IntegerType(64, Signedness.UNSIGNED)
    u32 = IntegerType(32, Signedness.UNSIGNED)

    checks = [
        (f32, info.MPI_FLOAT),
        (f64, info.MPI_DOUBLE),
        (i32, info.MPI_INT),
        (u32, info.MPI_UNSIGNED),
        (i64, info.MPI_LONG_LONG_INT),
        (u64, info.MPI_UNSIGNED_LONG_LONG),
    ]

    for width in (8, 16):
        for sign in (Signedness.UNSIGNED, Signedness.SIGNLESS, Signedness.SIGNED):
            sign_str = "UNSIGNED_" if sign == Signedness.UNSIGNED else ""
            name = "CHAR" if width == 8 else "SHORT"
            mpi_type = IntegerType(width, sign)
            checks.append((mpi_type, getattr(info, f"MPI_{sign_str}{name}")))

    for type, target in checks:
        # we test a private member function here, so we ignore pyright
        translate_to_mpi_type = (
            lowering._translate_to_mpi_type  # pyright: ignore[reportPrivateUsage]
        )
        assert translate_to_mpi_type(type) == target
