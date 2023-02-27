from xdsl.dialects import mpi, func, llvm, builtin
from xdsl.ir import Operation, Attribute, OpResult, SSAValue
from xdsl.irdl import irdl_op_definition, VarOpResult

lowering = mpi.MpiLowerings(mpi.MpiLibraryInfo())


def extract_func_call(ops: list[Operation],
                      name: str = 'MPI_') -> func.Call | None:
    for op in ops:
        if isinstance(op,
                      func.Call) and op.callee.string_value().startswith(name):
            return op


def check_emitted_function_signature(ops: list[Operation],
                                     name: str,
                                     types: tuple[type[Attribute] | None, ...],
                                     ignore_list: tuple[SSAValue] = tuple()):
    call = extract_func_call(ops, name)
    assert call is not None, f"Missing func.Call op to {name} in output!"
    assert len(call.arguments) == len(types)
    for i, type_ in enumerate(types):
        arg = call.arguments[i]

        # check that the argument type is correct (if constraint present)
        if type_ is not None:
            assert isinstance(
                arg.typ, type_
            ), f"Expected argument to be of type {type_} (got {arg.typ} instead)"

        # check that all arguments originate in the emitted operations, except for the exceptions
        if arg in ignore_list:
            continue

        assert isinstance(arg, OpResult)
        assert arg.op in ops, f"Expected {arg.op} to be present in the emitted operations!"


@irdl_op_definition
class TestOp(Operation):
    name = "testing.test"
    result: VarOpResult

    @staticmethod
    def get(*types: Attribute):
        return TestOp.build(result_types=[list(types)])


def test_lower_mpi_init():
    ops, result = lowering.lower_mpi_init(mpi.Init.build())

    assert len(result) == 0
    assert len(ops) == 2

    nullop, call = ops

    assert isinstance(call, func.Call)
    assert isinstance(nullop, llvm.NullOp)
    assert call.callee.string_value() == 'MPI_Init'
    assert len(call.arguments) == 2
    assert all(arg == nullop.nullptr for arg in call.arguments)


def test_lower_mpi_finalize():
    ops, result = lowering.lower_mpi_finalize(mpi.Finalize.build())

    assert len(result) == 0
    assert len(ops) == 1

    call, = ops

    assert isinstance(call, func.Call)
    assert call.callee.string_value() == 'MPI_Finalize'
    assert len(call.arguments) == 0


def test_lower_mpi_wait_no_status():
    request, = TestOp.get(mpi.RequestType()).results

    ops, result = lowering.lower_mpi_wait(mpi.Wait.get(request))

    assert len(result) == 0
    call = extract_func_call(ops)
    assert call is not None
    assert call.callee.string_value() == 'MPI_Wait'
    assert len(call.arguments) == 2


def test_lower_mpi_wait_with_status():
    request, = TestOp.get(mpi.RequestType()).results

    ops, result = lowering.lower_mpi_wait(
        mpi.Wait.get(request, ignore_status=False))

    assert len(result) == 1
    assert isinstance(result[0].typ, llvm.LLVMPointerType)
    call = extract_func_call(ops)
    assert call is not None
    assert call.callee.string_value() == 'MPI_Wait'
    assert len(call.arguments) == 2
    assert isinstance(call.arguments[1], OpResult)
    assert isinstance(call.arguments[1].op, llvm.AllocaOp)


def test_lower_mpi_comm_rank():
    ops, result = lowering.lower_mpi_comm_rank(mpi.CommRank.get())

    assert len(result) == 1
    assert result[0].typ == mpi.t_int

    # check signature of emitted function call
    # int MPI_Comm_rank(MPI_Comm comm, int *rank)
    check_emitted_function_signature(
        ops,
        'MPI_Comm_rank',
        (None, llvm.LLVMPointerType),
    )


def test_lower_mpi_send():
    buff, dest = TestOp.get(
        mpi.MemRefType.from_element_type_and_shape(builtin.f64, [32, 32, 32]),
        mpi.t_int).results

    ops, result = lowering.lower_mpi_send(mpi.Send.get(buff, dest, 1))
    """
    Check for function with signature like:
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)
    """

    # send has no return
    assert len(result) == 0

    check_emitted_function_signature(
        ops, 'MPI_Send', (llvm.LLVMPointerType, type(mpi.t_int), None,
                          type(mpi.t_int), type(mpi.t_int), None), (dest, ))


def test_lower_mpi_isend():
    buff, dest = TestOp.get(
        mpi.MemRefType.from_element_type_and_shape(builtin.f64, [32, 32, 32]),
        mpi.t_int).results

    ops, result = lowering.lower_mpi_isend(mpi.ISend.get(buff, dest, 1))
    """
    Check for function with signature like:
    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
    """

    # send has no return
    assert len(result) == 1

    check_emitted_function_signature(
        ops, 'MPI_Isend',
        (llvm.LLVMPointerType, type(mpi.t_int), None, type(
            mpi.t_int), type(mpi.t_int), None, llvm.LLVMPointerType), (dest, ))


def test_lower_mpi_recv_no_status():
    buff, source = TestOp.get(
        mpi.MemRefType.from_element_type_and_shape(builtin.f64, [32, 32, 32]),
        mpi.t_int).results
    """
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
    """

    ops, result = lowering.lower_mpi_recv(
        mpi.Recv.get(source, buff, tag=3, ignore_status=True))

    assert len(result) == 0

    check_emitted_function_signature(
        ops, 'MPI_Recv',
        (llvm.LLVMPointerType, type(mpi.t_int), None, type(mpi.t_int),
         type(mpi.t_int), None, llvm.LLVMPointerType), (source, ))


def test_lower_mpi_recv_with_status():
    buff, source = TestOp.get(
        mpi.MemRefType.from_element_type_and_shape(builtin.f64, [32, 32, 32]),
        mpi.t_int).results
    """
    int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag,
             MPI_Comm comm, MPI_Status *status)
    """

    ops, result = lowering.lower_mpi_recv(
        mpi.Recv.get(source, buff, tag=3, ignore_status=False))

    assert len(result) == 1

    check_emitted_function_signature(
        ops, 'MPI_Recv',
        (llvm.LLVMPointerType, type(mpi.t_int), None, type(mpi.t_int),
         type(mpi.t_int), None, llvm.LLVMPointerType), (source, ))


def test_lower_mpi_irecv():
    buff, source = TestOp.get(
        mpi.MemRefType.from_element_type_and_shape(builtin.f64, [32, 32, 32]),
        mpi.t_int).results
    """
    int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
            int source, int tag, MPI_Comm comm, MPI_Request *request)
    """

    ops, result = lowering.lower_mpi_irecv(mpi.IRecv.get(source, buff, tag=3))

    assert len(result) == 1

    check_emitted_function_signature(
        ops, 'MPI_Irecv',
        (llvm.LLVMPointerType, type(mpi.t_int), None, type(mpi.t_int),
         type(mpi.t_int), None, llvm.LLVMPointerType), (source, ))


def test_mpi_type_conversion():
    """
    Test that each builtin datatype is correctly mapped to an MPI datatype

    """
    lowering = mpi.MpiLowerings(
        mpi.MpiLibraryInfo(
            MPI_UNSIGNED_CHAR=1,
            MPI_UNSIGNED_SHORT=2,
            MPI_UNSIGNED=3,
            MPI_UNSIGNED_LONG_LONG=4,
            MPI_CHAR=5,
            MPI_SHORT=6,
            MPI_INT=7,
            MPI_LONG_LONG_INT=8,
        ))
    from xdsl.dialects.builtin import f64, f32, IntegerType, i32, i64, Signedness
    u64 = IntegerType.from_width(64, Signedness.UNSIGNED)
    u32 = IntegerType.from_width(32, Signedness.UNSIGNED)

    checks = [
        (f32, lowering.info.MPI_FLOAT),
        (f64, lowering.info.MPI_DOUBLE),
        (i32, lowering.info.MPI_INT),
        (u32, lowering.info.MPI_UNSIGNED),
        (i64, lowering.info.MPI_LONG_LONG_INT),
        (u64, lowering.info.MPI_UNSIGNED_LONG_LONG),
    ]

    for width in (8, 16):
        for sign in (Signedness.UNSIGNED, Signedness.SIGNLESS,
                     Signedness.SIGNED):
            sign_str = 'UNSIGNED_' if sign == Signedness.UNSIGNED else ''
            name = 'CHAR' if width == 8 else 'SHORT'
            typ = IntegerType.from_width(width, sign)
            checks.append((typ, getattr(lowering.info,
                                        f'MPI_{sign_str}{name}')))

    for type, target in checks:
        # we test a private member function here, so we need to tell pyright that that's okay
        assert lowering._translate_to_mpi_type(type) == target  # type: ignore
