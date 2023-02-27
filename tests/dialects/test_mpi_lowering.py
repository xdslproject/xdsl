from xdsl.dialects import mpi, func, llvm, builtin
from xdsl.ir import Operation, Attribute, OpResult
from xdsl.irdl import irdl_op_definition, VarOpResult

lowering = mpi.MpiLowerings(mpi.MpiLibraryInfo())


def extract_func_call(ops: list[Operation]) -> func.Call | None:
    for op in ops:
        if isinstance(op, func.Call) and op.callee.string_value().startswith('MPI_'):
            return op


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

    ops, result = lowering.lower_mpi_wait(mpi.Wait.get(request, ignore_status=False))

    assert len(result) == 1
    assert isinstance(result[0].typ, llvm.LLVMPointerType)
    call = extract_func_call(ops)
    assert call is not None
    assert call.callee.string_value() == 'MPI_Wait'
    assert len(call.arguments) == 2
    assert isinstance(call.arguments[1], OpResult)
    assert isinstance(call.arguments[1].op, llvm.AllocaOp)



def test_lower_mpi_send():
    buff, dest = TestOp.get(mpi.MemRefType.from_element_type_and_shape(builtin.f64, [32,32,32]), mpi.t_int).results

    ops, result = lowering.lower_mpi_send(mpi.Send.get(buff, dest, 1))

    """
    Check for function with signature like:
    int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)
    """

    # send has no return
    assert len(result) == 0

    # get call op
    call = extract_func_call(ops)
    assert call is not None
    assert call.callee.string_value() == 'MPI_Send'
    assert len(call.arguments) == 6

    # check arg types
    types = (llvm.LLVMPointerType, type(mpi.t_int), None, type(mpi.t_int), type(mpi.t_int), None)
    for i, type_ in enumerate(types):
        arg = call.arguments[i]
        assert isinstance(arg, OpResult)

        # check that all arguments originate in the emitted operations
        if arg != dest:  # (except for the dest arg)
            assert arg.op in ops

        if type_ is None:  # if no type info is given, that's enough
            continue

        assert isinstance(arg.typ, type_)



def test_lower_mpi_isend():
    buff, dest = TestOp.get(mpi.MemRefType.from_element_type_and_shape(builtin.f64, [32,32,32]), mpi.t_int).results

    ops, result = lowering.lower_mpi_isend(mpi.ISend.get(buff, dest, 1))

    """
    Check for function with signature like:
    int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
              int tag, MPI_Comm comm, MPI_Request *request)
    """

    # send has no return
    assert len(result) == 1

    # get call op
    call = extract_func_call(ops)
    assert call is not None
    assert call.callee.string_value() == 'MPI_Isend'
    assert len(call.arguments) == 7

    # check arg types
    types = (llvm.LLVMPointerType, type(mpi.t_int), None, type(mpi.t_int), type(mpi.t_int), None, llvm.LLVMPointerType)
    for i, type_ in enumerate(types):
        arg = call.arguments[i]
        assert isinstance(arg, OpResult)

        # check that all arguments originate in the emitted operations
        if arg != dest:  # (except for the dest arg)
            assert arg.op in ops

        if type_ is None:  # if no type info is given, that's enough
            continue

        assert isinstance(arg.typ, type_)

