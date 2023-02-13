from conftest import assert_print_op
from xdsl.dialects.mpi import MPIRequestOp, MPIRequest, MPI_Wait
from xdsl.dialects.builtin import ModuleOp


def test_MPIRequest():
    # This test constructs and MPIRequest and feeds it as input to
    # an MPI_Wait operation
    request = MPIRequest()
    request_op = MPIRequestOp.from_attr(request)
    mpiwait = MPI_Wait.from_callable(request_op)

    assert request_op.results[0].typ is request
    assert mpiwait.operands[0] is request_op.results[0]

    op0 = ModuleOp.from_region_or_ops([request_op, mpiwait])

    expected0 = \
    """
    builtin.module() {
  %0 : !MPI_Request = mpi.MPIRequestOp()
  %1 : !MPI_Status = mpi.MPI_Wait(%0 : !MPI_Request)
}
    """

    # TOFIX
    # Check type
    assert_print_op(op0, expected0, None)
