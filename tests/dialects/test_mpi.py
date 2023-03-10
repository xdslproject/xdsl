from xdsl.dialects import mpi, memref
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import f64, i32


def test_mpi_baseop():
    """
    This test is used to track changes in `.get` and other accessors
    """
    alloc0 = memref.Alloc.get(f64, 32, [100, 14, 14])
    dest = Constant.from_int_and_width(1, i32)
    unwrap = mpi.UnwrapMemrefOp.get(alloc0)
    tag = Constant.from_int_and_width(1, i32)
    send = mpi.ISend.get(unwrap.ptr, unwrap.len, unwrap.typ, dest, tag)
    wait = mpi.Wait.get(send.request, ignore_status=False)
    recv = mpi.IRecv.get(unwrap.ptr, unwrap.len, unwrap.typ, dest, tag)
    test_res = mpi.Test.get(recv.request)
    source = mpi.GetStatusField.get(wait.status,
                                    mpi.StatusTypeField.MPI_SOURCE)
