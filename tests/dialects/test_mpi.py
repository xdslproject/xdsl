from xdsl.dialects import memref, mpi
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import f64, i32


def test_mpi_baseop():
    """
    This test is used to track changes in `.get` and other accessors
    """
    alloc0 = memref.Alloc.get(f64, 32, [100, 14, 14])
    dest = Constant.from_int_and_width(1, i32)
    unwrap = mpi.UnwrapMemrefOp(alloc0)
    req_vec = mpi.AllocateTypeOp(mpi.RequestType, dest)
    req_obj = mpi.VectorGetOp(req_vec, dest)
    tag = Constant.from_int_and_width(1, i32)
    send = mpi.Isend(unwrap.ptr, unwrap.len, unwrap.type, dest, tag, req_obj)
    wait = mpi.Wait(send.request, ignore_status=False)
    recv = mpi.Irecv(unwrap.ptr, unwrap.len, unwrap.type, dest, tag, req_obj)
    test_res = mpi.Test(recv.request)
    assert wait.status is not None
    source = mpi.GetStatusField(wait.status, mpi.StatusTypeField.MPI_SOURCE)

    assert unwrap.ref == alloc0.memref
    assert send.buffer == unwrap.ptr
    assert send.count == unwrap.len
    assert send.datatype == unwrap.type
    assert send.dest == dest.result
    assert send.tag == tag.result
    assert wait.request == send.request
    assert recv.buffer == unwrap.ptr
    assert recv.count == unwrap.len
    assert recv.datatype == unwrap.type
    assert recv.source == dest.result
    assert recv.tag == tag.result
    assert test_res.request == recv.request
    assert source.status == wait.status
    assert source.field.string == mpi.StatusTypeField.MPI_SOURCE.value
