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
    req_vec = mpi.AllocateTypeOp.get(mpi.RequestType, dest)
    req_obj = mpi.VectorGetOp.get(req_vec, dest)
    tag = Constant.from_int_and_width(1, i32)
    send = mpi.Isend.get(unwrap.ptr, unwrap.len, unwrap.typ, dest, tag, req_obj)
    wait = mpi.Wait.get(send.request, ignore_status=False)
    recv = mpi.Irecv.get(unwrap.ptr, unwrap.len, unwrap.typ, dest, tag, req_obj)
    test_res = mpi.Test.get(recv.request)
    assert wait.status is not None
    source = mpi.GetStatusField.get(wait.status, mpi.StatusTypeField.MPI_SOURCE)

    assert unwrap.ref == alloc0.memref
    assert send.buffer == unwrap.ptr
    assert send.count == unwrap.len
    assert send.datatype == unwrap.typ
    assert send.dest == dest.result
    assert send.tag == tag.result
    assert wait.request == send.request
    assert recv.buffer == unwrap.ptr
    assert recv.count == unwrap.len
    assert recv.datatype == unwrap.typ
    assert recv.source == dest.result
    assert recv.tag == tag.result
    assert test_res.request == recv.request
    assert source.status == wait.status
    assert source.field.data == mpi.StatusTypeField.MPI_SOURCE.value
