from xdsl.dialects import mpi, memref
from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import f64, i32


def test_mpi_baseop():
    alloc0 = memref.Alloc.get(f64, 32, [100, 14, 14])
    dest = Constant.from_int_and_width(1, i32)
    send = mpi.ISend.get(alloc0, dest, 1)
    recv = mpi.IRecv.get(dest, alloc0.memref, 1)
    test_res = mpi.Test.get(recv.request)
    code2 = mpi.Wait.get(recv.request)

    assert send.operands[0] is alloc0.results[0]
    assert send.operands[1] is dest.results[0]
    assert recv.operands[0] is send.operands[1]
    assert code2.operands[0] is recv.results[0]
    assert test_res.operands[0] is recv.results[0]
