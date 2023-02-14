from conftest import assert_print_op

from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import f64, ModuleOp
from xdsl.dialects.memref import Alloc
from xdsl.dialects.mpi import (ISend, IRecv, Test, StatusGetFlag,
                               StatusGetStatus, Wait, t_int)

from xdsl.ir import Region
from xdsl.printer import Printer


def test_mpi_combo():
    printer = Printer(target=Printer.Target.MLIR)

    # yapf: ignore
    region0 = Region.from_operation_list([
        memref := Alloc.get(f64, 32, [100, 14, 14]),
        dest := Constant.from_int_and_width(1, t_int),
        req := ISend.get(memref, dest, 1),
        res := IRecv.get(dest, memref.results[0].typ, 1),
        test_res := Test.get(res.results[1]),
        flag := StatusGetFlag.get(test_res.results[1]),
        code := StatusGetStatus.get(test_res.results[1]),
        code2 := Wait.get(res.results[1])
    ])  # yapf: disable

    expected = \
""""builtin.module"() ({
  %0 = "memref.alloc"() {"alignment" = 32 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<100x14x14xf64>
  %1 = "arith.constant"() {"value" = 1 : si32} : () -> si32
  %2 = "mpi.isend"(%0, %1) {"tag" = 1 : si32} : (memref<100x14x14xf64>, si32) -> #mpi.request
  %3, %4 = "mpi.irecv"(%1) {"tag" = 1 : si32} : (si32) -> (memref<100x14x14xf64>, #mpi.request)
  %5, %6 = "mpi.test"(%4) : (#mpi.request) -> (i1, si32)
  %7 = "mpi.status_get_flag"(%6) : (si32) -> i1
  %8 = "mpi.status_get_status"(%6) : (si32) -> si32
  %9 = "mpi.wait"(%4) : (#mpi.request) -> si32
}) : () -> ()
"""

    op = ModuleOp.from_region_or_ops(region0)
    assert_print_op(op, expected, target=Printer.Target.MLIR, diagnostic=None)
    printer.print_op(op)


"""
// [WIP] Example isend
// %in is the input memref
// %dest is a destination rank (si32)
%request = "mpi.isend"(%in, %dest) {"tag" = 1} : (!memref<3x2x2xi64>, !si32) -> (!mpi.request) 


// example irecv
// %source is the source rank (si32)
%data, %request = "mpi.irecv"(%source) {"tag" = 1} : (!si32) -> (!memref<3x2x2xi64>, !mpi.request)

// example test
// %request is an !mpi.request
%status_obj = "mpi.test"(%request) : (!mpi.request) -> !mpi.status
%flag = "mpi.get_status_flag"(%status_obj) : (!mpi.status) -> i1
%status = "mpi.get_status_code"(%status_obj) : (!mpi.status) -> si32

// example wait
// %request is an !mpi.request
%status = "mpi.wait"(%request) : (!mpi.request) -> si32
"""


def test_mpi_baseop():

    alloc0 = Alloc.get(f64, 32, [100, 14, 14])
    dest = Constant.from_int_and_width(1, t_int)
    send = ISend.get(alloc0, dest, 1)
    recv = IRecv.get(dest, alloc0.results[0].typ, 1)
    test_res = Test.get(recv.results[1])
    status_flag = StatusGetFlag.get(test_res.results[1])
    status = StatusGetStatus.get(test_res.results[1])
    code2 = Wait.get(recv.results[1])

    assert send.operands[0] is alloc0.results[0]
    assert send.operands[1] is dest.results[0]
    assert recv.operands[0] is send.operands[1]
    assert status_flag.operands[0] is test_res.results[1]
    assert status.operands[0] is test_res.results[1]
    assert code2.operands[0] is recv.results[1]
