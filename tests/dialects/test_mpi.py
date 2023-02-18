from conftest import assert_print_op

from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import f64, ModuleOp
from xdsl.dialects.memref import Alloc
from xdsl.dialects.mpi import (ISend, IRecv, Test, Wait, t_int, GetStatusField,
                               StatusTypeField)

from xdsl.ir import Region
from xdsl.printer import Printer


def test_mpi_combo():
    printer = Printer(target=Printer.Target.MLIR)

    region0 = Region.from_operation_list([
        memref := Alloc.get(f64, 32, [100, 14, 14]),
        dest := Constant.from_int_and_width(1, t_int),
        req := ISend.get(memref, dest, 1),
        res := IRecv.get(dest, memref.results[0].typ, 1),
        test_res := Test.get(res.request),
        tag := GetStatusField.get(test_res.status, StatusTypeField.MPI_TAG),
        src := GetStatusField.get(test_res.status, StatusTypeField.MPI_SOURCE),
        err := GetStatusField.get(test_res.status, StatusTypeField.MPI_ERROR),
        code2 := Wait.get(res.request)
    ])  # yapf: disable

    expected = \
        """"builtin.module"() ({
  %0 = "memref.alloc"() {"alignment" = 32 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<100x14x14xf64>
  %1 = "arith.constant"() {"value" = 1 : si32} : () -> si32
  %2 = "mpi.isend"(%0, %1) {"tag" = 1 : si32} : (memref<100x14x14xf64>, si32) -> #mpi.request
  %3, %4 = "mpi.irecv"(%1) {"tag" = 1 : si32} : (si32) -> (memref<100x14x14xf64>, #mpi.request)
  %5, %6 = "mpi.test"(%4) : (#mpi.request) -> (i1, #mpi.status)
  %7 = "mpi.status.get"(%6) {"field" = "MPI_TAG"} : (#mpi.status) -> si32
  %8 = "mpi.status.get"(%6) {"field" = "MPI_SOURCE"} : (#mpi.status) -> si32
  %9 = "mpi.status.get"(%6) {"field" = "MPI_ERROR"} : (#mpi.status) -> si32
  %10 = "mpi.wait"(%4) : (#mpi.request) -> si32
}) : () -> ()
"""

    op = ModuleOp.from_region_or_ops(region0)
    assert_print_op(op, expected, target=Printer.Target.MLIR, diagnostic=None)


def test_mpi_baseop():
    alloc0 = Alloc.get(f64, 32, [100, 14, 14])
    dest = Constant.from_int_and_width(1, t_int)
    send = ISend.get(alloc0, dest, 1)
    recv = IRecv.get(dest, alloc0.memref.typ, 1)
    test_res = Test.get(recv.request)
    code2 = Wait.get(recv.request)

    assert send.operands[0] is alloc0.results[0]
    assert send.operands[1] is dest.results[0]
    assert recv.operands[0] is send.operands[1]
    assert code2.operands[0] is recv.results[1]
    assert test_res.operands[0] is recv.results[1]
