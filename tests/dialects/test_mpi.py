from conftest import assert_print_op

from xdsl.dialects.arith import Constant
from xdsl.dialects.builtin import f64, ModuleOp
from xdsl.dialects.memref import Alloc
from xdsl.dialects.mpi import (ISend, IRecv, Test, Wait, t_int, GetStatusField,
                               StatusTypeField)
from xdsl.dialects import func, arith, mpi, scf, memref

from xdsl.ir import Region
from xdsl.printer import Printer


def test_mpi_combo():
    printer = Printer(target=Printer.Target.MLIR)

    module = ModuleOp.from_region_or_ops([
    func.FuncOp.from_callable('main', [], [], lambda: [
        mpi.Init.build(),
        rank := mpi.CommRank.get(),
        lit0 := arith.Constant.from_int_and_width(0, 32),
        cond := arith.Cmpi.from_mnemonic(rank, lit0, 'eq'),
        buff := memref.Alloc.get(f64, 32, [100, 14, 14]),
        scf.If.get(cond, [], [  # if rank == 0
            dest := arith.Constant.from_int_and_width(1, mpi.t_int),
            mpi.Send.get(buff, dest, 1),
            # mpi.Wait.get(req, ignore_status=False),
            scf.Yield.get(),
        ], [  # else
            source := arith.Constant.from_int_and_width(0, mpi.t_int),
            status := mpi.Recv.get(source, buff, 1, ignore_status=False),
            GetStatusField.get(status, StatusTypeField.MPI_TAG),
            #mpi.Wait.get(recv.request),
            scf.Yield.get(),
        ]),
        mpi.Finalize.build(),
        func.Return.get()
    ])
])  # yapf: disable

    expected = \
        """"builtin.module"() ({
  "func.func"() ({
    "mpi.init"() : () -> ()
    %0 = "mpi.comm.rank"() : () -> i32
    %1 = "arith.constant"() {"value" = 0 : i32} : () -> i32
    %2 = "arith.cmpi"(%0, %1) {"predicate" = 0 : i64} : (i32, i32) -> i1
    %3 = "memref.alloc"() {"alignment" = 32 : i64, "operand_segment_sizes" = array<i32: 0, 0>} : () -> memref<100x14x14xf64>
    "scf.if"(%2) ({
      %4 = "arith.constant"() {"value" = 1 : i32} : () -> i32
      "mpi.send"(%3, %4) {"tag" = 1 : i32} : (memref<100x14x14xf64>, i32) -> ()
      "scf.yield"() : () -> ()
    }, {
      %5 = "arith.constant"() {"value" = 0 : i32} : () -> i32
      %6 = "mpi.recv"(%5, %3) {"tag" = 1 : i32} : (i32, memref<100x14x14xf64>) -> !mpi.status
      %7 = "mpi.status.get"(%6) {"field" = "MPI_TAG"} : (!mpi.status) -> i32
      "scf.yield"() : () -> ()
    }) : (i1) -> ()
    "mpi.finalize"() : () -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()
"""

    assert_print_op(module, expected, target=Printer.Target.MLIR, diagnostic=None)


def test_mpi_baseop():
    alloc0 = Alloc.get(f64, 32, [100, 14, 14])
    dest = Constant.from_int_and_width(1, t_int)
    send = ISend.get(alloc0, dest, 1)
    recv = IRecv.get(dest, alloc0.memref, 1)
    test_res = Test.get(recv.request)
    code2 = Wait.get(recv.request)

    assert send.operands[0] is alloc0.results[0]
    assert send.operands[1] is dest.results[0]
    assert recv.operands[0] is send.operands[1]
    assert code2.operands[0] is recv.results[0]
    assert test_res.operands[0] is recv.results[0]
