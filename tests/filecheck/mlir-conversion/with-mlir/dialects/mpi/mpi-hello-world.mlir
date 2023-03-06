// RUN: xdsl-opt %s -t mlir -p lower-mpi | mlir-opt --convert-func-to-llvm --finalize-memref-to-llvm --reconcile-unrealized-casts | filecheck %s

"builtin.module"() ({
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
      "scf.yield"() : () -> ()
    }) : (i1) -> ()
    "mpi.finalize"() : () -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()

// we don't really care about the whole structure, we just want to make sure mlir-opt can lower all this down to llvm

// CHECK: llvm.call @MPI_Init({{%\d+}}, {{%\d+}}) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK: llvm.call @MPI_Comm_rank({{%\d+}}, {{%\d+}}) : (i32, !llvm.ptr<i32>) -> i32
// CHECK: llvm.call @MPI_Send({{%\d+}}, {{%\d+}}, {{%\d+}}, {{%\d+}}, {{%\d+}}, {{%\d+}}) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
// CHECK: llvm.call @MPI_Recv({{%\d+}}, {{%\d+}}, {{%\d+}}, {{%\d+}}, {{%\d+}}, {{%\d+}}, {{%\d+}}) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

// also check that external funcs were declared correctly:

// CHECK: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Comm_rank(i32, !llvm.ptr<i32>) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Send(!llvm.ptr, i32, i32, i32, i32, i32) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Recv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32 attributes {sym_visibility = "private"}
// CHECK: llvm.func @MPI_Finalize() -> i32 attributes {sym_visibility = "private"}
